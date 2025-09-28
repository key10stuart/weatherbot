#!/usr/bin/env python3
# train_cnn_forecaster.py
# Starter path: H=24, L=168, features [sin/cos HOD, sin/cos DOY, temp]
# Normalize per-feature with train μ/σ; apply to all splits.
# CNN: k=5, B=6, dilations [1,2,4,8,16,32], C=64, GELU, LayerNorm, dropout=0.1, residual blocks.
# Loss: Huber; Optim: AdamW lr=3e-3, wd 1e-2, OneCycle, bs=128, epochs=80, clip=1.0.
# Early stop on val RMSE; report per-horizon metrics & skill vs persistence.

import argparse, math, os, sys, time, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Utils
# ---------------------------
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    # CUDA-specific seeding and cuDNN settings removed per request.

def find_array(d, names):
    for n in names:
        if n in d: return d[n]
    lower = {k.lower(): k for k in d.files}
    for n in names:
        if n.lower() in lower: return d[lower[n.lower()]]
    raise KeyError(f"Could not find any of {names} in NPZ keys: {list(d.files)}")

def load_npz(path):
    npz = np.load(path)
    X_train = find_array(npz, ["X_train","train_x","x_train"])
    y_train = find_array(npz, ["y_train","train_y","Y_train"])
    X_val   = find_array(npz, ["X_val","val_x","x_val"])
    y_val   = find_array(npz, ["y_val","val_y","Y_val"])
    X_test  = find_array(npz, ["X_test","test_x","x_test"])
    y_test  = find_array(npz, ["y_test","test_y","Y_test"])
    if y_train.ndim == 3 and y_train.shape[-1] == 1:
        y_train = y_train[...,0]; y_val = y_val[...,0]; y_test = y_test[...,0]
    return (X_train, y_train, X_val, y_val, X_test, y_test)

def zscore_fit(X):  # X: [N, L, F]
    mu = X.reshape(-1, X.shape[-1]).mean(axis=0)
    sd = X.reshape(-1, X.shape[-1]).std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)

def zscore_apply(X, mu, sd):
    return (X - mu) / sd

class TSWindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)  # [N,L,F]
        self.y = torch.as_tensor(y, dtype=torch.float32)  # [N,H]
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

# ---------------------------
# Device selection & autotune
# ---------------------------
def pick_device():
    # Prefer MPS (Apple Silicon). Otherwise, CPU. (No CUDA.)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def autotune_hparams(args, device):
    """
    Heuristics:
      - MPS (Apple Silicon): good throughput; keep defaults, bump workers a bit.
      - CPU (non-M3 assumed ~16 GB RAM): clamp to a 16 GB profile without prompting.
    """
    tuned = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "channels": args.channels,
        "num_workers": args.num_workers,
        "lr": args.lr,
    }

    if device.type == "mps":
        tuned["num_workers"] = max(tuned["num_workers"], 4)
        return tuned

    # CPU path: assume ~16 GB RAM per instruction
    tuned["batch_size"] = min(args.batch_size, 64)
    tuned["channels"]   = min(args.channels, 48)
    tuned["epochs"]     = min(args.epochs, 80)
    tuned["num_workers"]= max(args.num_workers, 2)
    tuned["lr"]         = min(args.lr, 2e-3)
    return tuned

# ---------------------------
# Model
# ---------------------------
class CausalConv1d(nn.Conv1d):
    """Left-pad only so no future leakage."""
    def __init__(self, in_ch, out_ch, k, dilation=1):
        super().__init__(in_ch, out_ch, kernel_size=k, dilation=dilation, padding=0, bias=True)
        self.left_pad = (k - 1) * dilation
    def forward(self, x):  # x: [B,C,T]
        if self.left_pad > 0:
            x = torch.nn.functional.pad(x, (self.left_pad, 0))
        return super().forward(x)

class LayerNormChannel(nn.Module):
    """LayerNorm over channel dim at each timestep (expects input [B,C,T])."""
    def __init__(self, C):
        super().__init__()
        self.ln = nn.LayerNorm(C)

    def forward(self, x):  # x: [B,C,T]
        x = x.transpose(1, 2).contiguous()   # [B,T,C]
        x = self.ln(x)
        x = x.transpose(1, 2).contiguous()   # back to [B,C,T]
        return x

class ResBlock(nn.Module):
    def __init__(self, C, k, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(C, C, k, dilation)
        self.act1  = nn.GELU()
        self.norm1 = LayerNormChannel(C)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(C, C, k, dilation)
        self.act2  = nn.GELU()
        self.norm2 = LayerNormChannel(C)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):  # x: [B,C,T]
        residual = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.norm1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.norm2(x)
        x = self.drop2(x)

        return x + residual  # residual connection

class DilatedCausalCNN(nn.Module):
    def __init__(self, in_feats, C=64, k=5, dilations=(1,2,4,8,16,32), horizon=24, dropout=0.1):
        super().__init__()
        self.in_feats = in_feats
        self.in_proj = nn.Conv1d(in_feats, C, kernel_size=1)
        self.blocks = nn.ModuleList([ResBlock(C, k=k, dilation=d, dropout=dropout) for d in dilations])
        self.head_norm = LayerNormChannel(C)
        self.head = nn.Linear(C, horizon)  # last-timestep embedding -> H outputs

    def forward(self, x):  # x: [B,L,F]
        x = x.transpose(1, 2).contiguous()  # [B,F,L]
        x = self.in_proj(x)                 # [B,C,L]
        for b in self.blocks:
            x = b(x)                        # keep [B,C,L]
        x = self.head_norm(x)               # [B,C,L]
        last = x[:, :, -1].contiguous()     # [B,C]
        yhat = self.head(last)              # [B,H]
        return yhat

# ---------------------------
# Metrics
# ---------------------------
def rmse(a, b, dim=None, eps=1e-9):
    return torch.sqrt(torch.mean((a - b)**2, dim=dim) + eps)

def persistence_baseline(x_last, H):
    return x_last.unsqueeze(1).repeat(1, H)

def per_horizon_rmse(yhat, y):
    err = (yhat - y) ** 2
    ph = torch.sqrt(err.mean(dim=0))
    overall = torch.sqrt(err.mean())
    return ph, overall

def skill_vs_persistence(yhat, y, y_last):
    base = persistence_baseline(y_last, y.shape[1])
    ph_m, overall_m = per_horizon_rmse(yhat, y)
    ph_b, overall_b = per_horizon_rmse(base, y)
    ph_skill = 1.0 - (ph_m / (ph_b + 1e-12))
    overall_skill = 1.0 - (overall_m / (overall_b + 1e-12))
    return ph_skill, overall_skill, ph_m, overall_m, ph_b, overall_b

# ---------------------------
# Training
# ---------------------------
def train_loop(model, loaders, device, epochs=80, max_lr=3e-3, clip=1.0, patience=8):
    train_loader, val_loader = loaders
    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=1e-2, betas=(0.9, 0.999))
    steps_per_epoch = len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.1, div_factor=25.0, final_div_factor=1e4, three_phase=False
    )
    criterion = torch.nn.HuberLoss(delta=1.0, reduction='mean')

    best_val = float('inf')
    best_state = None
    bad = 0

    for ep in range(1, epochs+1):
        model.train()
        run_loss = 0.0
        t0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)

            yhat = model(xb)
            loss = criterion(yhat, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            opt.step()
            sched.step()

            run_loss += loss.item() * xb.size(0)
        train_loss = run_loss / (len(train_loader.dataset))

        # Validate
        model.eval()
        vsum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                yhat = model(xb)
                vsum += torch.sqrt(((yhat - yb) ** 2).mean()).item() * xb.size(0)
        val_rmse = vsum / len(val_loader.dataset)

        dt = time.time() - t0
        print(f"Epoch {ep:03d}/{epochs} | train_loss={train_loss:.5f} | val_RMSE={val_rmse:.5f} | dt={dt:.1f}s")

        if val_rmse < best_val - 1e-6:
            best_val = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {ep} (no improvement for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to NPZ with X_* and y_* arrays")
    ap.add_argument("--horizon", type=int, default=24, help="H (default 24)")
    ap.add_argument("--lookback", type=int, default=168, help="L (default 168)")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--dilations", type=str, default="1,2,4,8,16,32")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", type=str, default="best_model.pt")
    ap.add_argument("--auto_tune", action="store_true",
                    help="Auto-tune bs/epochs/channels/workers (and LR) for MPS/CPU(16GB)")
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    set_seed(args.seed)

    device = pick_device()
    print(f"Device: {device}")

    # Load data
    Xtr, ytr, Xva, yva, Xte, yte = load_npz(args.npz)

    # Basic shape checks / trims to desired L,H if needed
    L, H = args.lookback, args.horizon
    if Xtr.shape[1] != L:
        if Xtr.shape[1] > L:
            Xtr = Xtr[:, -L:, :]; Xva = Xva[:, -L:, :]; Xte = Xte[:, -L:, :]
            print(f"Trimmed lookback to last {L} steps.")
        else:
            raise ValueError(f"X lookback {Xtr.shape[1]} < desired L={L}.")
    if ytr.shape[1] != H:
        if ytr.shape[1] > H:
            ytr = ytr[:, :H]; yva = yva[:, :H]; yte = yte[:, :H]
            print(f"Trimmed horizon to first {H} steps.")
        else:
            raise ValueError(f"y horizon {ytr.shape[1]} < desired H={H}.")

    # Z-score per feature using TRAIN only
    mu, sd = zscore_fit(Xtr)
    Xtr = zscore_apply(Xtr, mu, sd)
    Xva = zscore_apply(Xva, mu, sd)
    Xte = zscore_apply(Xte, mu, sd)

    # Optional auto-tune
    if args.auto_tune:
        tuned = autotune_hparams(args, device)
        args.batch_size = tuned["batch_size"]
        args.epochs     = tuned["epochs"]
        args.channels   = tuned["channels"]
        args.num_workers= tuned["num_workers"]
        args.lr         = tuned["lr"]

    print(f"[Config] bs={args.batch_size}  epochs={args.epochs}  C={args.channels}  "
          f"workers={args.num_workers}  lr={args.lr:.2e}")

    # Dataloaders (no pin_memory; CUDA-specific optimization removed)
    train_ds = TSWindowDataset(Xtr, ytr)
    val_ds   = TSWindowDataset(Xva, yva)
    test_ds  = TSWindowDataset(Xte, yte)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
                              num_workers=args.num_workers, pin_memory=False)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
                              num_workers=args.num_workers, pin_memory=False)

    # Model
    F = Xtr.shape[-1]
    dilations = tuple(int(x) for x in args.dilations.split(",") if x.strip())
    model = DilatedCausalCNN(in_feats=F, C=args.channels, k=args.kernel, dilations=dilations,
                             horizon=H, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,} | F={F}, L={L}, H={H}, dilations={dilations}")

    # Train
    best_val = train_loop(
        model, (train_loader, val_loader), device,
        epochs=args.epochs, max_lr=args.lr, clip=1.0, patience=8
    )
    print(f"Best val RMSE: {best_val:.6f}")

    # Save best
    torch.save({"state_dict": model.state_dict(),
                "mu": mu, "sd": sd,
                "config": vars(args)}, args.save)
    print(f"Saved: {args.save}")

    # Evaluate on test with per-horizon & skill vs persistence
    model.eval()
    yhats, ys, lasts = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device); yb = yb.to(device)
            yhat = model(xb)
            yhats.append(yhat.cpu()); ys.append(yb.cpu())
            # Baseline: use last feature at last lookback step as temp proxy (adjust index if needed)
            temp_proxy = xb[:,-1,-1].detach().cpu()  # [B]
            lasts.append(temp_proxy)

    yhat = torch.cat(yhats, dim=0)  # [N,H]
    y    = torch.cat(ys, dim=0)     # [N,H]
    last = torch.cat(lasts, dim=0)  # [N]

    ph_skill, overall_skill, ph_rmse_m, overall_rmse_m, ph_rmse_b, overall_rmse_b = skill_vs_persistence(yhat, y, last)

    # Print metrics
    print("\n=== Test Metrics ===")
    print(f"Overall RMSE (model):       {overall_rmse_m.item():.4f}")
    print(f"Overall RMSE (persistence): {overall_rmse_b.item():.4f}")
    print(f"Overall Skill vs persist:   {overall_skill.item():.4f}")
    print("\nPer-horizon (t+1..t+H):")
    for h in range(H):
        print(f"h+{h+1:02d}: RMSE_model={ph_rmse_m[h].item():.4f}  "
              f"RMSE_persist={ph_rmse_b[h].item():.4f}  Skill={ph_skill[h].item():.4f}")

if __name__ == "__main__":
    main()
