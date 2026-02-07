import argparse
import json
import os
import time
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataprep import (
    set_seed,
    load_npz,
    zscore_fit,
    zscore_apply,
    TSWindowDataset,
)
from models.tcn_1d import DilatedCausalCNN


# -------------------------
# Device / small heuristics
# -------------------------
def pick_device() -> torch.device:
    """Prefer Apple MPS if available, else CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autotune_hparams(cfg: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    tuned = dict(cfg)
    if device.type == "mps":
        tuned["num_workers"] = max(int(tuned.get("num_workers", 0)), 0)
        tuned["batch_size"] = min(int(tuned.get("batch_size", 128)), 128)
        tuned["channels"] = min(int(tuned.get("channels", 64)), 64)
        tuned["lr"] = min(float(tuned.get("lr", 3e-3)), 3e-3)
    else:
        tuned["num_workers"] = max(int(tuned.get("num_workers", 0)), 2)
        tuned["batch_size"] = min(int(tuned.get("batch_size", 128)), 64)
        tuned["channels"] = min(int(tuned.get("channels", 64)), 48)
        tuned["lr"] = min(float(tuned.get("lr", 3e-3)), 2e-3)
    return tuned


# -------------------------
# Utility: metrics/baselines
# -------------------------
@torch.no_grad()
def per_horizon_rmse(yhat: torch.Tensor, y: torch.Tensor, eps: float = 1e-9) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    yhat, y: (N, H) or (N, H, C). Returns per-horizon RMSE and overall RMSE.
    """
    err = (yhat - y) ** 2
    # Horizon dim assumed dim=1
    ph = torch.sqrt(err.mean(dim=0) + eps)  # (H, ...) if extra dims exist
    overall = torch.sqrt(err.mean() + eps)
    return ph, overall


@torch.no_grad()
def persistence_baseline(last_vals: torch.Tensor, H: int) -> torch.Tensor:
    """Repeat last observed value H times. last_vals: (N,) => (N,H)."""
    return last_vals.unsqueeze(1).repeat(1, H)


@torch.no_grad()
def skill_vs_persistence(yhat: torch.Tensor, y: torch.Tensor, last_vals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (per_horizon_skill, overall_skill).
    Skill = 1 - RMSE_model / RMSE_persistence
    """
    H = y.shape[1]
    base = persistence_baseline(last_vals, H)
    ph_m, overall_m = per_horizon_rmse(yhat, y)
    ph_b, overall_b = per_horizon_rmse(base, y)
    ph_skill = 1.0 - (ph_m / (ph_b + 1e-12))
    overall_skill = 1.0 - (overall_m / (overall_b + 1e-12))
    return ph_skill, overall_skill


# -------------------------
# Utility: data cleaning
# -------------------------
def _squeeze_last_if_singleton(a: np.ndarray) -> np.ndarray:
    """If target is (N,H,1) or (...,1), drop the last dim."""
    return a[..., 0] if (a.ndim >= 1 and a.shape[-1] == 1) else a


def _finite_window_mask_X(X: np.ndarray) -> np.ndarray:
    """Return boolean mask of windows with all finite values. X: (N,L,F)."""
    return np.isfinite(X).all(axis=(1, 2))


def _finite_window_mask_y(y: np.ndarray) -> np.ndarray:
    """
    Return boolean mask of windows with all finite values.
    y: (N,H) or (N,H,C)
    """
    axes = tuple(range(1, y.ndim))
    return np.isfinite(y).all(axis=axes)


def drop_nonfinite_windows(X: np.ndarray, y: np.ndarray, name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Drop any samples where X or y contains NaN/Inf.
    This is the simplest fix for the "loss=nan from epoch 1" issue.
    """
    y = _squeeze_last_if_singleton(y)
    m = _finite_window_mask_X(X) & _finite_window_mask_y(y)
    dropped = int((~m).sum())
    if dropped > 0:
        print(f"[{name}] Dropping {dropped} / {len(m)} windows with NaN/Inf in X or y")
    return X[m], y[m]


def describe_array(a: np.ndarray, name: str) -> None:
    a = np.asarray(a)
    finite = np.isfinite(a)
    print(
        f"{name}: shape={a.shape}  finite={finite.all()}  "
        f"nan={np.isnan(a).any()}  inf={np.isinf(a).any()}  "
        f"min={np.nanmin(a):.4f}  max={np.nanmax(a):.4f}"
    )


# -------------------------
# Training loop
# -------------------------
def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    max_lr: float,
    clip: float,
    patience: int,
    weight_decay: float = 1e-2,
) -> Dict[str, Any]:
    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    steps_per_epoch = max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1e4,
        three_phase=False,
    )
    criterion = torch.nn.HuberLoss(delta=1.0, reduction="mean")

    best_val = float("inf")
    best_state = None
    bad = 0

    history = {"train_loss": [], "val_rmse": []}

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        run_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            yhat = model(xb)

            # If yhat/y have unexpected extra dims, let it error loudly.
            loss = criterion(yhat, yb)

            if not torch.isfinite(loss):
                raise RuntimeError("Loss became NaN/Inf. Check data after normalization or reduce lr.")

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            opt.step()
            sched.step()

            run_loss += loss.item() * xb.size(0)

        train_loss = run_loss / len(train_loader.dataset)

        # Validate RMSE
        model.eval()
        vsum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device).contiguous()
                yb = yb.to(device).contiguous()
                yhat = model(xb)
                rmse = torch.sqrt(((yhat - yb) ** 2).mean() + 1e-9)
                vsum += rmse.item() * xb.size(0)

        val_rmse = vsum / len(val_loader.dataset)
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_rmse"].append(val_rmse)

        print(f"Epoch {ep:03d}/{epochs} | train_loss={train_loss:.6f} | val_RMSE={val_rmse:.6f} | dt={dt:.1f}s")

        if val_rmse < best_val - 1e-6:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {ep} (no improvement for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_val_rmse": best_val, "history": history}


# -------------------------
# Main
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train dilated causal CNN forecaster")
    p.add_argument("--npz", type=str, default="data/processed/sav_0927_v3.npz", help="Path to NPZ dataset")
    p.add_argument("--lookback", type=int, default=168, help="Context length L")
    p.add_argument("--horizon", type=int, default=24, help="Forecast horizon H")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--kernel", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--dilations", type=str, default="1,2,4,8,16,32")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--auto_tune", action="store_true")
    p.add_argument("--save", type=str, default="best_model.pt")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--target_feat_idx", type=int, default=-1, help="Which input feature to use for persistence baseline")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = vars(args)

    set_seed(cfg["seed"])
    device = pick_device()
    print(f"Device: {device}")

    if cfg["auto_tune"]:
        cfg = autotune_hparams(cfg, device)

    print("[Config]\n" + json.dumps(cfg, indent=2))

    # Load
    Xtr, ytr, Xva, yva, Xte, yte = load_npz(cfg["npz"])
    ytr = _squeeze_last_if_singleton(ytr)
    yva = _squeeze_last_if_singleton(yva)
    yte = _squeeze_last_if_singleton(yte)

    # Trim to requested L/H (like notebook)
    L, H = int(cfg["lookback"]), int(cfg["horizon"])
    if Xtr.shape[1] != L:
        if Xtr.shape[1] > L:
            Xtr = Xtr[:, -L:, :]
            Xva = Xva[:, -L:, :]
            Xte = Xte[:, -L:, :]
            print(f"Trimmed lookback to last {L} steps.")
        else:
            raise ValueError(f"X lookback {Xtr.shape[1]} < desired L={L}.")

    if ytr.shape[1] != H:
        if ytr.shape[1] > H:
            ytr = ytr[:, :H]
            yva = yva[:, :H]
            yte = yte[:, :H]
            print(f"Trimmed horizon to first {H} steps.")
        else:
            raise ValueError(f"y horizon {ytr.shape[1]} < desired H={H}.")

    # Drop NaN/Inf windows BEFORE normalization (critical)
    Xtr, ytr = drop_nonfinite_windows(Xtr, ytr, "train")
    Xva, yva = drop_nonfinite_windows(Xva, yva, "val")
    Xte, yte = drop_nonfinite_windows(Xte, yte, "test")

    describe_array(Xtr, "X_train (pre-norm)")
    describe_array(ytr, "y_train")
    F = Xtr.shape[-1]
    print(f"Dataset sizes — train: {len(Xtr)}, val: {len(Xva)}, test: {len(Xte)}; F={F}, L={L}, H={H}")

    # Normalize using TRAIN only
    mu, sd = zscore_fit(Xtr)
    Xtr = zscore_apply(Xtr, mu, sd)
    Xva = zscore_apply(Xva, mu, sd)
    Xte = zscore_apply(Xte, mu, sd)

    # Drop again just in case normalization produced non-finite (shouldn't if train was finite + sd guarded)
    Xtr, ytr = drop_nonfinite_windows(Xtr, ytr, "train-postnorm")
    Xva, yva = drop_nonfinite_windows(Xva, yva, "val-postnorm")
    Xte, yte = drop_nonfinite_windows(Xte, yte, "test-postnorm")

    # Dataloaders
    train_ds = TSWindowDataset(Xtr, ytr)
    val_ds = TSWindowDataset(Xva, yva)
    test_ds = TSWindowDataset(Xte, yte)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        drop_last=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=False,
    )

    # Model
    dilations = tuple(int(x) for x in str(cfg["dilations"]).split(",") if x.strip())
    model = DilatedCausalCNN(
        in_feats=F,
        C=int(cfg["channels"]),
        k=int(cfg["kernel"]),
        dilations=dilations,
        horizon=H,
        dropout=float(cfg["dropout"]),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,} | dilations={dilations}")

    # Train
    out = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=int(cfg["epochs"]),
        max_lr=float(cfg["lr"]),
        clip=float(cfg["clip"]),
        patience=int(cfg["patience"]),
    )
    best_val = out["best_val_rmse"]
    print(f"Best val RMSE: {best_val:.6f}")

    # Save
    payload = {"state_dict": model.state_dict(), "mu": mu, "sd": sd, "config": cfg}
    torch.save(payload, cfg["save"])
    print(f"Saved: {cfg['save']}")

    # Evaluate on test
    model.eval()
    yhats, ys, lasts = [], [], []
    target_feat_idx = int(cfg["target_feat_idx"])
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yhat = model(xb)

            yhats.append(yhat.cpu())
            ys.append(yb.cpu())

            # Persistence baseline uses *input* last step of a chosen feature index
            last_val = xb[:, -1, target_feat_idx].detach().cpu()
            lasts.append(last_val)

    yhat = torch.cat(yhats, dim=0)
    y = torch.cat(ys, dim=0)
    last_vals = torch.cat(lasts, dim=0)

    ph_m, overall_m = per_horizon_rmse(yhat, y)
    ph_skill, overall_skill = skill_vs_persistence(yhat, y, last_vals)
    base = persistence_baseline(last_vals, H)
    ph_b, overall_b = per_horizon_rmse(base, y)

    print("\n=== Test Metrics ===")
    print(f"Overall RMSE (model):       {overall_m.item():.4f}")
    print(f"Overall RMSE (persistence): {overall_b.item():.4f}")
    print(f"Overall Skill vs persist:   {overall_skill.item():.4f}")

    print("\nPer-horizon (t+1..t+H):")
    # ph_* could be (H,) or (H,C) depending on target dims; print the scalar case neatly
    if ph_m.ndim == 1:
        for h in range(H):
            print(
                f"h+{h+1:02d}: RMSE_model={ph_m[h].item():.4f}  "
                f"RMSE_persist={ph_b[h].item():.4f}  Skill={ph_skill[h].item():.4f}"
            )
    else:
        print(f"Per-horizon RMSE has shape {tuple(ph_m.shape)} (multi-target). Printing overall only.")


if __name__ == "__main__":
    main()

