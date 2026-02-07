import json
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across numpy, torch, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ___________________________________________________________________________________________________
# NPZ IO
# ___________________________________________________________________________________________________


def _squeeze_last_if_singleton(a: np.ndarray) -> np.ndarray:
    """If target is (N, H, 1) or (N, 1), drop the last dim."""
    return a[..., 0] if (a.ndim >= 2 and a.shape[-1] == 1) else a


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load NPZ with keys: X_train, y_train, X_val, y_val, X_test, y_test."""
    npz = np.load(path)
    X_train = npz["X_train"]
    y_train = _squeeze_last_if_singleton(npz["y_train"])
    X_val = npz["X_val"]
    y_val = _squeeze_last_if_singleton(npz["y_val"])
    X_test = npz["X_test"]
    y_test = _squeeze_last_if_singleton(npz["y_test"])
    return X_train, y_train, X_val, y_val, X_test, y_test


# ___________________________________________________________________________________________________
# Normalization & Dataset (MASKING)
# ___________________________________________________________________________________________________


def zscore_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-feature μ/σ from TRAIN windows only. X: [N, L, F]. NaN-aware."""
    mu = np.nanmean(X.reshape(-1, X.shape[-1]), axis=0)
    sd = np.nanstd(X.reshape(-1, X.shape[-1]), axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def zscore_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Apply per-feature z-scoring. Broadcasts μ/σ across [N, L, F]."""
    return (X - mu) / sd


class TSWindowDataset(Dataset):
    """
    Dataset that builds masks for NaNs and zero-imputes AFTER capturing masks.

    x_mask: [N, L] where 1.0 = valid timestep (all feats present), 0.0 = masked
    y_mask: [N, H] where 1.0 = valid target, 0.0 = masked
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Capture masks BEFORE imputation
        x_nan_mask = np.isnan(X).any(axis=2)  # [N, L]
        y_nan_mask = np.isnan(y)              # [N, H] (or [N] if squeezed)

        self.x_mask = torch.as_tensor(~x_nan_mask, dtype=torch.float32)
        self.y_mask = torch.as_tensor(~y_nan_mask, dtype=torch.float32)

        # Zero-impute for model input/targets
        X = X.copy()
        y = y.copy()
        X[np.isnan(X)] = 0.0
        y[np.isnan(y)] = 0.0

        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int):
        return self.X[i], self.y[i], self.x_mask[i], self.y_mask[i]
