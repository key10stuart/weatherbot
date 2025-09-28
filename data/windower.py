import argparse
import numpy as np
import pandas as pd
import os

def make_windows(data: np.ndarray, L: int, N: int):
    """
    Create input/output windows for forecasting.
    data: (M, F) array of features
    L: context length
    N: horizon length
    Returns:
        X: (num_windows, L, F)
        y: (num_windows, N)
    """
    M, F = data.shape
    num_windows = M - L - N + 1
    X = np.zeros((num_windows, L, F), dtype=np.float32)
    y = np.zeros((num_windows, N), dtype=np.float32)

    for i in range(num_windows):
        X[i] = data[i:i+L, :]
        y[i] = data[i+L:i+L+N, 0]  # temp column only

    return X, y

def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        "train": (X[:train_end], y[:train_end]),
        "val":   (X[train_end:val_end], y[train_end:val_end]),
        "test":  (X[val_end:], y[val_end:])
    }
    return splits

def main(args):
    # load CSV
    # print(type(args.input))
    df = pd.read_csv(args.input)

    # drop the timestamp column if present
    if "time_est" in df.columns:
        df = df.drop(columns=["time_est"])

    data = df.values.astype(np.float32)  # shape (M, F)

    # make windows
    X, y = make_windows(data, args.L, args.N)

    # split chronologically
    splits = split_data(X, y, args.train_ratio, args.val_ratio)

    # save to npz
    foldername = "processed"
    filepath = os.path.join(foldername, args.output)

    os.makedirs(foldername, exist_ok=True)
    np.savez_compressed(
        filepath,
        X_train=splits["train"][0], y_train=splits["train"][1],
        X_val=splits["val"][0],     y_val=splits["val"][1],
        X_test=splits["test"][0],   y_test=splits["test"][1],
        meta=dict(L=args.L, N=args.N, features=data.shape[1])
    )
    print(f"Saved dataset to {args.output}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input CSV file with columns [time_est,temp,sin_hod,cos_hod,sin_doy,cos_doy]")
    parser.add_argument("--output", type=str, required=True,
                        help="Filename to output .npz file")
    parser.add_argument("--L", type=int, required=True,
                        help="Context length (hours)")
    parser.add_argument("--N", type=int, required=True,
                        help="Horizon length (hours)")
    parser.add_argument("--train_ratio", type=float, default=0.7
                        )
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()
    main(args)
