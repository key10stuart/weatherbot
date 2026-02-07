# data/preprocessor.py
"""
NCEI Global Hourly -> Preprocessed Features (for TCN temp forecasting)

Outputs a CSV in ./processed/ with (by default):
  sin_doy, cos_doy, sin_hod, cos_hod, temp_C
Optionally adds:
  temp_missing   (indicator that temp was originally missing before imputation)

Key fixes vs older version:
- Consistent timezone handling (UTC -> US/Eastern)
- Regularize to an exact hourly grid
- Robust missing handling: interpolate + ffill/bfill so temp_C has no NaNs
- Optional missingness indicator feature
"""

import sys
from datetime import datetime
from pathlib import Path
from io import StringIO

import requests
import numpy as np
import pandas as pd


def safe_input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def main() -> None:
    print("— NCEI Global Hourly → Preprocessed Features —")

    # --- DATE INPUT ---
    try:
        start = datetime.strptime(safe_input("Start (YYYYMMDD): ").strip(), "%Y%m%d")
        end = datetime.strptime(safe_input("End   (YYYYMMDD): ").strip(), "%Y%m%d")
    except ValueError:
        print("Use YYYYMMDD, e.g., 20150828")
        sys.exit(1)

    # We'll request whole days; later we convert to hourly index anyway.
    start_date = start.strftime("%Y-%m-%d")
    end_date = end.strftime("%Y-%m-%d")

    # --- STATION INPUT ---
    station = safe_input("Station ID (e.g., USW00003822 or 72207003822): ").strip()
    if not station:
        print("Station ID required.")
        sys.exit(1)

    # --- NOAA API URL ---
    base_url = "https://www.ncei.noaa.gov/access/services/data/v1"
    params = {
        "dataset": "global-hourly",
        "stations": station,
        "startDate": start_date,
        "endDate": end_date,
        "format": "csv",
        "includeStationName": "true",
        "units": "metric",
    }

    # --- API call ---
    print(f"\nFetching hourly data from NCEI for {station}…")
    response = requests.get(base_url, params=params, timeout=60)
    if response.status_code != 200:
        print(f"API request failed: {response.status_code} {response.text}")
        sys.exit(1)

    df = pd.read_csv(StringIO(response.text))
    if df.empty:
        print("No data returned (empty response). Check station/date range.")
        sys.exit(1)

    # --- raw pathing ---
    raw_dir = Path("./raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_name = safe_input("\nEnter a name for the RAW file (without .csv): ").strip()
    if not raw_name:
        print("Raw filename is required.")
        sys.exit(1)

    raw_safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in raw_name)
    raw_path = raw_dir / f"{raw_safe}.csv"
    df.to_csv(raw_path, index=False)
    print(f"Saved raw to: {raw_path}")

    # ----------------------------
    # Filtering / parsing features
    # ----------------------------

    # Keep primary hourly reports (as in your original script)
    if "REPORT_TYPE" in df.columns:
        dfp = df[df["REPORT_TYPE"] == "FM-15"].copy()
    else:
        dfp = df.copy()

    if dfp.empty:
        print("After filtering REPORT_TYPE == FM-15, no rows remain. Aborting.")
        sys.exit(1)

    # Parse timestamps: DATE is typically ISO-ish; treat as UTC then convert to Eastern.
    if "DATE" not in dfp.columns:
        print("Expected a DATE column from NCEI, but it was not found.")
        sys.exit(1)

    dt_utc = pd.to_datetime(dfp["DATE"], errors="coerce", utc=True)
    bad_dt = dt_utc.isna().sum()
    if bad_dt:
        print(f"Warning: {bad_dt} rows had unparseable DATE and will be dropped.")
    dfp = dfp.loc[dt_utc.notna()].copy()
    dt_utc = dt_utc.loc[dt_utc.notna()]

    # Convert to US/Eastern (handles DST correctly)
    dt_est = dt_utc.dt.tz_convert("US/Eastern")
    dfp["dt_est"] = dt_est

    # Parse temperature
    # NCEI TMP is often like "0123,1" meaning 12.3C with a QC flag.
    if "TMP" not in dfp.columns:
        print("Expected TMP column from NCEI, but it was not found.")
        sys.exit(1)

    tmp_split = dfp["TMP"].astype(str).str.split(",", expand=True)
    tmp_tenths = pd.to_numeric(tmp_split[0], errors="coerce")  # tenths of C
    dfp["temp_C"] = tmp_tenths / 10.0
    dfp["temp_qc"] = tmp_split[1] if tmp_split.shape[1] > 1 else None

    print("Initial temp_C NaNs (from parsing):", int(dfp["temp_C"].isna().sum()))

    # Index by time (Eastern), sort, drop duplicates by keeping last (you can change policy)
    dfp = dfp.set_index("dt_est").sort_index()
    dupes = int(dfp.index.duplicated().sum())
    if dupes:
        print(f"Found {dupes} duplicate timestamps. Keeping last occurrence.")
        dfp = dfp[~dfp.index.duplicated(keep="last")]

    # ----------------------------
    # Regularize to strict hourly grid
    # ----------------------------
    do_regularize = safe_input("Interpolate and regularize to strict hourly grid? (y/n): ").strip().lower()
    if do_regularize == "y":
        # Build a strict hourly grid in Eastern time covering observed range
        start_ts = dfp.index.min().floor("h")
        end_ts = dfp.index.max().ceil("h")
        hourly_idx = pd.date_range(start=start_ts, end=end_ts, freq="h", tz="US/Eastern")

        # Reindex -> creates NaNs for missing hours
        dfp = dfp.reindex(hourly_idx)
        dfp.index.name = "time_est"

        # Track missingness BEFORE filling
        dfp["temp_missing"] = dfp["temp_C"].isna().astype(np.float32)

        # Robust fill strategy:
        # 1) interpolate across gaps (both directions)
        # 2) ffill/bfill for any remaining edges
        dfp["temp_C"] = dfp["temp_C"].interpolate(method="linear", limit_direction="both")
        dfp["temp_C"] = dfp["temp_C"].ffill().bfill()
    else:
        # Even if we don't regularize, it is still safer to handle missing temps
        dfp["temp_missing"] = dfp["temp_C"].isna().astype(np.float32)
        dfp["temp_C"] = dfp["temp_C"].interpolate(method="linear", limit_direction="both")
        dfp["temp_C"] = dfp["temp_C"].ffill().bfill()

    # Confirm no NaNs in temp_C (critical for TCN inputs/targets)
    remaining = int(dfp["temp_C"].isna().sum())
    if remaining:
        print(f"Error: temp_C still contains {remaining} NaNs after imputation. Aborting.")
        sys.exit(1)

    # ----------------------------
    # Time features (cyclical)
    # ----------------------------
    # Note: index is tz-aware Eastern.
    hod = dfp.index.hour + dfp.index.minute / 60.0
    dfp["sin_hod"] = np.sin(2 * np.pi * hod / 24.0)
    dfp["cos_hod"] = np.cos(2 * np.pi * hod / 24.0)

    doy = dfp.index.dayofyear.astype(float)
    year_period = 365.2425
    dfp["sin_doy"] = np.sin(2 * np.pi * doy / year_period)
    dfp["cos_doy"] = np.cos(2 * np.pi * doy / year_period)

    # Basic QC output
    print("Final dfp length:", len(dfp))
    diffs = dfp.index.to_series().diff().dropna().value_counts().head(5)
    print("Index diffs (top):")
    print(diffs)

    # ----------------------------
    # Save processed CSV
    # ----------------------------
    processed_dir = Path("./processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    out_name = safe_input("state_your_name_: ").strip()
    if not out_name:
        print("Output name required.")
        sys.exit(1)

    # Choose columns
    include_missing_feature = safe_input("Include temp_missing feature? (y/n): ").strip().lower() == "y"
    cols = ["sin_doy", "cos_doy", "sin_hod", "cos_hod", "temp_C"]
    if include_missing_feature:
        cols.append("temp_missing")

    out_path = processed_dir / f"{out_name}.csv"
    dfp[cols].to_csv(out_path, index=True)
    print(f"Saved processed to: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()

