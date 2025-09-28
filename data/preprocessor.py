#downloader
import sys
from datetime import datetime
from pathlib import Path
import requests
import numpy as np
import pandas as pd
from io import StringIO

# --- TITLE ---
print("— NCEI Global Hourly → Preprocessed Features —")

# --- DATE INPUT ---
try:
    start = datetime.strptime(input("Start (YYYYMMDD): ").strip(), "%Y%m%d")
    end   = datetime.strptime(input("End   (YYYYMMDD): ").strip(), "%Y%m%d").replace(hour=23, minute=59)
except ValueError:
    print("Use YYYYMMDD, e.g., 20150828"); sys.exit(1)

# --- STATION INPUT ---
station = input("Station ID (e.g., USW00003822 or 72207003822): ").strip()
if not station:
    print("Station ID required."); sys.exit(1)

# --- NOAA API URL ---
BASE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
params = {
    "dataset": "global-hourly",
    "stations": station,
    "startDate": start.strftime("%Y-%m-%d"),
    "endDate": end.strftime("%Y-%m-%d"),
    "format": "csv",
    "includeStationName": "true",
    "units": "metric"
}

# --- API call ---
print(f"\nFetching hourly data from NCEI for {station}…")
response = requests.get(BASE_URL, params=params, timeout=60)
if response.status_code != 200:
    print(f"API request failed: {response.status_code} {response.text}")
    sys.exit(1)
df = pd.read_csv(StringIO(response.text))
df.head(10)

# --- raw pathing ---
RAW_DIR  = Path("./raw")
raw_name = input("\nEnter a name for the RAW file (without .csv): ").strip()
if not raw_name:
    print("Raw filename is required."); sys.exit(1)
raw_safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in raw_name)
raw_path = RAW_DIR / f"{raw_safe}.csv"
df.to_csv(raw_path, index=False)

# --- CNN prep ---
fm15 = df[df['REPORT_TYPE'] == 'FM-15']
print("FM15 rows:", len(fm15))  # should be 87,322

# Reset index but drop old index to avoid confusion
dfp = fm15.reset_index(drop=True)
dfp['dt_utc'] = pd.to_datetime(dfp['DATE'])
dfp['dt_est'] = dfp['dt_utc'] - pd.Timedelta(hours=5)
tmp_split = dfp['TMP'].astype(str).str.split(',', expand=True)
dfp['temp_C'] = pd.to_numeric(tmp_split[0], errors='coerce') / 10.0  # tenths of °C → °C
dfp['temp_qc'] = tmp_split[1]

# Set EST time as index before reindexing to continuous hourly grid
dfp = dfp.set_index('dt_est').sort_index()

# Add date parts
dfp['year'] = dfp.index.year
dfp['month'] = dfp.index.month
dfp['day'] = dfp.index.day
dfp['hour'] = dfp.index.hour

# Cyclical features
hod = dfp.index.hour + dfp.index.minute / 60.0
dfp['sin_hod'] = np.sin(2 * np.pi * hod / 24.0)
dfp['cos_hod'] = np.cos(2 * np.pi * hod / 24.0)

doy = dfp.index.dayofyear.astype(float)
year_period = 365.2425
dfp['sin_doy'] = np.sin(2 * np.pi * doy / year_period)
dfp['cos_doy'] = np.cos(2 * np.pi * doy / year_period)

print("Final dfp length:", len(dfp))
print("Expected Hourly Index Length for 10 yrs : 87720")

# Differences between consecutive timestamps
time_diffs = dfp.index.to_series().diff()

# Summary
print(time_diffs.value_counts().head(10))  # should show exactly 1h only

# Any gaps?
gaps = dfp[time_diffs > pd.Timedelta(hours=1)]
print("Gaps found:", len(gaps))

# Any duplicates?
duplicates = dfp.index.duplicated().sum()
print("Duplicate timestamps:", duplicates)

interpolator = str(input('interpolate and regularize ?_').lower())
if interpolator == 'y' :
    # Build hourly range starting at the FIRST actual timestamp
    start = dfp.index.min()   # first obs, e.g., 2015-09-07 00:53:00+00:00
    end   = dfp.index.max()   # last obs
    hourly_index = pd.date_range(start=start, end=end, freq='h', tz='EST')
    
    # Create a continuous hourly time index
    idx = pd.date_range(start=start, end=end, freq='h')
    dfp = dfp.reindex(idx)
    dfp.index.name = 'time_est'

    dfp['temp_C'] = dfp['temp_C'].interpolate(limit=1)
else :
    print("very well.")

diffs = dfp.index.to_series().diff().dropna().value_counts()
print(diffs)  # should be exactly 1:00:00