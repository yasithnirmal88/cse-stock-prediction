"""Process the S&P SL20 CSE multi-stock historical Excel dataset into clean daily OHLCV CSVs.

Adapted from reference ideas:
- Filters for ordinary voting shares ('N' series) to avoid cross-security contamination.
- Corrects missing High/Low/Open bounds where appropriate.
- Handles trading halts, zero-volume days, and corporate action splits.
- Standardizes output symbols to standard CSE format: '{CompanyCode}.N0000.csv'.
- Saves individual stock OHLCV series into cse-quant-system/data/processed/.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default file paths inside cse-quant-system
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_FILE = PROJECT_ROOT / "data" / "raw" / "SP20_CSE_Dataset_2001_2025_Adjusted.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

USE_COLS = [
    "Date",
    "CompanyCode",
    "ShortName",
    "MainType",
    "Open",
    "High",
    "Low",
    "Close",
    "TradeVolume",
    "ShareVolume",
    "Turnover",
]


def clean_stock_series(df: pd.DataFrame, gap_days: int = 10, split_threshold: float = 0.35) -> pd.DataFrame:
    """Clean and standardize an individual stock's time series."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

    # Filter invalid close prices
    df = df[df["Close"] > 0].copy()

    # If Open is missing, fill with previous Close (or current Close if first row)
    if "Open" in df.columns:
        prev_close = df["Close"].shift(1)
        df["Open"] = df["Open"].fillna(prev_close).fillna(df["Close"])

    # If High / Low are missing, fall back to max/min of Open & Close
    if "High" in df.columns:
        df["High"] = df["High"].fillna(df[["Open", "Close"]].max(axis=1))
    if "Low" in df.columns:
        df["Low"] = df["Low"].fillna(df[["Open", "Close"]].min(axis=1))

    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)

    # Volume: use ShareVolume, fallback to TradeVolume, fallback to 0
    if "ShareVolume" in df.columns:
        vol = df["ShareVolume"].fillna(df.get("TradeVolume", 0)).fillna(0)
    else:
        vol = df.get("TradeVolume", 0).fillna(0)
    df["Volume"] = vol.astype(float)

    # Calculate returns and cleaning flags (adapted from reference cleaning logic)
    df["gap_days"] = df["Date"].diff().dt.days
    df["log_return"] = np.log(df["Close"]).diff()
    df["flag_gap"] = df["gap_days"] > gap_days
    df["flag_split"] = df["log_return"].abs() > split_threshold

    # Clean log returns where split or large gap occurs
    df["log_return_clean"] = df["log_return"].where(~(df["flag_gap"] | df["flag_split"]))

    df["Source"] = "CSE_Adjusted_2001_2025"

    output_cols = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "log_return",
        "log_return_clean",
        "flag_gap",
        "flag_split",
        "Source",
    ]
    return df[output_cols]


def process_excel_dataset(
    excel_path: Path = DEFAULT_RAW_FILE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sheet_name: str = "Full_Dataset_2001_2025",
) -> dict[str, int]:
    """Read the full multi-stock Excel workbook and export individual stock CSVs."""
    if not excel_path.exists():
        raise FileNotFoundError(f"Input file not found at {excel_path}")

    logger.info(f"Reading raw Excel dataset from: {excel_path}")
    raw_df = pd.read_excel(excel_path, sheet_name=sheet_name, usecols=USE_COLS)
    logger.info(f"Loaded {len(raw_df)} total rows from {sheet_name}.")

    # Keep only ordinary voting shares ('N')
    ordinary_df = raw_df[raw_df["MainType"] == "N"].copy()
    logger.info(f"Retained {len(ordinary_df)} rows for MainType == 'N'.")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    companies = sorted(ordinary_df["CompanyCode"].dropna().unique())
    logger.info(f"Processing {len(companies)} companies: {companies}")

    for code in companies:
        company_subset = ordinary_df[ordinary_df["CompanyCode"] == code]
        cleaned_series = clean_stock_series(company_subset)

        # Standard CSE symbol convention used in universe.yaml (e.g. JKH.N0000)
        symbol = f"{code}.N0000"
        csv_filename = f"{symbol}.csv"
        csv_path = output_dir / csv_filename

        cleaned_series.to_csv(csv_path, index=False)
        summary[symbol] = len(cleaned_series)
        logger.info(
            f"Saved {symbol}: {len(cleaned_series)} rows ({cleaned_series['Date'].min().date()} to {cleaned_series['Date'].max().date()}) -> {csv_path.name}"
        )

    # Also build and save an equal-weighted Market Proxy index (S&P SL20 core proxy)
    build_market_proxy(output_dir, companies)

    logger.info("Excel dataset processing complete.")
    return summary


def build_market_proxy(output_dir: Path, companies: list[str]) -> None:
    """Build an equal-weighted daily market proxy return & synthetic index series."""
    series_list = []
    for code in companies:
        p = output_dir / f"{code}.N0000.csv"
        if p.exists():
            df = pd.read_csv(p, usecols=["Date", "log_return_clean", "Volume", "Close"])
            df["Date"] = pd.to_datetime(df["Date"])
            df["CompanyCode"] = code
            series_list.append(df)

    if not series_list:
        return

    all_data = pd.concat(series_list, ignore_index=True)
    pivot_ret = all_data.pivot(index="Date", columns="CompanyCode", values="log_return_clean")
    
    # Require at least 8 stocks present to establish a valid daily market index proxy
    valid_days = pivot_ret.notna().sum(axis=1) >= 8
    market_ret = pivot_ret.mean(axis=1, skipna=True)[valid_days]

    proxy_df = pd.DataFrame({
        "Date": market_ret.index,
        "market_log_return": market_ret.values,
    }).sort_values("Date")

    # Reconstruct normalized proxy price index starting at base 1000
    proxy_df["Proxy_Index"] = 1000.0 * np.exp(proxy_df["market_log_return"].cumsum())
    proxy_path = output_dir / "CSE_MARKET_PROXY.csv"
    proxy_df.to_csv(proxy_path, index=False)
    logger.info(f"Saved Market Proxy index with {len(proxy_df)} days -> {proxy_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSE historical Excel dataset into clean OHLCV CSVs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW_FILE, help="Path to raw Excel file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Path to output directory")
    args = parser.parse_args()

    process_excel_dataset(args.input, args.output)
