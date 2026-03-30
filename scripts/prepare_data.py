#!/usr/bin/env python3
"""
Prepare BTC-USD daily data for modeling.
Fetches data from ARF Data API, computes log returns, and saves to CSV.

Usage:
    python scripts/prepare_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.data_loader import load_btc_data, save_processed_data


def main():
    print("Fetching BTC-USD daily data from ARF Data API...")
    df = load_btc_data(ticker="BTC-USD", interval="1d", period="10y")

    print(f"  Rows: {len(df)}")
    print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  NaN in returns: {df['returns'].isna().sum()}")

    output_path = save_processed_data(df)
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
