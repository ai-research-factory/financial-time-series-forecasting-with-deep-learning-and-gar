"""
BTC-USD data loading and preprocessing pipeline.
Fetches data from ARF Data API and computes log returns.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

API_BASE = "https://ai.1s.xyz/api/data/ohlcv"
DEFAULT_TICKER = "BTC-USD"
DEFAULT_INTERVAL = "1d"
DEFAULT_PERIOD = "10y"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"


def fetch_btc_data(
    ticker: str = DEFAULT_TICKER,
    interval: str = DEFAULT_INTERVAL,
    period: str = DEFAULT_PERIOD,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch BTC-USD OHLCV data from the ARF Data API.

    Args:
        ticker: Ticker symbol.
        interval: Data interval (e.g., '1d').
        period: Data period (e.g., '10y').
        use_cache: If True, use cached CSV if available.

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{ticker.replace('/', '_')}_{interval}_{period}.csv"

    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=["timestamp"])
        return df

    url = f"{API_BASE}?ticker={ticker}&interval={interval}&period={period}"
    df = pd.read_csv(url, parse_dates=["timestamp"])

    df.to_csv(cache_file, index=False)
    return df


def load_btc_data(
    ticker: str = DEFAULT_TICKER,
    interval: str = DEFAULT_INTERVAL,
    period: str = DEFAULT_PERIOD,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Load and preprocess BTC-USD data for modeling.

    Steps:
      1. Fetch raw OHLCV data from ARF Data API.
      2. Deduplicate by timestamp.
      3. Filter out any future dates.
      4. Compute log returns: log(Close[t] / Close[t-1]).
      5. Drop rows with NaN returns.

    Args:
        ticker: Ticker symbol.
        interval: Data interval.
        period: Data period.
        use_cache: Whether to use local cache.

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume, returns.
        Indexed by integer, with 'Date' as a datetime column.
    """
    raw = fetch_btc_data(ticker=ticker, interval=interval, period=period, use_cache=use_cache)

    df = raw.copy()
    df = df.rename(columns={
        "timestamp": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    today = pd.Timestamp.now().normalize()
    df = df[df["Date"] <= today].copy()

    df["returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna(subset=["returns"]).reset_index(drop=True)

    return df


def save_processed_data(df: pd.DataFrame, filename: str = "btc_usd_daily.csv") -> Path:
    """
    Save processed DataFrame to data/processed/.

    Args:
        df: Processed DataFrame with Date, Close, returns columns.
        filename: Output filename.

    Returns:
        Path to the saved file.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / filename
    df.to_csv(output_path, index=False)
    return output_path
