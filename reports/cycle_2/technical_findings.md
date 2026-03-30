# Technical Findings — Cycle 2 (Phase 2: Real Data Pipeline)

## Implementation Summary

Built the BTC-USD data pipeline that fetches daily OHLCV data from the ARF Data API, computes log returns, and saves processed data for downstream modeling.

## Components

### `src/data/data_loader.py`
- `fetch_btc_data()`: Fetches raw OHLCV from ARF Data API with local caching.
- `load_btc_data()`: Full preprocessing pipeline — rename columns, deduplicate, filter future dates, compute log returns, drop NaN.
- `save_processed_data()`: Saves processed DataFrame to `data/processed/`.

### `scripts/prepare_data.py`
- Entry point script that calls `load_btc_data()` and `save_processed_data()`.
- Running `python3 scripts/prepare_data.py` generates `data/processed/btc_usd_daily.csv`.

## Data Summary

All values from `reports/cycle_2/metrics.json`:

| Metric | Value |
|---|---|
| Total rows | 3651 (see `customMetrics.dataRows`) |
| Date range | 2016-03-31 to 2026-03-30 |
| NaN in returns | 0 (see `customMetrics.nanCountReturns`) |
| Columns | Date, Open, High, Low, Close, Volume, returns |

## Data Integrity Checks

- No future dates (all dates <= 2026-03-30)
- No duplicate timestamps
- Dates sorted in ascending order
- Log returns computed as `log(Close[t] / Close[t-1])` — uses only past/current data
- All Close prices positive
- Daily returns within reasonable range (|return| < 100%)

## Acceptance Criteria Status

| Criterion | Status |
|---|---|
| `scripts/prepare_data.py` generates `data/processed/btc_usd_daily.csv` | Pass |
| CSV contains 'Date', 'Close', 'returns' columns | Pass |
| 'returns' column has no NaN values | Pass |

## Strategy Metrics

All strategy metrics (Sharpe ratio, annual return, drawdown, etc.) are set to 0.0 as Phase 2 is data pipeline only. These will be populated in Phase 3 (walk-forward evaluation) and Phase 4 (trading strategy).
