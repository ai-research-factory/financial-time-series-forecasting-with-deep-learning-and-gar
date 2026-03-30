# Preflight Check — Cycle 2 (Phase 2: Real Data Pipeline)

## 1. Data Boundary Table

| Item | Value |
|---|---|
| Data Source | ARF Data API (BTC-USD via yfinance cache) |
| Data Interval | 1d (daily) |
| Data Period Requested | 10y |
| Data Acquisition End Date | Before 2026-03-30 (today) |
| Train Period | N/A (Phase 2 is data pipeline only, no model training) |
| Validation Period | N/A |
| Test Period | N/A |
| Duplicate Check | Yes — enforced via timestamp dedup in data_loader.py |
| No Future Dates | Yes — enforced via filter in data_loader.py |

**Note**: Phase 2 builds the data pipeline. Train/Val/Test splits will be defined in Phase 3 (walk-forward evaluation).

## 2. Feature Timestamp Contract

- All features use data at time t-1 or earlier for predictions at time t? → **Yes** (log returns use `Close[t]/Close[t-1]`, which is known at end of day t)
- Scaler/Imputer fit on train data only? → **N/A** (no scaling in Phase 2; will enforce in Phase 3)
- No centered rolling windows? → **Yes** (no rolling windows used in Phase 2)

## 3. Paper Spec Difference Table

| Parameter | Paper Value | Current Implementation | Match? |
|---|---|---|---|
| Universe | BTC-USD daily | BTC-USD daily from ARF Data API | Yes |
| Data Source | yfinance | ARF Data API (yfinance-backed) | Yes |
| Period | ~10 years | 10y period request | Yes |
| Features (Phase 2) | Log returns of Close | `log(Close/Close.shift(1))` | Yes |
| NaN Handling | Drop first row | Drop rows with NaN returns | Yes |
| Lookback Period | Not specified for data loading | N/A | N/A |
| Rebalance Frequency | Daily | N/A (Phase 2 is data only) | N/A |
| Cost Model | Not applicable in Phase 2 | N/A | N/A |
