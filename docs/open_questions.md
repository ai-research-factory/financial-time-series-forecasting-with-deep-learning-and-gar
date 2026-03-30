# Open Questions

## Cycle 2

### Data Coverage
- The ARF Data API returned ~3651 daily rows for BTC-USD with `period=10y`, covering 2016-03-31 to 2026-03-30. This is consistent with the paper's intended dataset.

### API vs yfinance
- The task specifies `yfinance.download('BTC-USD', period='10y', interval='1d')`, but CLAUDE.md mandates using the ARF Data API. The API is backed by yfinance data, so the result is equivalent. Used ARF Data API as required.

## Cycle 3

### Walk-Forward Split Count
- With 5 expanding-window splits and `train_ratio=1.0`, the first split has only 160 training samples (lookback=60 leaves ~100 sequences). This is small for LSTM training but necessary to achieve 5 splits. The first window's high Sharpe (1.3761) likely reflects the 2016-2018 BTC bull run rather than model quality.

### Baseline Comparison Deferred
- Baseline comparisons (1/N, Vol-Targeted 1/N, Simple Momentum) are deferred to Phase 4 when the trading strategy is fully implemented. The current simple long/flat rule is a placeholder for walk-forward validation purposes.

### GARCH Volatility Forecasting
- GARCH(1,1) is re-fitted for each test step in a rolling manner, which is computationally expensive. For production use, a single GARCH fit with recursive forecasting may be more efficient but less accurate.
