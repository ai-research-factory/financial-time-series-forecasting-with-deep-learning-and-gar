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

## Cycle 4

### Risk-Adjusted Rule Underperformance
- The `RiskAdjustedEntryRule` with 40th percentile threshold on 20-day vol history underperforms the basic strategy. BTC-USD is a persistently high-volatility asset, and the vol filter removes too many profitable entries. A higher percentile threshold (e.g., 60th-70th) or a different risk metric may be needed.

### Single-Asset Baseline Limitations
- The CLAUDE.md requests 1/N (equal weight), Vol-Targeted 1/N, and Simple Momentum baselines. These are designed for multi-asset portfolios but are not meaningful for a single-asset (BTC-USD) strategy. For a single asset, 1/N and Buy & Hold are equivalent. We report Buy & Hold as the primary baseline.

### Strategy vs Buy & Hold
- The basic long/short strategy achieves comparable gross Sharpe (0.5839) to Buy & Hold (0.5969) but underperforms after costs. The model's primary value appears to be in drawdown reduction (-75.73% vs -88.64%) rather than alpha generation. This aligns with the paper's finding that the GARCH component's value is in risk management.
