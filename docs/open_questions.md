# Open Questions

## Cycle 2

### Data Coverage
- The ARF Data API returned ~3651 daily rows for BTC-USD with `period=10y`, covering 2016-03-31 to 2026-03-30. This is consistent with the paper's intended dataset.

### API vs yfinance
- The task specifies `yfinance.download('BTC-USD', period='10y', interval='1d')`, but CLAUDE.md mandates using the ARF Data API. The API is backed by yfinance data, so the result is equivalent. Used ARF Data API as required.
