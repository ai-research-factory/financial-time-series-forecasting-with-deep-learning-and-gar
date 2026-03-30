# Preflight Check — Cycle 4

## 1. Data Boundary Table

| Item | Value |
|---|---|
| Data acquisition end date | 2026-03-30 (before today) |
| Train period | 2016-03-31 ~ varies per walk-forward window |
| Validation period | N/A (walk-forward OOS used instead) |
| Test period | Varies per walk-forward window ~ 2026-03-30 |
| No overlap confirmed | Yes |
| No future dates confirmed | Yes |

## 2. Feature Timestamp Contract

- All features use data at t-1 or earlier for prediction at time t? → Yes
- Scaler / Imputer fitted on train data only? → Yes (`scaler.fit_transform(train_returns)` in `lstm_garch.py:73`)
- No centered rolling windows used? → Yes (no `center=True` anywhere)

## 3. Paper Spec Difference Table

| Parameter | Paper Value | Current Implementation | Match? |
|---|---|---|---|
| Universe | BTC-USD | BTC-USD | Yes |
| Lookback period | 60 days | 60 days | Yes |
| Rebalance frequency | Daily | Daily | Yes |
| Features | Log returns | Log returns | Yes |
| Cost model | Transaction costs | 10bps fee + 5bps slippage | Yes |
| LSTM hidden size | 50 | 50 | Yes |
| LSTM epochs | 50 | 50 | Yes |
| GARCH specification | GARCH(1,1) | GARCH(1,1) | Yes |
| Volatility model | GARCH on LSTM residuals | GARCH on LSTM residuals | Yes |
| Walk-forward | Expanding window | Expanding window, 5 splits | Yes |
| Signal rule | Long if pred_return > k*pred_vol | Long/Short + RiskAdjusted (Phase 4) | Yes |

## 4. Phase 4 Specific Checks

- Backtest engine will use walk-forward predictions from existing LSTM+GARCH model
- Transaction costs: 5bps per signal change (as specified in task)
- Gross and Net P&L will be computed separately
- Signal: Long (+1) if pred_return > 0, Short (-1) if pred_return < 0
- RiskAdjustedEntryRule: Long only if pred_return > 0 AND pred_vol < 40th percentile of 20-day vol history
