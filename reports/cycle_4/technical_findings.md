# Technical Findings — Cycle 4: Trading Strategy and Cost Model

## Implementation Summary

Phase 4 implements a complete backtest framework for the LSTM+GARCH model with:
1. **SignalGenerator** (`src/backtest/strategy.py`): Basic long/short signals from predicted returns
2. **BacktestEngine** (`src/backtest/engine.py`): P&L computation with transaction cost model
3. **RiskAdjustedEntryRule** (`src/trading_rules.py`): GARCH volatility-filtered entries
4. **BuyAndHoldBaseline** (`src/evaluation/baselines.py`): Baseline comparison

## Strategy Definitions

### Basic Long/Short
- Long (+1) if predicted return > 0
- Short (-1) if predicted return < 0

### Risk-Adjusted (RiskAdjustedEntryRule)
- Long (+1) if predicted return > 0 AND predicted volatility < 40th percentile of past 20-day vol
- Short (-1) if predicted return < 0 AND predicted volatility < 40th percentile of past 20-day vol
- Flat (0) otherwise (high volatility regime — stay out)

### Buy & Hold
- Always long (+1), no trading

## Cost Model
- Transaction cost: 5 bps applied on each signal change
- Applied to the absolute change in position (e.g., long-to-short = 2x cost)

## Performance Comparison

All metrics from `metrics.json` and `backtest_summary.json`:

| Metric | Basic (Gross) | Basic (Net) | Risk-Adjusted (Gross) | Risk-Adjusted (Net) | Buy & Hold |
|---|---|---|---|---|---|
| Sharpe Ratio | 0.5839 | 0.4641 | 0.1754 | 0.0521 | 0.5969 |
| Annual Return | 18.17% | 10.41% | -0.20% | -4.56% | 18.93% |
| Max Drawdown | -75.73% | -80.68% | -69.33% | -75.36% | -88.64% |
| Hit Rate | 50.40% | 49.91% | 25.24% | 25.04% | 52.35% |
| Total Trades | 1,881 | 1,881 | 1,236 | 1,236 | 0 |

## Key Observations

### 1. Basic Strategy vs Buy & Hold
- The basic long/short strategy (gross Sharpe 0.5839) is close to Buy & Hold (0.5969) but slightly underperforms
- After costs (net Sharpe 0.4641), the gap widens — the model does not convincingly outperform Buy & Hold
- The strategy does offer a lower max drawdown (-75.73% vs -88.64%), indicating some risk management value from the short signals

### 2. Risk-Adjusted Strategy Underperformance
- The RiskAdjustedEntryRule significantly reduces both returns and Sharpe ratio
- Gross Sharpe drops from 0.5839 (basic) to 0.1754 (risk-adjusted)
- This is because the volatility filter removes ~40% of trades (1,236 vs 1,881), including many profitable entries
- The 40th percentile threshold is too aggressive for BTC-USD, which has persistent high volatility
- Max drawdown does improve slightly (-69.33% vs -75.73%), confirming the vol filter provides some risk reduction

### 3. Cost Impact
- Basic strategy: Sharpe drops from 0.5839 to 0.4641 (20.5% reduction) with 5bps costs
- Risk-adjusted: Sharpe drops from 0.1754 to 0.0521 (70.3% reduction) — costs are proportionally more damaging to a low-return strategy
- Total costs: 0.9405 for basic (1,881 trades), 0.618 for risk-adjusted (1,236 trades)

### 4. Defeat Analysis (vs Buy & Hold Baseline)
The LSTM+GARCH strategy **loses** to Buy & Hold on:
- **Sharpe ratio**: -0.5448 difference (net risk-adjusted vs Buy & Hold)
- **Annual return**: -23.49% difference
- **Hit rate**: 25.04% vs 52.35%

The strategy **wins** on:
- **Max drawdown**: +13.28% improvement (less severe drawdowns due to flat positions)

### 5. Walk-Forward Window Analysis

| Window | Train Period | Test Period | MSE | MAE |
|---|---|---|---|---|
| 1 | 2016-03-31 to 2016-09-06 | 2016-09-08 to 2018-08-06 | 0.002051 | 0.002348 |
| 2 | 2016-03-31 to 2018-08-05 | 2018-08-07 to 2020-07-04 | 0.001648 | 0.002135 |
| 3 | 2016-03-31 to 2020-07-03 | 2020-07-05 to 2022-06-02 | 0.001421 | 0.001654 |
| 4 | 2016-03-31 to 2022-06-01 | 2022-06-03 to 2024-04-30 | 0.000795 | 0.001079 |
| 5 | 2016-03-31 to 2024-04-29 | 2024-05-01 to 2026-03-30 | 0.000619 | 0.000834 |

MSE and MAE consistently improve with more training data, confirming the expanding window approach is beneficial for model accuracy.

## Conclusions

1. The hybrid LSTM+GARCH model generates signals with marginal predictive value (basic gross Sharpe 0.5839), but does not outperform a simple Buy & Hold strategy after costs
2. The GARCH-based volatility filter reduces drawdowns but at the expense of capturing profitable trends — the vol threshold may need tuning for the BTC-USD volatility regime
3. Transaction costs are a significant drag, especially for the more active risk-adjusted strategy
4. The model's primary value may lie in risk management (reduced drawdowns) rather than alpha generation
