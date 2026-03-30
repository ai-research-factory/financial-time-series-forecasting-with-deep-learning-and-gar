# Technical Findings — Cycle 3: Walk-Forward Evaluation Framework

## Overview

Phase 3 implements the walk-forward evaluation framework for the LSTM+GARCH hybrid model. This is the core validation infrastructure that enables robust out-of-sample assessment of the model's return and volatility forecasts.

## Implementation

### Components Built

1. **`src/models/lstm_garch.py`** — `LSTMGARCHModel` class
   - Stage 1: LSTM (1 layer, 50 hidden units, lookback=60) predicts conditional mean of returns
   - Stage 2: GARCH(1,1) fits LSTM residuals to forecast conditional volatility
   - Scaler fitted on training data only (no leakage)
   - Uses Adam optimizer with lr=0.001, 50 epochs, batch_size=32

2. **`src/evaluation/validator.py`** — `WalkForwardEvaluator` class
   - Wraps the ARF-provided `WalkForwardValidator` from `src/backtest.py`
   - Expanding window: all splits start from beginning of data, progressively larger training sets
   - 5 splits with gap=1 between train and test (prevents leakage)
   - Computes MSE (return prediction) and MAE (volatility prediction) per split
   - Generates strategy returns using simple long/flat rule (long if predicted return > 0)
   - Transaction costs: 10 bps fee + 5 bps slippage

3. **`src/main.py`** — CLI entry point
   - `--run-walkforward` flag to execute evaluation
   - Configurable: `--n-splits`, `--lookback`, `--epochs`, `--hidden-size`, `--lr`, `--train-ratio`

4. **`tests/test_backtest.py`** — Walk-forward validation tests
   - Verifies no temporal overlap between train/test
   - Verifies train indices strictly before test indices
   - Verifies gap is respected
   - Verifies expanding window property
   - Verifies minimum train size constraint

## Walk-Forward Results

All metrics from `metrics.json` and `walkforward_metrics.json`:

### Per-Split Performance

| Window | Train Period | Test Period | n_train | n_test | MSE | MAE | Sharpe |
|--------|-------------|-------------|---------|--------|-----|-----|--------|
| 1 | 2016-03-31 to 2016-09-06 | 2016-09-08 to 2018-08-06 | 160 | 698 | 0.00204 | 0.00233 | 1.3761 |
| 2 | 2016-03-31 to 2018-08-05 | 2018-08-07 to 2020-07-04 | 858 | 698 | 0.00167 | 0.00216 | -0.0398 |
| 3 | 2016-03-31 to 2020-07-03 | 2020-07-05 to 2022-06-02 | 1556 | 698 | 0.00148 | 0.00167 | 0.5980 |
| 4 | 2016-03-31 to 2022-06-01 | 2022-06-03 to 2024-04-30 | 2254 | 698 | 0.00077 | 0.00108 | 0.6276 |
| 5 | 2016-03-31 to 2024-04-29 | 2024-05-01 to 2026-03-30 | 2952 | 698 | 0.00062 | 0.00083 | -0.1480 |

### Aggregate Metrics (from metrics.json)

| Metric | Value |
|--------|-------|
| Avg Sharpe Ratio | 0.4828 |
| Avg Annual Return | 22.94% |
| Max Drawdown | -72.44% |
| Avg Hit Rate | 35.5% |
| Total Trades | 750 |
| Positive Windows | 3 / 5 (60%) |
| Avg OOS Sharpe (net) | 0.4828 |
| Avg MSE | 0.001316 |
| Avg MAE | 0.001614 |
| Fee (bps) | 10 |
| Slippage (bps) | 5 |

## Key Observations

1. **MSE decreases with more training data**: MSE drops from 0.00204 (160 training samples) to 0.00062 (2952 samples), indicating the LSTM benefits from larger training sets.

2. **Volatility prediction (MAE) also improves**: MAE follows the same trend, from 0.00233 to 0.00083.

3. **3 of 5 windows have positive Sharpe**: Windows 1, 3, and 4 are profitable. Window 2 (2018-2020 bear market) and Window 5 (2024-2026) show negative performance.

4. **High drawdowns across all windows**: Max drawdown of -72.44% indicates the simple long/flat strategy is vulnerable during market downturns. A more sophisticated position sizing using GARCH volatility could help (Phase 4).

5. **First window caveat**: Window 1 has only 160 training samples but shows the highest Sharpe (1.3761). This coincides with the 2016-2018 BTC bull run and likely reflects market regime rather than model quality.

## Leakage Verification

- Scaler fitted on training data only (`scaler.fit(train_returns)`)
- GARCH fitted on training residuals only
- Walk-forward gap=1 between train end and test start
- No centered rolling windows used
- All 7 temporal integrity tests pass
