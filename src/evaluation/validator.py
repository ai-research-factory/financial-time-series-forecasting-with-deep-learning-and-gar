"""
Walk-Forward Validator for LSTM+GARCH model evaluation.

Implements expanding-window walk-forward validation with MSE and MAE metrics.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from src.backtest_framework import WalkForwardValidator, BacktestConfig, BacktestResult, compute_metrics, calculate_costs, generate_metrics_json
from src.models.lstm_garch import LSTMGARCHModel
from src.data.data_loader import load_btc_data, PROJECT_ROOT


class WalkForwardEvaluator:
    """
    Orchestrates walk-forward evaluation of the LSTM+GARCH model.

    For each split:
      1. Train LSTM on training returns.
      2. Predict on test returns (with lookback context from train).
      3. Compute MSE (return prediction) and MAE (volatility prediction).
      4. Collect BacktestResults for ARF metrics generation.
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_ratio: float = 1.0,
        lookback: int = 60,
        lstm_epochs: int = 50,
        lstm_hidden: int = 50,
        lstm_lr: float = 0.001,
    ):
        self.config = BacktestConfig(
            n_splits=n_splits,
            train_ratio=train_ratio,
            min_train_size=lookback + 100,  # Enough for LSTM sequences + some margin
            gap=1,
        )
        self.validator = WalkForwardValidator(self.config)
        self.lookback = lookback
        self.lstm_epochs = lstm_epochs
        self.lstm_hidden = lstm_hidden
        self.lstm_lr = lstm_lr

    def run(self, df: pd.DataFrame) -> dict:
        """
        Execute walk-forward evaluation.

        Args:
            df: DataFrame with 'Date' and 'returns' columns.

        Returns:
            Dict with per-split metrics and aggregate results.
        """
        returns = df["returns"].values
        dates = df["Date"].values

        split_results = []
        backtest_results = []

        for window_idx, (train_idx, test_idx) in enumerate(self.validator.split(df)):
            print(f"\n=== Walk-Forward Window {window_idx + 1} ===")
            print(f"  Train: idx {train_idx[0]}-{train_idx[-1]} "
                  f"({pd.Timestamp(dates[train_idx[0]]).strftime('%Y-%m-%d')} to "
                  f"{pd.Timestamp(dates[train_idx[-1]]).strftime('%Y-%m-%d')})")
            print(f"  Test:  idx {test_idx[0]}-{test_idx[-1]} "
                  f"({pd.Timestamp(dates[test_idx[0]]).strftime('%Y-%m-%d')} to "
                  f"{pd.Timestamp(dates[test_idx[-1]]).strftime('%Y-%m-%d')})")

            train_returns = returns[train_idx]
            test_returns_full = returns[train_idx[-self.lookback:]] if len(train_idx) >= self.lookback else returns[train_idx]
            test_returns_full = np.concatenate([
                returns[max(train_idx[-1] - self.lookback + 1, train_idx[0]):train_idx[-1] + 1],
                returns[test_idx],
            ])

            # Train model
            model = LSTMGARCHModel(
                lookback=self.lookback,
                hidden_size=self.lstm_hidden,
                epochs=self.lstm_epochs,
                lr=self.lstm_lr,
            )
            model.fit(train_returns)

            # Predict on test (with lookback context)
            pred_returns, pred_vol = model.predict(test_returns_full)

            # Align: predictions correspond to the test portion
            n_test = len(test_idx)
            if len(pred_returns) > n_test:
                pred_returns = pred_returns[-n_test:]
                pred_vol = pred_vol[-n_test:]
            elif len(pred_returns) < n_test:
                n_test = len(pred_returns)

            actual_returns = returns[test_idx[:n_test]]

            # MSE for return prediction
            mse = float(np.mean((actual_returns - pred_returns) ** 2))

            # MAE for volatility: compare predicted vol to realized squared residuals
            residuals_sq = (actual_returns - pred_returns) ** 2
            mae = float(np.mean(np.abs(residuals_sq - pred_vol ** 2)))

            print(f"  MSE (returns): {mse:.8f}")
            print(f"  MAE (volatility): {mae:.8f}")

            # Strategy: long if predicted return > 0, else flat
            positions = pd.Series(np.where(pred_returns > 0, 1.0, 0.0))
            strategy_returns = pd.Series(actual_returns * positions.values)
            net_returns = calculate_costs(strategy_returns, positions, self.config)
            metrics = compute_metrics(net_returns)

            split_results.append({
                "window": window_idx + 1,
                "train_start": str(pd.Timestamp(dates[train_idx[0]]).date()),
                "train_end": str(pd.Timestamp(dates[train_idx[-1]]).date()),
                "test_start": str(pd.Timestamp(dates[test_idx[0]]).date()),
                "test_end": str(pd.Timestamp(dates[test_idx[-1]]).date()),
                "n_train": len(train_idx),
                "n_test": n_test,
                "mse": mse,
                "mae": mae,
                "sharpe": metrics["sharpeRatio"],
                "annual_return": metrics["annualReturn"],
                "max_drawdown": metrics["maxDrawdown"],
                "hit_rate": metrics["hitRate"],
            })

            backtest_results.append(BacktestResult(
                window=window_idx + 1,
                train_start=str(pd.Timestamp(dates[train_idx[0]]).date()),
                train_end=str(pd.Timestamp(dates[train_idx[-1]]).date()),
                test_start=str(pd.Timestamp(dates[test_idx[0]]).date()),
                test_end=str(pd.Timestamp(dates[test_idx[-1]]).date()),
                gross_sharpe=metrics["sharpeRatio"],
                net_sharpe=metrics["sharpeRatio"],
                annual_return=metrics["annualReturn"],
                max_drawdown=metrics["maxDrawdown"],
                total_trades=int(positions.diff().abs().sum()),
                hit_rate=metrics["hitRate"],
            ))

        # Generate aggregate metrics
        custom_metrics = {
            "phase": "3-walkforward-evaluation",
            "n_splits": len(split_results),
            "per_split_mse": [s["mse"] for s in split_results],
            "per_split_mae": [s["mae"] for s in split_results],
            "avg_mse": float(np.mean([s["mse"] for s in split_results])) if split_results else 0.0,
            "avg_mae": float(np.mean([s["mae"] for s in split_results])) if split_results else 0.0,
            "lookback": self.lookback,
            "lstm_hidden": self.lstm_hidden,
            "lstm_epochs": self.lstm_epochs,
        }

        metrics_json = generate_metrics_json(backtest_results, self.config, custom_metrics)

        return {
            "splits": split_results,
            "metrics_json": metrics_json,
            "walkforward_metrics": {
                "n_splits": len(split_results),
                "splits": split_results,
                "aggregate": {
                    "avg_mse": custom_metrics["avg_mse"],
                    "avg_mae": custom_metrics["avg_mae"],
                },
            },
        }


def save_walkforward_results(results: dict, output_dir: Optional[Path] = None) -> tuple[Path, Path]:
    """Save walk-forward results to reports/cycle_3/."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "reports" / "cycle_3"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save walkforward_metrics.json
    wf_path = output_dir / "walkforward_metrics.json"
    with open(wf_path, "w") as f:
        json.dump(results["walkforward_metrics"], f, indent=2)

    # Save metrics.json (ARF standard)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results["metrics_json"], f, indent=2)

    return wf_path, metrics_path
