"""
Baseline strategies for comparison with the LSTM+GARCH model.
"""
import numpy as np
import pandas as pd

from src.backtest_framework import compute_metrics


class BuyAndHoldBaseline:
    """
    Buy & Hold baseline: always long (+1) throughout the test period.

    This is the simplest benchmark — measures whether the model adds
    value over simply holding the asset.
    """

    def run(self, actual_returns: np.ndarray) -> dict:
        """
        Compute Buy & Hold metrics.

        Args:
            actual_returns: Array of realized returns.

        Returns:
            Dict with performance metrics.
        """
        returns_series = pd.Series(actual_returns)
        metrics = compute_metrics(returns_series)
        cumulative = (1 + returns_series).cumprod()

        return {
            "metrics": metrics,
            "cumulative": cumulative,
            "total_trades": 0,  # No trading
        }
