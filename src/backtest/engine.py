"""
Backtest engine for computing portfolio P&L with transaction costs.

Takes walk-forward signals and returns, computes gross and net performance.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.backtest_framework import compute_metrics


class BacktestEngine:
    """
    Computes portfolio P&L from signals and actual returns.

    Supports:
      - Gross P&L (no costs)
      - Net P&L (with transaction costs deducted on signal changes)
      - Performance metrics (Sharpe, annual return, max drawdown)
      - P&L curve plotting
    """

    def __init__(self, cost_bps: float = 5.0):
        """
        Args:
            cost_bps: Transaction cost in basis points, applied on each signal change.
        """
        self.cost_bps = cost_bps

    def run(
        self,
        signals: np.ndarray,
        actual_returns: np.ndarray,
        dates: np.ndarray = None,
    ) -> dict:
        """
        Run backtest on signals and actual returns.

        Args:
            signals: Array of positions (+1, -1, 0).
            actual_returns: Array of realized returns.
            dates: Optional date array for indexing.

        Returns:
            Dict with gross/net metrics, P&L series, and trade count.
        """
        signals = np.asarray(signals, dtype=float)
        actual_returns = np.asarray(actual_returns, dtype=float)

        # Gross strategy returns: position * actual return
        gross_returns = signals * actual_returns

        # Transaction costs: applied when signal changes
        signal_changes = np.abs(np.diff(signals, prepend=0))
        cost_per_change = self.cost_bps / 10000.0
        costs = signal_changes * cost_per_change

        # Net returns
        net_returns = gross_returns - costs

        # Build series
        index = pd.RangeIndex(len(signals)) if dates is None else dates
        gross_series = pd.Series(gross_returns, index=index)
        net_series = pd.Series(net_returns, index=index)

        # Compute metrics
        gross_metrics = compute_metrics(gross_series)
        net_metrics = compute_metrics(net_series)

        # Trade count: number of signal changes
        total_trades = int(signal_changes.sum())

        # Cumulative P&L
        gross_cumulative = (1 + gross_series).cumprod()
        net_cumulative = (1 + net_series).cumprod()

        return {
            "gross_metrics": gross_metrics,
            "net_metrics": net_metrics,
            "gross_returns": gross_series,
            "net_returns": net_series,
            "gross_cumulative": gross_cumulative,
            "net_cumulative": net_cumulative,
            "total_trades": total_trades,
            "total_costs": float(costs.sum()),
        }

    def plot_pnl(self, result: dict, output_path: Path, title: str = "Cumulative P&L"):
        """
        Plot gross and net cumulative P&L curves.

        Args:
            result: Output from self.run().
            output_path: Path to save the PNG file.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(result["gross_cumulative"].values, label="Gross P&L", linewidth=1.5)
        ax.plot(result["net_cumulative"].values, label="Net P&L (after costs)", linewidth=1.5)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Cumulative Return")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
