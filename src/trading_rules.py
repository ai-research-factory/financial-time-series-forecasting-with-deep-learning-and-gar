"""
Trading rules for LSTM+GARCH strategy.

Implements risk-adjusted entry rules using GARCH volatility forecasts.
"""
import numpy as np


class RiskAdjustedEntryRule:
    """
    Risk-adjusted entry rule using GARCH volatility.

    Generates a long signal only when:
      1. LSTM predicted return > 0, AND
      2. GARCH predicted volatility < 40th percentile of past 20-day volatility

    Otherwise, position is flat (0).
    This filters out entries during high-volatility regimes.
    """

    def __init__(self, vol_lookback: int = 20, vol_percentile: float = 40.0):
        """
        Args:
            vol_lookback: Number of past days to compute volatility percentile.
            vol_percentile: Percentile threshold (0-100). Only enter if current
                           predicted vol is below this percentile of recent vol.
        """
        self.vol_lookback = vol_lookback
        self.vol_percentile = vol_percentile

    def generate(
        self,
        pred_returns: np.ndarray,
        pred_vol: np.ndarray,
    ) -> np.ndarray:
        """
        Generate risk-adjusted signals.

        Args:
            pred_returns: LSTM predicted returns.
            pred_vol: GARCH predicted volatility.

        Returns:
            Array of signals: +1 (long), -1 (short), 0 (flat).
        """
        n = len(pred_returns)
        signals = np.zeros(n)

        for i in range(n):
            # Compute volatility threshold from past vol_lookback predictions
            start = max(0, i - self.vol_lookback + 1)
            vol_window = pred_vol[start:i + 1]

            if len(vol_window) < 2:
                # Not enough history; use basic signal
                if pred_returns[i] > 0:
                    signals[i] = 1.0
                elif pred_returns[i] < 0:
                    signals[i] = -1.0
                continue

            vol_threshold = np.percentile(vol_window, self.vol_percentile)

            if pred_returns[i] > 0 and pred_vol[i] < vol_threshold:
                signals[i] = 1.0
            elif pred_returns[i] < 0 and pred_vol[i] < vol_threshold:
                signals[i] = -1.0
            # else: flat (0) — high volatility regime, stay out

        return signals
