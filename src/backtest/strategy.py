"""
Signal generation for the LSTM+GARCH trading strategy.

Converts model predictions into trading signals:
  - Long (+1) if predicted return > 0
  - Short (-1) if predicted return < 0
"""
import numpy as np
import pandas as pd


class SignalGenerator:
    """
    Generates trading signals from LSTM+GARCH model predictions.

    Signals:
      +1 (Long)  when predicted_return > 0
      -1 (Short) when predicted_return < 0
       0 (Flat)  when predicted_return == 0 (rare for continuous predictions)
    """

    def generate(self, pred_returns: np.ndarray) -> np.ndarray:
        """
        Generate long/short signals from predicted returns.

        Args:
            pred_returns: Array of predicted returns from LSTM.

        Returns:
            Array of signals: +1 (long), -1 (short), 0 (flat).
        """
        signals = np.sign(pred_returns)
        return signals

    def generate_series(self, pred_returns: np.ndarray, index=None) -> pd.Series:
        """Generate signals as a pandas Series."""
        signals = self.generate(pred_returns)
        return pd.Series(signals, index=index, name="signal")
