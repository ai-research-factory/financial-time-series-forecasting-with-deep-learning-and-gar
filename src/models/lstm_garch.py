"""
LSTM + GARCH(1,1) hybrid model for financial time-series forecasting.

Stage 1: LSTM predicts one-step-ahead conditional mean (expected return).
Stage 2: GARCH(1,1) fits LSTM residuals to forecast conditional volatility.
"""
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from arch import arch_model


class LSTMNet(nn.Module):
    """LSTM network for return prediction."""

    def __init__(self, input_size: int = 1, hidden_size: int = 50, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class LSTMGARCHModel:
    """
    Hybrid LSTM + GARCH(1,1) model.

    The LSTM forecasts the conditional mean of returns.
    GARCH(1,1) models the volatility of LSTM residuals.
    """

    def __init__(
        self,
        lookback: int = 60,
        hidden_size: int = 50,
        num_layers: int = 1,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.001,
    ):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self.scaler = StandardScaler()
        self.net = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_sequences(self, returns: np.ndarray):
        """Create lookback sequences from return series."""
        X, y = [], []
        for i in range(self.lookback, len(returns)):
            X.append(returns[i - self.lookback : i])
            y.append(returns[i])
        return np.array(X), np.array(y)

    def fit(self, train_returns: np.ndarray):
        """
        Train LSTM on return series and fit GARCH on residuals.

        Args:
            train_returns: 1D array of training returns.
        """
        # Fit scaler on training data only
        scaled = self.scaler.fit_transform(train_returns.reshape(-1, 1)).flatten()

        X, y = self._create_sequences(scaled)
        if len(X) == 0:
            raise ValueError("Not enough data for given lookback period")

        X_t = torch.FloatTensor(X).unsqueeze(-1).to(self.device)
        y_t = torch.FloatTensor(y).to(self.device)

        self.net = LSTMNet(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.net.train()
        for _ in range(self.epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self.net(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

        # Compute training residuals for GARCH
        self.net.eval()
        with torch.no_grad():
            train_pred_scaled = self.net(X_t).cpu().numpy()

        train_pred = self.scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).flatten()
        actual = train_returns[self.lookback :]
        self.train_residuals = actual - train_pred

        # Fit GARCH(1,1) on residuals
        self._fit_garch(self.train_residuals)

    def _fit_garch(self, residuals: np.ndarray):
        """Fit GARCH(1,1) to residuals."""
        # Scale residuals for numerical stability
        self.resid_scale = max(np.std(residuals), 1e-8)
        scaled_resid = residuals / self.resid_scale * 100  # percentage scale

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            garch = arch_model(scaled_resid, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
            self.garch_result = garch.fit(disp="off", show_warning=False)

    def predict(self, test_returns: np.ndarray):
        """
        Generate predictions for test period.

        Uses the trained LSTM to predict returns and GARCH for volatility.
        The model makes predictions using a rolling window that includes
        prior data for context.

        Args:
            test_returns: 1D array of test returns (must have lookback
                         context prepended from training data).

        Returns:
            pred_returns: Predicted returns for each test step.
            pred_vol: Predicted volatility for each test step.
        """
        if self.net is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        scaled = self.scaler.transform(test_returns.reshape(-1, 1)).flatten()

        X, _ = self._create_sequences(scaled)
        if len(X) == 0:
            return np.array([]), np.array([])

        X_t = torch.FloatTensor(X).unsqueeze(-1).to(self.device)

        self.net.eval()
        with torch.no_grad():
            pred_scaled = self.net(X_t).cpu().numpy()

        pred_returns = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

        # GARCH volatility forecast from residuals
        actual = test_returns[self.lookback :]
        residuals = actual - pred_returns

        # Combine train residuals with new residuals for GARCH forecasting
        all_residuals = np.concatenate([self.train_residuals, residuals])
        pred_vol = self._forecast_volatility(all_residuals, len(residuals))

        return pred_returns, pred_vol

    def _forecast_volatility(self, all_residuals: np.ndarray, n_test: int) -> np.ndarray:
        """Forecast volatility using GARCH for each test step."""
        scaled_resid = all_residuals / self.resid_scale * 100
        vols = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(n_test):
                end_idx = len(all_residuals) - n_test + i
                window = scaled_resid[:end_idx]
                try:
                    garch = arch_model(window, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
                    res = garch.fit(disp="off", show_warning=False)
                    forecast = res.forecast(horizon=1)
                    vol = np.sqrt(forecast.variance.iloc[-1, 0]) * self.resid_scale / 100
                except Exception:
                    vol = np.std(all_residuals[:end_idx]) if end_idx > 0 else 1e-8
                vols.append(vol)

        return np.array(vols)
