"""
Data integrity tests for the BTC-USD data pipeline.
Verifies data quality, no leakage, and correct preprocessing.
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.data.data_loader import load_btc_data, save_processed_data, PROCESSED_DIR


@pytest.fixture(scope="module")
def btc_data():
    """Load BTC-USD data once for all tests."""
    return load_btc_data(ticker="BTC-USD", interval="1d", period="10y")


class TestDataColumns:
    """Verify required columns exist and have correct types."""

    def test_has_required_columns(self, btc_data):
        required = ["Date", "Close", "returns"]
        for col in required:
            assert col in btc_data.columns, f"Missing column: {col}"

    def test_date_is_datetime(self, btc_data):
        assert pd.api.types.is_datetime64_any_dtype(btc_data["Date"])

    def test_close_is_numeric(self, btc_data):
        assert pd.api.types.is_numeric_dtype(btc_data["Close"])

    def test_returns_is_numeric(self, btc_data):
        assert pd.api.types.is_numeric_dtype(btc_data["returns"])


class TestNoNaN:
    """Verify no NaN values in critical columns."""

    def test_no_nan_in_returns(self, btc_data):
        assert btc_data["returns"].isna().sum() == 0

    def test_no_nan_in_close(self, btc_data):
        assert btc_data["Close"].isna().sum() == 0

    def test_no_nan_in_date(self, btc_data):
        assert btc_data["Date"].isna().sum() == 0


class TestNoLeakage:
    """Verify no future data leakage."""

    def test_no_future_dates(self, btc_data):
        today = pd.Timestamp.now().normalize()
        assert btc_data["Date"].max() <= today

    def test_no_duplicate_dates(self, btc_data):
        assert btc_data["Date"].duplicated().sum() == 0

    def test_dates_are_sorted(self, btc_data):
        assert btc_data["Date"].is_monotonic_increasing

    def test_returns_use_past_data_only(self, btc_data):
        """Log returns at row i use Close[i] and Close[i-1], both known at time i."""
        # Recompute returns from the Close prices in the processed data
        close = btc_data["Close"].values
        expected = np.log(close[1:] / close[:-1])
        actual = btc_data["returns"].values[1:]  # skip first row (uses pre-dataset close)
        np.testing.assert_allclose(actual, expected, rtol=1e-10)


class TestDataQuality:
    """Basic data quality checks."""

    def test_sufficient_rows(self, btc_data):
        assert len(btc_data) >= 1000, f"Only {len(btc_data)} rows"

    def test_close_prices_positive(self, btc_data):
        assert (btc_data["Close"] > 0).all()

    def test_returns_reasonable_range(self, btc_data):
        """Daily log returns should rarely exceed 50%."""
        assert btc_data["returns"].abs().max() < 1.0

    def test_returns_not_constant(self, btc_data):
        assert btc_data["returns"].std() > 0


class TestSaveLoad:
    """Verify save/load round-trip."""

    def test_save_creates_file(self, btc_data, tmp_path):
        output = tmp_path / "test_output.csv"
        # Use save function logic directly
        btc_data.to_csv(output, index=False)
        assert output.exists()

    def test_saved_csv_has_required_columns(self, btc_data, tmp_path):
        output = tmp_path / "test_output.csv"
        btc_data.to_csv(output, index=False)
        loaded = pd.read_csv(output)
        for col in ["Date", "Close", "returns"]:
            assert col in loaded.columns
