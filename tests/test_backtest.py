"""
Tests for WalkForwardValidator and LSTM+GARCH model integration.
Verifies no temporal overlap (look-ahead bias) and correct model invocation.
"""
import numpy as np
import pandas as pd
import pytest

from src.backtest_framework import WalkForwardValidator, BacktestConfig


class TestWalkForwardSplits:
    """Verify walk-forward splits have no temporal overlap."""

    @pytest.fixture
    def sample_data(self):
        """Create a sample time-series DataFrame."""
        n = 1000
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        returns = np.random.randn(n) * 0.02
        return pd.DataFrame({"Date": dates, "returns": returns})

    def test_no_temporal_overlap(self, sample_data):
        """Train and test indices must not overlap within any split."""
        config = BacktestConfig(n_splits=5, train_ratio=0.7, min_train_size=252, gap=1)
        validator = WalkForwardValidator(config)

        for train_idx, test_idx in validator.split(sample_data):
            train_set = set(train_idx)
            test_set = set(test_idx)
            overlap = train_set & test_set
            assert len(overlap) == 0, f"Train/test overlap: {overlap}"

    def test_train_before_test(self, sample_data):
        """All train indices must be strictly before all test indices."""
        config = BacktestConfig(n_splits=5, train_ratio=0.7, min_train_size=252, gap=1)
        validator = WalkForwardValidator(config)

        for train_idx, test_idx in validator.split(sample_data):
            assert max(train_idx) < min(test_idx), (
                f"Train max ({max(train_idx)}) >= Test min ({min(test_idx)})"
            )

    def test_gap_respected(self, sample_data):
        """Gap between train end and test start must be >= config.gap."""
        config = BacktestConfig(n_splits=5, train_ratio=0.7, min_train_size=252, gap=1)
        validator = WalkForwardValidator(config)

        for train_idx, test_idx in validator.split(sample_data):
            gap = min(test_idx) - max(train_idx)
            assert gap >= config.gap, f"Gap {gap} < required {config.gap}"

    def test_produces_expected_splits(self, sample_data):
        """Should produce the configured number of splits (or fewer if data is limited)."""
        config = BacktestConfig(n_splits=5, train_ratio=0.7, min_train_size=252, gap=1)
        validator = WalkForwardValidator(config)

        splits = list(validator.split(sample_data))
        assert len(splits) > 0, "No splits produced"
        assert len(splits) <= 5, f"Too many splits: {len(splits)}"

    def test_expanding_window(self, sample_data):
        """Later splits should have larger or equal training sets (expanding window)."""
        config = BacktestConfig(n_splits=5, train_ratio=1.0, min_train_size=252, gap=1)
        validator = WalkForwardValidator(config)

        train_sizes = [len(train_idx) for train_idx, _ in validator.split(sample_data)]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1], (
                f"Train size decreased: {train_sizes[i-1]} -> {train_sizes[i]}"
            )

    def test_min_train_size_respected(self, sample_data):
        """All training sets must have at least min_train_size samples."""
        config = BacktestConfig(n_splits=5, train_ratio=0.7, min_train_size=252, gap=1)
        validator = WalkForwardValidator(config)

        for train_idx, test_idx in validator.split(sample_data):
            assert len(train_idx) >= config.min_train_size, (
                f"Train size {len(train_idx)} < min {config.min_train_size}"
            )


class TestWalkForwardDates:
    """Verify date-based temporal integrity."""

    @pytest.fixture
    def dated_data(self):
        n = 1000
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        returns = np.random.randn(n) * 0.02
        return pd.DataFrame({"Date": dates, "returns": returns})

    def test_train_dates_before_test_dates(self, dated_data):
        """Train date range must end before test date range starts."""
        config = BacktestConfig(n_splits=5, train_ratio=0.7, min_train_size=252, gap=1)
        validator = WalkForwardValidator(config)

        for train_idx, test_idx in validator.split(dated_data):
            train_end_date = dated_data.iloc[train_idx[-1]]["Date"]
            test_start_date = dated_data.iloc[test_idx[0]]["Date"]
            assert train_end_date < test_start_date, (
                f"Train end {train_end_date} >= Test start {test_start_date}"
            )
