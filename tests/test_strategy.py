"""
Tests for Phase 4 backtest strategy, engine, trading rules, and baselines.
"""
import numpy as np
import pandas as pd
import pytest

from src.backtest.strategy import SignalGenerator
from src.backtest.engine import BacktestEngine
from src.trading_rules import RiskAdjustedEntryRule
from src.evaluation.baselines import BuyAndHoldBaseline


class TestSignalGenerator:
    """Verify SignalGenerator produces correct signals."""

    def test_positive_returns_give_long(self):
        gen = SignalGenerator()
        pred = np.array([0.01, 0.05, 0.001])
        signals = gen.generate(pred)
        assert (signals == 1.0).all()

    def test_negative_returns_give_short(self):
        gen = SignalGenerator()
        pred = np.array([-0.01, -0.05, -0.001])
        signals = gen.generate(pred)
        assert (signals == -1.0).all()

    def test_mixed_signals(self):
        gen = SignalGenerator()
        pred = np.array([0.01, -0.02, 0.03, -0.04])
        signals = gen.generate(pred)
        expected = np.array([1.0, -1.0, 1.0, -1.0])
        np.testing.assert_array_equal(signals, expected)

    def test_generate_series_returns_pandas(self):
        gen = SignalGenerator()
        pred = np.array([0.01, -0.01])
        result = gen.generate_series(pred)
        assert isinstance(result, pd.Series)
        assert result.name == "signal"


class TestBacktestEngine:
    """Verify BacktestEngine computes P&L correctly."""

    def test_gross_returns_are_signal_times_actual(self):
        engine = BacktestEngine(cost_bps=0.0)
        signals = np.array([1.0, -1.0, 1.0])
        actual = np.array([0.02, -0.01, 0.03])
        result = engine.run(signals, actual)
        expected_gross = np.array([0.02, 0.01, 0.03])
        np.testing.assert_allclose(result["gross_returns"].values, expected_gross)

    def test_costs_reduce_returns(self):
        engine = BacktestEngine(cost_bps=5.0)
        signals = np.array([1.0, -1.0, 1.0])
        actual = np.array([0.02, -0.01, 0.03])
        result = engine.run(signals, actual)
        # Signal changes at every step, so costs should be deducted
        assert result["total_costs"] > 0
        # Net cumulative should be less than gross cumulative
        assert result["net_cumulative"].iloc[-1] < result["gross_cumulative"].iloc[-1]

    def test_net_sharpe_leq_gross_sharpe(self):
        """Net Sharpe must be <= Gross Sharpe (acceptance criterion 2)."""
        engine_gross = BacktestEngine(cost_bps=0.0)
        engine_net = BacktestEngine(cost_bps=5.0)
        np.random.seed(42)
        signals = np.sign(np.random.randn(500))
        actual = np.random.randn(500) * 0.02
        gross = engine_gross.run(signals, actual)
        net = engine_net.run(signals, actual)
        assert net["net_metrics"]["sharpeRatio"] <= gross["gross_metrics"]["sharpeRatio"]

    def test_no_trades_no_costs(self):
        engine = BacktestEngine(cost_bps=10.0)
        signals = np.ones(100)  # Always long, no changes
        actual = np.random.randn(100) * 0.01
        result = engine.run(signals, actual)
        # Only the initial entry costs something (signal goes from 0 to 1)
        assert result["total_trades"] == 1


class TestRiskAdjustedEntryRule:
    """Verify RiskAdjustedEntryRule filters by volatility."""

    def test_filters_high_vol_entries(self):
        rule = RiskAdjustedEntryRule(vol_lookback=5, vol_percentile=40.0)
        pred_returns = np.array([0.01] * 20)  # All positive
        # Low vol for first 10, then high vol
        pred_vol = np.concatenate([np.ones(10) * 0.01, np.ones(10) * 0.10])
        signals = rule.generate(pred_returns, pred_vol)
        # High vol period should have more flat signals
        high_vol_flat = (signals[10:] == 0.0).sum()
        assert high_vol_flat > 0, "Should filter some entries in high vol regime"

    def test_output_shape_matches_input(self):
        rule = RiskAdjustedEntryRule()
        n = 50
        pred_returns = np.random.randn(n) * 0.01
        pred_vol = np.abs(np.random.randn(n)) * 0.02
        signals = rule.generate(pred_returns, pred_vol)
        assert len(signals) == n

    def test_signals_are_valid_values(self):
        rule = RiskAdjustedEntryRule()
        pred_returns = np.random.randn(100) * 0.01
        pred_vol = np.abs(np.random.randn(100)) * 0.02
        signals = rule.generate(pred_returns, pred_vol)
        valid = {-1.0, 0.0, 1.0}
        assert all(s in valid for s in signals)


class TestBuyAndHoldBaseline:
    """Verify Buy & Hold baseline."""

    def test_returns_metrics(self):
        bh = BuyAndHoldBaseline()
        actual = np.random.randn(500) * 0.02
        result = bh.run(actual)
        assert "metrics" in result
        assert "sharpeRatio" in result["metrics"]
        assert result["total_trades"] == 0

    def test_cumulative_matches_asset(self):
        bh = BuyAndHoldBaseline()
        actual = np.array([0.01, 0.02, -0.01])
        result = bh.run(actual)
        expected_cum = (1 + pd.Series(actual)).cumprod()
        pd.testing.assert_series_equal(
            result["cumulative"].reset_index(drop=True),
            expected_cum.reset_index(drop=True),
        )
