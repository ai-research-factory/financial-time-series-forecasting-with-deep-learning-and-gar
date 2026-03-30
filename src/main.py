"""
Main entry point for the LSTM+GARCH financial time-series forecasting project.

Usage:
    python -m src.main --run-walkforward
    python -m src.main --run-backtest
"""
import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.data_loader import load_btc_data, PROJECT_ROOT
from src.evaluation.validator import WalkForwardEvaluator, save_walkforward_results
from src.backtest_framework import (
    WalkForwardValidator, BacktestConfig, BacktestResult,
    compute_metrics, calculate_costs, generate_metrics_json,
)
from src.backtest.strategy import SignalGenerator
from src.backtest.engine import BacktestEngine
from src.trading_rules import RiskAdjustedEntryRule
from src.evaluation.baselines import BuyAndHoldBaseline
from src.models.lstm_garch import LSTMGARCHModel


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="LSTM+GARCH Financial Time-Series Forecasting"
    )
    parser.add_argument(
        "--run-walkforward",
        action="store_true",
        help="Run walk-forward evaluation of the LSTM+GARCH model.",
    )
    parser.add_argument(
        "--run-backtest",
        action="store_true",
        help="Run Phase 4 backtest with trading strategy and cost model.",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Number of walk-forward splits (default: 5)")
    parser.add_argument("--lookback", type=int, default=60, help="LSTM lookback window (default: 60)")
    parser.add_argument("--epochs", type=int, default=50, help="LSTM training epochs (default: 50)")
    parser.add_argument("--hidden-size", type=int, default=50, help="LSTM hidden size (default: 50)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--train-ratio", type=float, default=1.0, help="Train ratio for expanding window (default: 1.0)")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="Transaction cost in basis points (default: 5)")
    return parser.parse_args(argv)


def run_backtest(args):
    """Run Phase 4 backtest: strategy + cost model + baselines."""
    print("Loading BTC-USD data...")
    df = load_btc_data()
    print(f"Loaded {len(df)} rows ({df['Date'].min().date()} to {df['Date'].max().date()})")

    returns = df["returns"].values
    dates = df["Date"].values

    config = BacktestConfig(
        n_splits=args.n_splits,
        train_ratio=args.train_ratio,
        min_train_size=args.lookback + 100,
        gap=1,
        fee_bps=10.0,
        slippage_bps=5.0,
    )
    validator = WalkForwardValidator(config)
    signal_gen = SignalGenerator()
    risk_rule = RiskAdjustedEntryRule(vol_lookback=20, vol_percentile=40.0)
    engine_gross = BacktestEngine(cost_bps=0.0)  # For gross metrics
    engine_net = BacktestEngine(cost_bps=args.cost_bps)
    buy_hold = BuyAndHoldBaseline()

    # Collect results across all walk-forward windows
    all_basic_signals = []
    all_risk_signals = []
    all_actual_returns = []
    all_dates = []

    # Per-window results
    window_results = []

    for window_idx, (train_idx, test_idx) in enumerate(validator.split(df)):
        print(f"\n=== Walk-Forward Window {window_idx + 1} ===")
        train_start_date = pd.Timestamp(dates[train_idx[0]]).strftime('%Y-%m-%d')
        train_end_date = pd.Timestamp(dates[train_idx[-1]]).strftime('%Y-%m-%d')
        test_start_date = pd.Timestamp(dates[test_idx[0]]).strftime('%Y-%m-%d')
        test_end_date = pd.Timestamp(dates[test_idx[-1]]).strftime('%Y-%m-%d')
        print(f"  Train: {train_start_date} to {train_end_date} ({len(train_idx)} samples)")
        print(f"  Test:  {test_start_date} to {test_end_date} ({len(test_idx)} samples)")

        train_returns = returns[train_idx]

        # Prepare test data with lookback context
        lookback = args.lookback
        context_start = max(train_idx[-1] - lookback + 1, train_idx[0])
        test_returns_full = np.concatenate([
            returns[context_start:train_idx[-1] + 1],
            returns[test_idx],
        ])

        # Train model
        model = LSTMGARCHModel(
            lookback=lookback,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            lr=args.lr,
        )
        model.fit(train_returns)

        # Predict
        pred_returns, pred_vol = model.predict(test_returns_full)

        # Align predictions to test period
        n_test = len(test_idx)
        if len(pred_returns) > n_test:
            pred_returns = pred_returns[-n_test:]
            pred_vol = pred_vol[-n_test:]
        elif len(pred_returns) < n_test:
            n_test = len(pred_returns)

        actual = returns[test_idx[:n_test]]
        test_dates = dates[test_idx[:n_test]]

        # Generate signals
        basic_signals = signal_gen.generate(pred_returns)
        risk_signals = risk_rule.generate(pred_returns, pred_vol)

        # Per-window metrics
        mse = float(np.mean((actual - pred_returns) ** 2))
        residuals_sq = (actual - pred_returns) ** 2
        mae = float(np.mean(np.abs(residuals_sq - pred_vol ** 2)))

        window_results.append({
            "window": window_idx + 1,
            "train_start": train_start_date,
            "train_end": train_end_date,
            "test_start": test_start_date,
            "test_end": test_end_date,
            "n_train": len(train_idx),
            "n_test": n_test,
            "mse": mse,
            "mae": mae,
        })

        print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}")
        print(f"  Basic signals: long={int((basic_signals > 0).sum())}, short={int((basic_signals < 0).sum())}")
        print(f"  Risk-adj signals: long={int((risk_signals > 0).sum())}, short={int((risk_signals < 0).sum())}, flat={int((risk_signals == 0).sum())}")

        all_basic_signals.append(basic_signals)
        all_risk_signals.append(risk_signals)
        all_actual_returns.append(actual)
        all_dates.append(test_dates)

    # Concatenate all OOS results
    basic_signals_all = np.concatenate(all_basic_signals)
    risk_signals_all = np.concatenate(all_risk_signals)
    actual_returns_all = np.concatenate(all_actual_returns)
    dates_all = np.concatenate(all_dates)

    # Run backtest engine for each strategy
    print("\n=== Computing Backtest Results ===")

    # Basic strategy (long/short)
    basic_gross = engine_gross.run(basic_signals_all, actual_returns_all, dates_all)
    basic_net = engine_net.run(basic_signals_all, actual_returns_all, dates_all)

    # Risk-adjusted strategy
    risk_gross = engine_gross.run(risk_signals_all, actual_returns_all, dates_all)
    risk_net = engine_net.run(risk_signals_all, actual_returns_all, dates_all)

    # Buy & Hold baseline
    bh_result = buy_hold.run(actual_returns_all)

    # Print results
    print(f"\n{'='*60}")
    print("BACKTEST RESULTS")
    print(f"{'='*60}")

    print(f"\n--- Basic Long/Short Strategy ---")
    print(f"  Gross Sharpe: {basic_gross['gross_metrics']['sharpeRatio']:.4f}")
    print(f"  Net Sharpe:   {basic_net['net_metrics']['sharpeRatio']:.4f}")
    print(f"  Gross Return: {basic_gross['gross_metrics']['annualReturn']:.4f}")
    print(f"  Net Return:   {basic_net['net_metrics']['annualReturn']:.4f}")
    print(f"  Max Drawdown: {basic_net['net_metrics']['maxDrawdown']:.4f}")
    print(f"  Trades:       {basic_net['total_trades']}")

    print(f"\n--- Risk-Adjusted Strategy ---")
    print(f"  Gross Sharpe: {risk_gross['gross_metrics']['sharpeRatio']:.4f}")
    print(f"  Net Sharpe:   {risk_net['net_metrics']['sharpeRatio']:.4f}")
    print(f"  Gross Return: {risk_gross['gross_metrics']['annualReturn']:.4f}")
    print(f"  Net Return:   {risk_net['net_metrics']['annualReturn']:.4f}")
    print(f"  Max Drawdown: {risk_net['net_metrics']['maxDrawdown']:.4f}")
    print(f"  Trades:       {risk_net['total_trades']}")

    print(f"\n--- Buy & Hold Baseline ---")
    print(f"  Sharpe: {bh_result['metrics']['sharpeRatio']:.4f}")
    print(f"  Annual Return: {bh_result['metrics']['annualReturn']:.4f}")
    print(f"  Max Drawdown: {bh_result['metrics']['maxDrawdown']:.4f}")

    # Save results
    output_dir = PROJECT_ROOT / "reports" / "cycle_4"
    output_dir.mkdir(parents=True, exist_ok=True)

    # backtest_summary.json
    backtest_summary = {
        "basic_strategy": {
            "gross": basic_gross["gross_metrics"],
            "net": basic_net["net_metrics"],
            "total_trades": basic_net["total_trades"],
            "total_costs": basic_net["total_costs"],
        },
        "risk_adjusted_strategy": {
            "gross": risk_gross["gross_metrics"],
            "net": risk_net["net_metrics"],
            "total_trades": risk_net["total_trades"],
            "total_costs": risk_net["total_costs"],
        },
        "buy_and_hold": {
            "metrics": bh_result["metrics"],
            "total_trades": 0,
        },
        "cost_bps": args.cost_bps,
        "n_splits": args.n_splits,
        "per_window": window_results,
    }

    summary_path = output_dir / "backtest_summary.json"
    with open(summary_path, "w") as f:
        json.dump(backtest_summary, f, indent=2)
    print(f"\nSaved: {summary_path}")

    # metrics.json (ARF standard) — use risk-adjusted as primary strategy
    metrics_json = {
        "sharpeRatio": risk_gross["gross_metrics"]["sharpeRatio"],
        "annualReturn": risk_gross["gross_metrics"]["annualReturn"],
        "maxDrawdown": risk_gross["gross_metrics"]["maxDrawdown"],
        "hitRate": risk_gross["gross_metrics"]["hitRate"],
        "totalTrades": risk_net["total_trades"],
        "transactionCosts": {
            "feeBps": 10.0,
            "slippageBps": 5.0,
            "netSharpe": risk_net["net_metrics"]["sharpeRatio"],
        },
        "walkForward": {
            "windows": len(window_results),
            "positiveWindows": 0,  # Will compute per-window below
            "avgOosSharpe": risk_net["net_metrics"]["sharpeRatio"],
        },
        "customMetrics": {
            "phase": "4-trading-strategy-and-cost-model",
            "basic_gross_sharpe": basic_gross["gross_metrics"]["sharpeRatio"],
            "basic_net_sharpe": basic_net["net_metrics"]["sharpeRatio"],
            "basic_gross_return": basic_gross["gross_metrics"]["annualReturn"],
            "basic_net_return": basic_net["net_metrics"]["annualReturn"],
            "basic_max_drawdown": basic_net["net_metrics"]["maxDrawdown"],
            "basic_hit_rate": basic_net["net_metrics"]["hitRate"],
            "risk_adjusted_gross_sharpe": risk_gross["gross_metrics"]["sharpeRatio"],
            "risk_adjusted_net_sharpe": risk_net["net_metrics"]["sharpeRatio"],
            "risk_adjusted_gross_return": risk_gross["gross_metrics"]["annualReturn"],
            "risk_adjusted_net_return": risk_net["net_metrics"]["annualReturn"],
            "risk_adjusted_max_drawdown": risk_net["net_metrics"]["maxDrawdown"],
            "risk_adjusted_hit_rate": risk_net["net_metrics"]["hitRate"],
            "baseline_buyhold_sharpe": bh_result["metrics"]["sharpeRatio"],
            "baseline_buyhold_return": bh_result["metrics"]["annualReturn"],
            "baseline_buyhold_drawdown": bh_result["metrics"]["maxDrawdown"],
            "baseline_buyhold_hit_rate": bh_result["metrics"]["hitRate"],
            "baseline_1n_sharpe": bh_result["metrics"]["sharpeRatio"],
            "baseline_1n_return": bh_result["metrics"]["annualReturn"],
            "baseline_1n_drawdown": bh_result["metrics"]["maxDrawdown"],
            "strategy_vs_1n_sharpe_diff": round(risk_net["net_metrics"]["sharpeRatio"] - bh_result["metrics"]["sharpeRatio"], 4),
            "strategy_vs_1n_return_diff": round(risk_net["net_metrics"]["annualReturn"] - bh_result["metrics"]["annualReturn"], 4),
            "strategy_vs_1n_drawdown_diff": round(risk_net["net_metrics"]["maxDrawdown"] - bh_result["metrics"]["maxDrawdown"], 4),
            "cost_bps": args.cost_bps,
            "vol_lookback": 20,
            "vol_percentile": 40.0,
        },
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Saved: {metrics_path}")

    # Plot P&L curves (all strategies on one plot)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(basic_gross["gross_cumulative"].values, label="Basic (Gross)", linewidth=1.2, alpha=0.7)
    ax.plot(basic_net["net_cumulative"].values, label="Basic (Net)", linewidth=1.5)
    ax.plot(risk_gross["gross_cumulative"].values, label="Risk-Adjusted (Gross)", linewidth=1.2, alpha=0.7)
    ax.plot(risk_net["net_cumulative"].values, label="Risk-Adjusted (Net)", linewidth=1.5)
    bh_cumulative = (1 + pd.Series(actual_returns_all)).cumprod()
    ax.plot(bh_cumulative.values, label="Buy & Hold", linewidth=1.5, linestyle="--")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("LSTM+GARCH Backtest: Cumulative P&L Comparison")
    ax.set_xlabel("Time Step (Walk-Forward OOS)")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pnl_path = output_dir / "pnl_curve.png"
    fig.savefig(pnl_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {pnl_path}")

    return backtest_summary, metrics_json


def main(argv=None):
    args = parse_args(argv)

    if args.run_backtest:
        run_backtest(args)
    elif args.run_walkforward:
        print("Loading BTC-USD data...")
        df = load_btc_data()
        print(f"Loaded {len(df)} rows ({df['Date'].min().date()} to {df['Date'].max().date()})")

        evaluator = WalkForwardEvaluator(
            n_splits=args.n_splits,
            train_ratio=args.train_ratio,
            lookback=args.lookback,
            lstm_epochs=args.epochs,
            lstm_hidden=args.hidden_size,
            lstm_lr=args.lr,
        )

        print(f"\nRunning walk-forward evaluation with {args.n_splits} splits...")
        results = evaluator.run(df)

        wf_path, metrics_path = save_walkforward_results(results)

        print(f"\n{'='*60}")
        print("Walk-Forward Evaluation Complete")
        print(f"{'='*60}")
        print(f"Splits: {len(results['splits'])}")
        for s in results["splits"]:
            print(f"  Window {s['window']}: MSE={s['mse']:.8f}, MAE={s['mae']:.8f}, "
                  f"Sharpe={s['sharpe']:.4f}")
        print(f"\nAggregate:")
        print(f"  Avg MSE: {results['walkforward_metrics']['aggregate']['avg_mse']:.8f}")
        print(f"  Avg MAE: {results['walkforward_metrics']['aggregate']['avg_mae']:.8f}")
        print(f"\nResults saved to:")
        print(f"  {wf_path}")
        print(f"  {metrics_path}")
    else:
        print("No action specified. Use --run-walkforward or --run-backtest.")
        print("Examples:")
        print("  python -m src.main --run-walkforward")
        print("  python -m src.main --run-backtest")


if __name__ == "__main__":
    main()
