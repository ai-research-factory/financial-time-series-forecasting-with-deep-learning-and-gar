"""
Main entry point for the LSTM+GARCH financial time-series forecasting project.

Usage:
    python -m src.main --run-walkforward
    python -m src.main --run-walkforward --n-splits 5 --epochs 50
"""
import argparse
import sys

from src.data.data_loader import load_btc_data
from src.evaluation.validator import WalkForwardEvaluator, save_walkforward_results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="LSTM+GARCH Financial Time-Series Forecasting"
    )
    parser.add_argument(
        "--run-walkforward",
        action="store_true",
        help="Run walk-forward evaluation of the LSTM+GARCH model.",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Number of walk-forward splits (default: 5)")
    parser.add_argument("--lookback", type=int, default=60, help="LSTM lookback window (default: 60)")
    parser.add_argument("--epochs", type=int, default=50, help="LSTM training epochs (default: 50)")
    parser.add_argument("--hidden-size", type=int, default=50, help="LSTM hidden size (default: 50)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--train-ratio", type=float, default=1.0, help="Train ratio for expanding window (default: 1.0)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not args.run_walkforward:
        print("No action specified. Use --run-walkforward to run evaluation.")
        print("Example: python -m src.main --run-walkforward")
        return

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


if __name__ == "__main__":
    main()
