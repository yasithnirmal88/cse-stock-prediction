"""Run the Phase 1 momentum baseline against the CSE benchmark series."""
from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.backtester.simple_backtester import SimpleBacktester
from src.data_loader import CSEDataLoader
from src.models.momentum import SimpleMomentumStrategy


def load_universe(path: Path = ROOT / "data" / "universe.yaml") -> list[str]:
    """Read stock symbols from the project's universe YAML."""
    with path.open("r", encoding="utf-8") as stream:
        document: dict[str, Any] = yaml.safe_load(stream) or {}
    return [str(item["symbol"]) for item in document.get("stocks", []) if item.get("symbol")]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    universe = load_universe()
    if not universe:
        raise RuntimeError("data/universe.yaml contains no stock symbols")
    start = str(config.get("universe", {}).get("start_date", "2015-01-01"))
    end = str(config.get("universe", {}).get("end_date") or pd.Timestamp.now().strftime("%Y-%m-%d"))
    loader = CSEDataLoader(ROOT / "config.yaml")
    prices = loader.get_multiple_stocks(universe, start, end)
    prices = {symbol: frame for symbol, frame in prices.items() if not frame.empty and "Close" in frame}
    if not prices:
        raise RuntimeError("No stock price histories could be loaded; check the cache or data providers")
    benchmark = loader.get_benchmark_data(start, end)
    if benchmark.empty:
        raise RuntimeError("Neither S&P SL20 nor the labelled ASPI fallback could be loaded")
    source = str(benchmark["Source"].dropna().iloc[0]) if "Source" in benchmark and benchmark["Source"].notna().any() else "Benchmark"
    benchmark_label = "ASPI proxy" if "ASPI proxy" in source else "S&P SL20"
    strategy = SimpleMomentumStrategy(lookback_period=20, moving_average_type="sma")
    signaled = {symbol: strategy.generate_signals(frame) for symbol, frame in prices.items()}
    costs = config.get("costs", {})
    backtest = SimpleBacktester(
        initial_capital=float(config.get("backtest", {}).get("initial_capital", 1_000_000)),
        entry_cost=float(costs.get("commission_rate_entry", 0.0056)),
        exit_cost=float(costs.get("commission_rate_exit", 0.0056)),
    )
    result = backtest.run_backtest(prices, signaled, benchmark, benchmark_label)
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    metrics_path = reports / "phase1_metrics.csv"
    result["metrics"].to_csv(metrics_path)
    figure, axis = plt.subplots(figsize=(12, 6))
    result["equity_curves"].plot(ax=axis, linewidth=1.6)
    axis.set_title(f"Phase 1: Momentum Baseline vs {benchmark_label}")
    axis.set_xlabel("Date")
    axis.set_ylabel("Portfolio value (LKR)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    plot_path = reports / "phase1_momentum_baseline.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    print(f"Universe symbols: {len(universe)}; usable: {len(prices)}")
    print(f"Benchmark source: {source}")
    print(f"Transaction costs: {backtest.entry_cost:.2%} entry + {backtest.exit_cost:.2%} exit")
    print("\nPerformance metrics (returns/drawdown as fractions):")
    print(result["metrics"].to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nEquity-curve chart: {plot_path}")
    print(f"Metrics CSV: {metrics_path}")
    print("\nFinal equity values (LKR):")
    print(result["equity_curves"].iloc[-1].to_string(float_format=lambda value: f"{value:,.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
