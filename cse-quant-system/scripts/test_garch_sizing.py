"""Demonstrate GARCH forecasts and volatility-based position sizing.

Uses local OHLCV for JKH.N0000; no network download is triggered.
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.models.garch_model import VolatilityManager


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "JKH.N0000_features.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "garch_sizing")
    parser.add_argument("--capital", type=float, default=1_000_000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.data.exists():
        raise SystemExit(f"Missing {args.data}; generate local features or pass --data with OHLCV history.")
    frame = pd.read_csv(args.data, parse_dates=["Date"], index_col="Date")
    frame.index = pd.DatetimeIndex(frame.index)
    if "Close" not in frame:
        raise SystemExit("Input data must contain a Close column")
    manager = VolatilityManager()
    returns = manager.calculate_returns(frame)
    result = manager.fit_garch(returns)
    forecast = manager.forecast_volatility()
    print(f"Observations: {len(returns)}")
    print(f"Latest fitted conditional annual volatility: {manager.historical_volatility_.iloc[-1]:.2%}")
    print(f"Next-session GARCH annualized volatility forecast: {forecast:.2%}")
    print(f"Model convergence flag: {result.convergence_flag}")

    # Compare a range of volatility quantiles at the same price/capital to isolate sizing impact.
    history = manager.historical_volatility_.dropna()
    low_vol, high_vol = history.quantile([0.2, 0.8]).astype(float)
    price = float(pd.to_numeric(frame["Close"], errors="coerce").dropna().iloc[-1])
    low_size = manager.calculate_position_size(args.capital, price, low_vol)
    high_size = manager.calculate_position_size(args.capital, price, high_vol)
    print("Low-volatility sizing:", low_size)
    print("High-volatility sizing:", high_size)

    # Plot history and the out-of-sample next-period forecast, keeping units annualized.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    manager.historical_volatility_.plot(ax=ax, label="GARCH fitted conditional annual volatility")
    ax.scatter([manager.historical_volatility_.index[-1]], [forecast], color="red", marker="x",
               s=90, label="Next-session forecast")
    ax.set(title="JKH.N0000 GARCH volatility and next-session forecast", ylabel="Annualized volatility")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "jkh_garch_volatility.png", dpi=150)
    plt.close(fig)

    # Visualize position values at low/high volatility without implying historical backtest performance.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["Low volatility", "High volatility"], [low_size["position_value"], high_size["position_value"]],
           color=["seagreen", "indianred"])
    ax.set(title="Risk-budgeted position value (illustrative)", ylabel="Position value")
    fig.tight_layout()
    fig.savefig(args.output_dir / "jkh_position_size_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Plots saved under {args.output_dir}")


if __name__ == "__main__":
    main()
