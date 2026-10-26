"""Screen the configured local CSE universe for cointegrated pairs.

WARNING: full-history pair selection and same-period performance are in-sample,
look-ahead biased. Use walk-forward pair selection for any credible backtest.
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.pairs import PairsTradingEngine


def load_universe(data_dir: Path, universe_path: Path) -> dict[str, pd.DataFrame]:
    """Load local processed CSV/Parquet/Feather OHLCV files by universe symbol."""
    with universe_path.open(encoding="utf-8") as stream:
        stocks = (yaml.safe_load(stream) or {}).get("stocks", [])
    result = {}
    for item in stocks:
        symbol = str(item.get("symbol", ""))
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", symbol)
        candidates = [data_dir / f"{safe}_features.csv", data_dir / f"{safe}_features.parquet",
                      data_dir / f"{safe}_features.feather", data_dir / f"{safe}.csv"]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            continue
        if path.suffix == ".csv":
            frame = pd.read_csv(path)
        elif path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_feather(path)
        date = next((c for c in ("Date", "date", "Datetime", "datetime") if c in frame), None)
        if date:
            frame[date] = pd.to_datetime(frame[date], errors="coerce")
            frame = frame.dropna(subset=[date]).set_index(date)
        elif not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError(f"{path} needs a Date column")
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        result[symbol] = frame
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--universe", type=Path, default=ROOT / "data" / "universe.yaml")
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--significance", type=float, default=0.05)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--min-median-dollar-volume", type=float, default=0.0,
                        help="Positive threshold enables Volume-backed recent liquidity screen")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data = load_universe(args.data_dir, args.universe)
    print(f"Loaded {len(data)} of the configured universe; lookback: all aligned history, minimum 120 observations.")
    if not data:
        print(f"No local price files found in {args.data_dir}; create processed data or pass --data-dir.")
        return
    engine = PairsTradingEngine(min_median_dollar_volume=args.min_median_dollar_volume)
    pairs = engine.find_cointegrated_pairs(data, significance=args.significance)
    if not pairs:
        print("No cointegrated pairs found. Consider more history or inspect the liquidity filter.")
        return
    print("WARNING: pair ranking uses the full supplied sample (selection/look-ahead bias); use walk-forward selection.")
    print("Top pairs (Engle-Granger p-value; hedge ratio is log(price_y) ~ beta*log(price_x)):")
    for item in pairs[:max(1, min(args.top, 10))]:
        print(f"{item['symbol_y']} / {item['symbol_x']}  p={item['p_value']:.5g}  beta={item['hedge_ratio']:.4f}  n={item['observations']}")
    best = pairs[0]
    spread = engine.calculate_spread(best["pair"], data)
    signals = engine.generate_pairs_signals(spread)
    # Explicit one-period lag: today's signal is applied to the following price change.
    changes = spread.price_y.pct_change() - best["hedge_ratio"] * spread.price_x.pct_change()
    strategy = signals.signal.shift(1).fillna(0) * changes
    valid = strategy.dropna()
    cumulative = (1 + valid).prod() - 1 if len(valid) else 0.0
    drawdown = ((1 + valid).cumprod() / (1 + valid).cumprod().cummax() - 1).min() if len(valid) else 0.0
    print(f"Best pair: {best['pair']}; latest z-score={signals.z_score.iloc[-1]:.3f}; latest signal={int(signals.signal.iloc[-1])}")
    print(f"In-sample, one-bar-lag gross proxy: observations={len(valid)}, cumulative={cumulative:.2%}, max drawdown={drawdown:.2%}")
    print("Performance excludes fees, financing, borrow constraints and execution slippage; not an out-of-sample result.")
    print("Portfolio allocations:", engine.construct_portfolio(pairs[:max(1, min(args.top, 10))], args.capital))

if __name__ == "__main__":
    main()
