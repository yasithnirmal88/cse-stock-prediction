"""Fast functional smoke test for the Phase 3 walk-forward pipeline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtester import PerformanceAnalyzer, WalkForwardBacktester
from src.features import FeatureEngineer

REPORT_DIR = ROOT / "reports"
TEST_REPORT_DIR = REPORT_DIR / "smoke_test"
SYMBOLS = ["JKH.N0000", "COMB.N0000", "SAMP.N0000"]


def load_config() -> dict:
    with (ROOT / "config.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a mapping")
    return config


def read_cached(config: dict) -> dict[str, pd.DataFrame]:
    data_dir = Path(config.get("data", {}).get("processed_dir", "data/processed"))
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    cached: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        safe = symbol.replace("/", "_")
        candidates = list(data_dir.rglob(f"{safe}_1d.csv")) + list(data_dir.rglob(f"{safe}*.csv"))
        for path in candidates:
            try:
                frame = pd.read_csv(path)
                if frame.empty:
                    continue
                dated = FeatureEngineer._dated(frame)
                if {"Open", "High", "Low", "Close", "Volume"} <= set(dated.columns):
                    cached[symbol] = dated
                    break
            except (OSError, ValueError, pd.errors.ParserError):
                continue
    # First inspect conventional processed CSV caches; then use the loader's
    # validated cache API (no network requests are made by this smoke test).
    for symbol in SYMBOLS:
        if symbol in cached:
            continue
        try:
            from src.data_loader import CSEDataLoader
            frame = CSEDataLoader(ROOT / "config.yaml").load_from_cache(symbol, "1d")
            if frame is not None and not frame.empty:
                dated = FeatureEngineer._dated(frame)
                if {"Open", "High", "Low", "Close", "Volume"} <= set(dated.columns):
                    cached[symbol] = dated
        except (ImportError, OSError, ValueError, KeyError) as exc:
            logging.getLogger(__name__).info("Validated cache lookup unavailable for %s: %s", symbol, exc)
    return cached


def synthetic_ohlcv(symbols: list[str], rows: int = 260) -> dict[str, pd.DataFrame]:
    """Deterministic daily OHLCV suitable for model, signal and GARCH smoke testing."""
    rng = np.random.default_rng(20261026)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=rows)
    result = {}
    for index, symbol in enumerate(symbols):
        drift = 0.00025 + index * 0.00005
        shocks = rng.normal(drift, 0.012 + index * 0.001, rows)
        close = (35.0 + 12.0 * index) * np.exp(np.cumsum(shocks))
        open_ = close * (1.0 + rng.normal(0, 0.002, rows))
        high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.012, rows))
        low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.012, rows))
        volume = rng.integers(100_000, 2_000_000, rows)
        result[symbol] = pd.DataFrame({"Open": open_, "High": high, "Low": low,
                                      "Close": close, "Volume": volume}, index=dates)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    print(f"Config loaded: {ROOT / 'config.yaml'}")
    data = read_cached(config)
    source = "local cache"
    if len(data) < 3:
        print(f"Cached OHLCV found for {len(data)} selected symbols; using deterministic synthetic data.")
        data = synthetic_ohlcv(SYMBOLS)
        source = "deterministic synthetic OHLCV"
    else:
        # Restrict run to the first three configured liquid-universe names and recent data.
        data = {s: data[s] for s in SYMBOLS if s in data}
        end = min(f.index.max() for f in data.values())
        start = end - pd.Timedelta(days=600)
        data = {s: f.loc[start:end].copy() for s, f in data.items()}
        if min(map(len, data.values())) < 150:
            print("Cached sample too short for folds; switching to deterministic synthetic OHLCV.")
            data = synthetic_ohlcv(SYMBOLS)
            source = "deterministic synthetic OHLCV"
    print(f"Data source: {source}; symbols: {', '.join(data)}; rows: {min(map(len, data.values()))}–{max(map(len, data.values()))}")

    engineer = FeatureEngineer(config)
    featured = {symbol: engineer.create_features(frame, symbol=symbol) for symbol, frame in data.items()}
    smoke_config = dict(config)
    smoke_config["walk_forward"] = {**config.get("walk_forward", {}), "window_type": "expanding",
        "min_train_days": 120, "test_days": 21, "step_days": 21, "rolling_train_days": 120,
        "random_seed": 42, "initial_capital": float(config.get("backtest", {}).get("initial_capital", 1_000_000)),
        "model_type": "xgboost", "probability_threshold": 0.5, "max_position_pct": 0.10}
    analyzer = PerformanceAnalyzer(periods_per_year=252)
    results = WalkForwardBacktester(smoke_config, analyzer).run(featured)
    paths = analyzer.generate_reports(results, TEST_REPORT_DIR)

    metrics = results["metrics"]
    print("Key metrics:")
    print(f"  Total return: {metrics['total_return']:.6f}")
    print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.6f}")
    print(f"  Maximum drawdown: {metrics['max_drawdown']:.6f}")
    print(f"  Number of folds: {len(results['fold_metrics'])}")
    print(f"  Number of trades: {metrics['number_of_trades']}")
    print(f"  Predictions: {len(results['all_predictions'])}")
    for name in ("html", "pdf"):
        path = paths[name]
        print(f"{name.upper()} report: {path} (exists={path.is_file()}, bytes={path.stat().st_size if path.exists() else 0})")
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected non-empty {name.upper()} report at {path}")
    if len(results["fold_metrics"]) == 0:
        raise RuntimeError("Smoke test did not produce any complete walk-forward folds")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
