"""Run walk-forward analysis on real Colombo Stock Exchange (S&P SL20) historical data.

- Loads cleaned daily OHLCV series from data/processed/.
- Generates leak-free technical features with FeatureEngineer.
- Evaluates out-of-sample performance fold-by-fold using WalkForwardBacktester:
    * DirectionProbabilityModel (XGBoost) refitted on each fold's training history
    * SignalEngine for causal momentum gating and probability thresholding
    * GARCH(1,1) dynamic volatility regime classification and position sizing
    * Exact round-trip transaction costs (1.12%) applied at every position turnover
- Generates comprehensive HTML & PDF backtest reports.
- Updates reports/latest/ artifacts for the Streamlit dashboard.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtester import PerformanceAnalyzer, WalkForwardBacktester
from src.data_loader import load_config
from src.features import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("real_walk_forward")

# Selected core liquid S&P SL20 constituents
DEFAULT_SYMBOLS = [
    "JKH.N0000",
    "COMB.N0000",
    "SAMP.N0000",
    "HNB.N0000",
    "NDB.N0000",
    "DIAL.N0000",
    "HAYL.N0000",
    "LOLC.N0000",
]


def load_real_market_data(
    processed_dir: Path,
    symbols: list[str],
    start_date: str | None = None,
    min_history_rows: int = 250,
) -> dict[str, pd.DataFrame]:
    """Load clean real OHLCV data from data/processed/."""
    data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        p = processed_dir / f"{symbol}.csv"
        if not p.is_file():
            logger.warning(f"File not found for symbol {symbol}: {p}")
            continue

        df = pd.read_csv(p)
        if df.empty:
            continue

        dated = FeatureEngineer._dated(df)
        if start_date:
            dated = dated.loc[start_date:]

        if len(dated) < min_history_rows:
            logger.warning(f"Skipping {symbol}: insufficient history ({len(dated)} rows < {min_history_rows})")
            continue

        data[symbol] = dated
        logger.info(
            f"Loaded {symbol}: {len(dated)} rows from {dated.index.min().date()} to {dated.index.max().date()}"
        )

    if not data:
        raise FileNotFoundError(f"No usable stock CSV files found in {processed_dir} for symbols: {symbols}")

    return data


def run_real_walk_forward(
    symbols: list[str] | None = None,
    start_date: str | None = "2019-01-01",
    min_train_days: int = 504,
    test_days: int = 63,
    step_days: int = 63,
    window_type: str = "expanding",
    output_dir: Path = ROOT / "reports" / "real_walk_forward",
    update_latest: bool = True,
) -> dict[str, Any]:
    """Run full walk-forward backtest on real CSE market data."""
    symbols = symbols or DEFAULT_SYMBOLS
    config_path = ROOT / "config.yaml"
    config = load_config(config_path)

    # Configure walk-forward parameters
    wf_config = dict(config.get("walk_forward", {}))
    wf_config.update({
        "window_type": window_type,
        "min_train_days": min_train_days,
        "test_days": test_days,
        "step_days": step_days,
        "random_seed": 42,
        "initial_capital": float(config.get("backtest", {}).get("initial_capital", 1_000_000)),
        "round_trip_cost": float(config.get("costs", {}).get("round_trip", 0.0112)),
        "model_type": "xgboost",
        "probability_threshold": 0.52,
        "max_position_pct": 0.15,
    })
    run_config = dict(config)
    run_config["walk_forward"] = wf_config

    logger.info(f"Loading real CSE historical data (start_date={start_date})...")
    processed_dir = ROOT / "data" / "processed"
    data = load_real_market_data(
        processed_dir,
        symbols,
        start_date=start_date,
        min_history_rows=min_train_days + test_days,
    )

    logger.info(f"Engineering causal features across {len(data)} liquid equities...")
    fe = FeatureEngineer(run_config)
    featured_frames: dict[str, pd.DataFrame] = {}
    for symbol, raw_df in data.items():
        feat = fe.create_features(raw_df, symbol=symbol)
        featured_frames[symbol] = feat

    logger.info(
        f"Starting walk-forward backtest: window={window_type}, min_train={min_train_days}d (~{min_train_days//252}y), "
        f"test={test_days}d, step={step_days}d, round_trip_cost={wf_config['round_trip_cost']*100:.2f}%"
    )

    analyzer = PerformanceAnalyzer(periods_per_year=int(wf_config.get("periods_per_year", 252)))
    backtester = WalkForwardBacktester(run_config, analyzer)
    results = backtester.run(featured_frames)

    # Generate full reports (HTML + PDF)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_paths = analyzer.generate_reports(results, output_dir)
    logger.info(f"Reports written to: {output_dir}")

    # Synchronize reports/latest/ for dashboard consumption
    if update_latest:
        latest_dir = ROOT / "reports" / "latest"
        latest_dir.mkdir(parents=True, exist_ok=True)

        # 1. OHLCV series across evaluated stocks
        ohlcv_rows = []
        for symbol, source_frame in data.items():
            raw = source_frame.reset_index()
            raw.insert(0, "symbol", symbol)
            ohlcv_rows.append(
                raw.rename(columns={"Date": "date", **{c: c.lower() for c in ("Open", "High", "Low", "Close", "Volume")}})
            )
        pd.concat(ohlcv_rows, ignore_index=True).to_csv(latest_dir / "ohlcv.csv", index=False)

        # 2. Run metadata
        metadata_path = latest_dir / "run_metadata.json"
        metadata = {
            "data_source": "real_cse_ohlcv",
            "dataset": "SP20_CSE_Dataset_2001_2025_Adjusted",
            "symbol_count": len(data),
            "symbols": list(data.keys()),
            "window_type": window_type,
            "min_train_days": min_train_days,
            "test_days": test_days,
            "step_days": step_days,
            "transaction_cost_pct": wf_config["round_trip_cost"] * 100,
            "date_range": {
                "start": min(df.index.min().strftime("%Y-%m-%d") for df in data.values()),
                "end": max(df.index.max().strftime("%Y-%m-%d") for df in data.values()),
            },
            "folds": len(results.get("fold_metrics", [])),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info(f"Updated reports/latest/ metadata and OHLCV feed.")

    metrics = results["metrics"]
    logger.info("=" * 60)
    logger.info("REAL DATA WALK-FORWARD BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Folds Evaluated:     {len(results['fold_metrics'])}")
    logger.info(f"  Total Trades:        {metrics.get('number_of_trades', 0)}")
    logger.info(f"  Accuracy:            {metrics.get('accuracy', 0):.4f}")
    logger.info(f"  Precision:           {metrics.get('precision', 0):.4f}")
    logger.info(f"  Total Return:        {metrics.get('total_return', 0) * 100:.2f}%")
    logger.info(f"  CAGR:                {metrics.get('cagr', 0) * 100:.2f}%")
    logger.info(f"  Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.4f}")
    logger.info(f"  Sortino Ratio:       {metrics.get('sortino_ratio', 0):.4f}")
    logger.info(f"  Max Drawdown:        {metrics.get('max_drawdown', 0) * 100:.2f}%")
    logger.info(f"  Win Rate:            {metrics.get('win_rate', 0) * 100:.2f}%")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run real-data walk-forward backtest on CSE stocks.")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="List of stock symbols to evaluate")
    parser.add_argument("--start-date", type=str, default="2019-01-01", help="Start date for analysis window")
    parser.add_argument("--min-train", type=int, default=504, help="Minimum training days (e.g. 504 = ~2 years)")
    parser.add_argument("--test-days", type=int, default=63, help="Out-of-sample evaluation window in days")
    parser.add_argument("--step-days", type=int, default=63, help="Step forward stride in days")
    parser.add_argument("--window", type=str, default="expanding", choices=["expanding", "rolling"])
    args = parser.parse_args()

    run_real_walk_forward(
        symbols=args.symbols,
        start_date=args.start_date,
        min_train_days=args.min_train,
        test_days=args.test_days,
        step_days=args.step_days,
        window_type=args.window,
    )
