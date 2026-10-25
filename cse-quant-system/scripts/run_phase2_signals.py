"""Generate final Phase 2 signals from local feature data and a fitted model."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import sys
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.signals import SignalEngine
from src.models.xgboost_model import DirectionProbabilityModel


def load_price_data(data_dir: Path, universe_path: Path) -> dict[str, pd.DataFrame]:
    """Load configured local *_features files; do not fetch or backfill live data."""
    with universe_path.open(encoding="utf-8") as stream:
        stocks = (yaml.safe_load(stream) or {}).get("stocks", [])
    data = {}
    for item in stocks:
        symbol = str(item.get("symbol", ""))
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", symbol)
        paths = [data_dir / f"{safe}_features.csv", data_dir / f"{safe}_features.parquet",
                 data_dir / f"{safe}_features.feather"]
        path = next((p for p in paths if p.is_file()), None)
        if path is None: continue
        frame = (pd.read_csv(path) if path.suffix == ".csv" else
                 pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_feather(path))
        date_col = next((c for c in ("Date", "date", "Datetime") if c in frame), None)
        if date_col is None: raise ValueError(f"{path} must contain a Date column")
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        frame = frame.dropna(subset=[date_col]).set_index(date_col).sort_index()
        data[symbol] = frame.loc[~frame.index.duplicated(keep="last")]
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--universe", type=Path, default=ROOT / "data" / "universe.yaml")
    parser.add_argument("--model", type=Path, required=True,
                        help="Previously trained DirectionProbabilityModel joblib, fitted on prior data")
    parser.add_argument("--date", default=None, help="Signal date YYYY-MM-DD; defaults to latest available")
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--probability-threshold", type=float, default=0.55)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "latest_signals.csv")
    args = parser.parse_args()
    data = load_price_data(args.data_dir, args.universe)
    if not data:
        raise SystemExit(f"No local universe feature data found under {args.data_dir}.")
    if not args.model.is_file():
        raise SystemExit(f"Trained model not found: {args.model}. Train/export a prior-period model first.")
    model = DirectionProbabilityModel.load(args.model)
    engine = SignalEngine(data, model=model, capital=args.capital,
                          probability_threshold=args.probability_threshold, signal_path=args.output)
    signals = engine.generate_combined_signals(args.date)
    if signals.empty:
        print("No signals generated.")
        return
    print(signals.to_string(index=False, formatters={"probability": lambda x: "-" if pd.isna(x) else f"{x:.3f}",
                                                     "confidence_score": "{:.3f}".format}))
    path = engine.save_signals_to_csv()
    # Count directional holdings without netting the independent pairs sleeve.
    signed = signals.assign(signed_exposure=signals.signal * signals.suggested_position_size)
    gross = float(signals.suggested_position_size.sum())
    long = float(signals.loc[signals.signal > 0, "suggested_position_size"].sum())
    short = float(signals.loc[signals.signal < 0, "suggested_position_size"].sum())
    print(f"Saved: {path}")
    print(f"Exposure (gross allocated): {gross:,.2f} / {args.capital:,.2f} capital")
    print(f"Long: {long:,.2f}; short: {short:,.2f}; net signed: {float(signed.signed_exposure.sum()):,.2f}")
    print("Momentum+ML positions are long-only; pairs legs are separate and intended dollar-neutral.")

if __name__ == "__main__":
    main()
