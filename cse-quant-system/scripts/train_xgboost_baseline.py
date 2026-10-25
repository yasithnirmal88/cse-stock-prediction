"""Train a baseline direction model with chronological out-of-sample predictions.

Run from the project root: python scripts/train_xgboost_baseline.py
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.models.xgboost_model import DirectionProbabilityModel

LOGGER = logging.getLogger("train_xgboost_baseline")


def load_feature_file(path: Path) -> pd.DataFrame:
    """Read CSV (including the project's fallback) or parquet; restore Date index."""
    frame = (pd.read_parquet(path) if path.suffix.lower() == ".parquet" else
             pd.read_feather(path) if path.suffix.lower() == ".feather" else pd.read_csv(path))
    if "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date"]).set_index("Date")
    elif isinstance(frame.index, pd.RangeIndex):
        raise ValueError(f"No Date column in {path}")
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame.loc[~frame.index.isna()].sort_index()
    return frame.loc[~frame.index.duplicated(keep="last")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "xgboost_baseline")
    parser.add_argument("--train-window", type=int, default=500, help="Rows; 0 uses expanding history")
    parser.add_argument("--test-window", type=int, default=60)
    parser.add_argument("--step-size", type=int, default=60)
    parser.add_argument("--target", default="target_direction")
    parser.add_argument("--model-type", choices=["xgboost", "lightgbm"], default="xgboost")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    files = sorted(p for p in args.data_dir.glob("*_features.*") if p.suffix.lower() in (".csv", ".parquet", ".feather"))
    if not files:
        LOGGER.warning("No processed feature files found under %s; run universe feature generation first", args.data_dir)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Target/price columns are deliberately excluded: only precomputed predictors are eligible.
    targets = {"target_direction", "target_direction_aggressive", "target_return"}
    excluded = targets | {"Open", "High", "Low", "Close", "Volume", "Date"}
    def feature_columns(frame: pd.DataFrame) -> list[str]:
        return [c for c in frame.select_dtypes(include="number").columns if c not in excluded]

    # Exercise a single symbol first, then perform independent per-symbol walk-forward runs.
    LOGGER.info("Single-symbol smoke run: %s", files[0].name)
    first = load_feature_file(files[0])
    features = feature_columns(first)
    if args.target not in first or not features:
        LOGGER.warning("Skipping %s: missing target or predictors", files[0].name)
        return
    model = DirectionProbabilityModel(model_type=args.model_type)
    one = model.walk_forward_train_predict(first, features, args.train_window, args.test_window, args.step_size)
    if one.empty:
        LOGGER.warning("No valid OOS predictions for %s; need more history and both training classes", files[0].name)
        return
    one.insert(0, "symbol", files[0].name.removesuffix("_features" + files[0].suffix))
    one.to_csv(args.output_dir / "single_symbol_oos.csv", index_label="Date")

    outputs = [one]
    for path in files[1:]:
        try:
            frame = load_feature_file(path)
            cols = feature_columns(frame)
            if args.target not in frame or not cols:
                LOGGER.warning("Skipping %s: missing target or predictors", path.name)
                continue
            model = DirectionProbabilityModel(model_type=args.model_type)
            pred = model.walk_forward_train_predict(frame, cols, args.train_window, args.test_window, args.step_size)
            if pred.empty:
                LOGGER.warning("No valid OOS predictions for %s", path.name)
                continue
            pred.insert(0, "symbol", path.name.removesuffix("_features" + path.suffix))
            outputs.append(pred)
        except (ValueError, OSError, KeyError) as exc:
            LOGGER.exception("Skipping %s: %s", path.name, exc)
    combined = pd.concat(outputs).sort_index()
    combined.to_csv(args.output_dir / "universe_oos_predictions.csv", index_label="Date")
    print(classification_report(combined["y_true"], combined["signal"], labels=[0, 1],
                                target_names=["Cash/Down", "Buy/Up"], zero_division=0))
    LOGGER.info("Saved %d strictly OOS rows under %s", len(combined), args.output_dir)


if __name__ == "__main__":
    main()
