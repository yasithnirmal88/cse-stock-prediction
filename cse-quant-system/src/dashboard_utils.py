"""Safe readers for dashboard artifacts produced by research scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _find(folder: Path, names: tuple[str, ...]) -> Path | None:
    return next((folder / name for name in names if (folder / name).is_file()), None)


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def load_metrics(reports_dir: str | Path = REPORTS) -> dict[str, Any]:
    folder = Path(reports_dir)
    path = _find(folder, ("latest/metrics.json", "latest/walk_forward_metrics.json", "walk_forward_metrics.json", "smoke_test/walk_forward_metrics.json"))
    if path:
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(result, dict):
                return result
        except (OSError, json.JSONDecodeError):
            pass
    frame = _read_csv(_find(folder, ("latest/metrics.csv", "walk_forward_metrics.csv", "smoke_test/walk_forward_metrics.csv")))
    return frame.iloc[0].where(pd.notna(frame.iloc[0]), None).to_dict() if not frame.empty else {}


def _load_frame(reports_dir: str | Path, latest_names: tuple[str, ...], fallback_names: tuple[str, ...]) -> pd.DataFrame:
    folder = Path(reports_dir)
    frame = _read_csv(_find(folder, latest_names) or _find(folder, fallback_names))
    return frame


def load_equity_curve(reports_dir: str | Path = REPORTS) -> pd.DataFrame:
    frame = _load_frame(reports_dir, ("latest/equity_curve.csv",), ("walk_forward_equity.csv", "equity_curve.csv", "smoke_test/walk_forward_equity.csv"))
    date_col = next((c for c in ("date", "Date", "Datetime", "datetime") if c in frame), None)
    if date_col:
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        frame = frame.dropna(subset=[date_col]).set_index(date_col).sort_index()
    return frame


def load_fold_metrics(reports_dir: str | Path = REPORTS) -> pd.DataFrame:
    return _load_frame(reports_dir, ("latest/fold_metrics.csv",), ("walk_forward_folds.csv", "fold_metrics.csv", "smoke_test/walk_forward_folds.csv"))


def load_signals(reports_dir: str | Path = REPORTS) -> pd.DataFrame:
    frame = _load_frame(reports_dir, ("latest/signals.csv",), ("latest_signals.csv", "signals.csv"))
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        latest = frame["date"].max()
        if pd.notna(latest):
            frame = frame.loc[frame["date"] == latest]
    return frame


def load_predictions(reports_dir: str | Path = REPORTS) -> pd.DataFrame:
    return _load_frame(reports_dir, ("latest/predictions.csv",), ("walk_forward_predictions.csv", "predictions.csv", "smoke_test/walk_forward_predictions.csv"))


def load_run_metadata(reports_dir: str | Path = REPORTS) -> dict[str, Any]:
    path = _find(Path(reports_dir), ("latest/run_metadata.json",))
    if path:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def model_status(reports_dir: str | Path = REPORTS) -> dict[str, Any]:
    folder = Path(reports_dir)
    metrics_path = _find(folder, ("latest/metrics.json", "latest/walk_forward_metrics.json", "walk_forward_metrics.json", "smoke_test/walk_forward_metrics.json"))
    metadata = load_run_metadata(folder)
    pred = load_predictions(folder)
    dates = pd.to_datetime(pred["date"], errors="coerce").dropna() if "date" in pred else pd.Series(dtype="datetime64[ns]")
    return {"metrics_available": metrics_path is not None, "metrics_path": str(metrics_path) if metrics_path else None,
            "last_run": metadata.get("generated_at") or (metrics_path.stat().st_mtime if metrics_path else None),
            "signals_available": not load_signals(folder).empty,
            "fold_count": metadata.get("fold_count", len(load_fold_metrics(folder))),
            "data_start": metadata.get("data_start") or (str(dates.min()) if len(dates) else None),
            "data_end": metadata.get("data_end") or (str(dates.max()) if len(dates) else None),
            "prediction_count": metadata.get("prediction_count", len(pred))}
