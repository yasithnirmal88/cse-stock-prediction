"""Strict OHLCV validation and cleaning for CSE equity research."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)
PRICE_COLUMNS = ("Open", "High", "Low", "Close")
NUMERIC_COLUMNS = (*PRICE_COLUMNS, "Volume")
MAX_DAILY_JUMP = 0.40
OUTLIER_RETURN_LIMIT = 1.00

class DataValidator:
    """Validate and conservatively clean OHLCV data without hiding anomalies."""
    def __init__(self, config_path: str | Path = "config.yaml") -> None:
        self.project_root = Path(__file__).resolve().parent.parent
        path = Path(config_path)
        if not path.is_absolute():
            path = self.project_root / path
        with path.open("r", encoding="utf-8") as stream:
            self.config: dict[str, Any] = yaml.safe_load(stream) or {}
        self.default_start = pd.Timestamp(self.config.get("universe", {}).get("start_date", "2015-01-01"))
        configured_end = self.config.get("universe", {}).get("end_date")
        self.default_end = pd.Timestamp(configured_end) if configured_end else pd.Timestamp.now().normalize()

    @staticmethod
    def _dated_frame(df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        if "Date" in frame.columns:
            dates = pd.to_datetime(frame.pop("Date"), errors="coerce", utc=True).dt.tz_localize(None)
            frame.index = dates
        else:
            frame.index = pd.to_datetime(frame.index, errors="coerce", utc=True).tz_localize(None)
        frame.index.name = "Date"
        return frame

    def validate_ohlcv(self, df: pd.DataFrame, symbol: str) -> dict[str, Any]:
        """Return integrity metrics, date bounds, row count, and a 0-100 score."""
        frame = self._dated_frame(df)
        duplicates, rows = int(frame.index.duplicated(keep=False).sum()), len(frame)
        numeric = frame.reindex(columns=NUMERIC_COLUMNS).apply(pd.to_numeric, errors="coerce")
        missing = int(numeric.isna().sum().sum())
        negative = int((numeric[list(PRICE_COLUMNS)] < 0).sum().sum())
        zero_volume = int((numeric["Volume"] == 0).sum())
        jumps = int((numeric["Close"].sort_index().pct_change(fill_method=None).abs() > MAX_DAILY_JUMP).sum())
        dates = pd.DatetimeIndex(frame.index.dropna().unique()).sort_values()
        gaps = int(((dates[1:] - dates[:-1]).days - 1).clip(min=0).sum()) if len(dates) > 1 else 0
        normalized = pd.Series(frame.index.dropna()).dt.normalize()
        start = normalized.min() if not normalized.empty else pd.NaT
        end = normalized.max() if not normalized.empty else pd.NaT
        score = 100 - min(40, missing / max(rows * len(NUMERIC_COLUMNS), 1) * 100)
        score -= min(20, duplicates / max(rows, 1) * 100)
        score -= min(20, negative / max(rows * len(PRICE_COLUMNS), 1) * 100)
        score -= min(20, jumps / max(rows, 1) * 100)
        score -= min(10, zero_volume / max(rows, 1) * 10)
        score -= min(10, gaps / max(rows, 1) * 10)
        return {"symbol": symbol, "missing_values_count": missing, "duplicate_dates": duplicates,
            "negative_prices": negative, "zero_volume_days": zero_volume,
            "unrealistic_price_jumps": jumps, "date_gaps": gaps,
            "data_start": start, "data_end": end, "total_rows": rows,
            "quality_score": round(max(0.0, min(100.0, score)), 2)}

    def clean_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Sort, deduplicate, coerce, remove invalid prices, and flag suspect bars."""
        del symbol
        frame = self._dated_frame(df)
        frame = frame.loc[~frame.index.isna()].sort_index()
        frame = frame.loc[~frame.index.duplicated(keep="last")].copy()
        for column in NUMERIC_COLUMNS:
            if column not in frame:
                frame[column] = pd.NA
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        negative = frame[list(PRICE_COLUMNS)].lt(0).any(axis=1)
        frame.loc[negative, list(PRICE_COLUMNS)] = float("nan")
        extreme = frame["Close"].pct_change(fill_method=None).abs() > OUTLIER_RETURN_LIMIT
        invalid = ((frame["High"] < frame["Low"])
            | (frame["High"] < frame[["Open", "Close"]].max(axis=1))
            | (frame["Low"] > frame[["Open", "Close"]].min(axis=1)))
        invalid |= frame[list(PRICE_COLUMNS)].isna().any(axis=1) | negative | extreme
        frame["is_valid_bar"] = ~invalid
        if "Source" not in frame:
            frame["Source"] = "unknown"
        else:
            frame["Source"] = frame["Source"].fillna("unknown")
        return frame

    def check_point_in_time_safety(self, df: pd.DataFrame) -> bool:
        """Return False if dates are invalid or any observation is in the future."""
        try:
            dates = self._dated_frame(df).index
            return bool(not dates.isna().any() and (dates <= pd.Timestamp.now().tz_localize(None)).all())
        except (TypeError, ValueError, OverflowError):
            return False

    def generate_quality_report(self, symbols: list[str]) -> pd.DataFrame:
        """Load configured daily history and return one validation row per symbol."""
        from src.data_loader import CSEDataLoader
        loader = CSEDataLoader()
        start, end = self.default_start.strftime("%Y-%m-%d"), self.default_end.strftime("%Y-%m-%d")
        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            try:
                rows.append(self.validate_ohlcv(loader.get_stock_data(symbol, start, end), symbol))
            except Exception as exc:
                LOGGER.exception("Quality validation failed for %s: %s", symbol, exc)
                rows.append({"symbol": symbol, "missing_values_count": 0, "duplicate_dates": 0,
                    "negative_prices": 0, "zero_volume_days": 0, "unrealistic_price_jumps": 0,
                    "date_gaps": 0, "data_start": pd.NaT, "data_end": pd.NaT,
                    "total_rows": 0, "quality_score": 0.0})
        return pd.DataFrame(rows)
