"""Causal technical features and universe-level feature persistence."""
from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Any, TypeVar
import numpy as np
import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)
FrameT = TypeVar("FrameT", bound=pd.DataFrame)
FEATURE_GROUPS = ("trend_momentum", "mean_reversion", "volatility", "volume", "lags")
OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
TARGET_COLUMNS = ("target_direction", "target_return", "target_direction_aggressive")


def validate_price_frame(frame: FrameT) -> FrameT:
    """Validate a non-empty table indexed by dates."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Price data must use a pandas DatetimeIndex.")
    if frame.empty or len(frame.columns) == 0:
        raise ValueError("Price data must contain at least one row and one column.")
    return frame


class FeatureEngineer:
    """Create causal indicators; only labels intentionally use next-session data."""
    def __init__(self, config: dict[str, Any] | None = None, enabled_groups: list[str] | None = None) -> None:
        """Accept project config and optionally choose enabled feature groups."""
        self.config = config or {}
        section = self.config.get("features", {})
        configured = section.get("enabled_groups") if isinstance(section, dict) else None
        groups = enabled_groups if enabled_groups is not None else configured
        self.enabled_groups = tuple(FEATURE_GROUPS if groups is None else groups)
        unknown = set(self.enabled_groups) - set(FEATURE_GROUPS)
        if unknown:
            raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
        self._feature_names: list[str] = []

    @staticmethod
    def _dated(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        if "Date" in result.columns:
            result.index = pd.to_datetime(result.pop("Date"), errors="coerce", utc=True).dt.tz_localize(None)
        else:
            result.index = pd.to_datetime(result.index, errors="coerce", utc=True).tz_localize(None)
        result.index.name = "Date"
        result = result.loc[~result.index.isna()].sort_index()
        return result.loc[~result.index.duplicated(keep="last")]

    @staticmethod
    def _zscore(series: pd.Series, window: int) -> pd.Series:
        rolling = series.rolling(window, min_periods=window)
        return (series - rolling.mean()) / rolling.std(ddof=0).replace(0, np.nan)

    def create_features(self, df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
        """Return OHLCV, enabled causal features, and supervised forward labels."""
        del symbol
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("df must be a non-empty DataFrame")
        result = self._dated(df)
        missing = set(OHLCV_COLUMNS) - set(result.columns)
        if missing: raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
        for col in OHLCV_COLUMNS: result[col] = pd.to_numeric(result[col], errors="coerce")
        o, h, l, c, v = (result[col] for col in OHLCV_COLUMNS)
        ret = c.pct_change(fill_method=None)
        names: list[str] = []
        def add(name: str, value: Any) -> None:
            result[name] = value
            names.append(name)

        if "trend_momentum" in self.enabled_groups:
            for n in (10, 20, 50, 100):
                sma = c.rolling(n, min_periods=n).mean()
                add(f"SMA_{n}", sma)
                add(f"Close_to_SMA_{n}", c/sma-1)
            for n in (10, 20, 50): add(f"EMA_{n}", c.ewm(span=n, adjust=False, min_periods=n).mean())
            for left, right in ((20, 50), (50, 100)):
                valid = result[f"SMA_{left}"].notna() & result[f"SMA_{right}"].notna()
                add(f"SMA{left}_above_SMA{right}", (result[f"SMA_{left}"] > result[f"SMA_{right}"]).astype(float).where(valid))
            for n in (5, 10, 20):
                add(f"ROC_{n}", c/c.shift(n)-1)
                add(f"Momentum_{n}", c-c.shift(n))
            macd = c.ewm(span=12, adjust=False, min_periods=12).mean()-c.ewm(span=26, adjust=False, min_periods=26).mean()
            add("MACD", macd)
            signal_line = macd.ewm(span=9, adjust=False, min_periods=9).mean()
            add("MACD_signal", signal_line)
            add("MACD_histogram", macd-signal_line)

            up, down = h.diff(), -l.diff()
            plus = up.where((up > down) & (up > 0), 0.0)
            minus = down.where((down > up) & (down > 0), 0.0)
            prev = c.shift(1)
            tr = pd.concat([h-l, (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
            atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
            pdi = 100*plus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()/atr
            mdi = 100*minus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()/atr
            dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0, np.nan)
            add("ADX_14", dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean())

        if "mean_reversion" in self.enabled_groups:
            delta = c.diff()
            gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
            rsi = 100-100/(1+gain/loss.replace(0, np.nan))
            add("RSI_14", rsi.where(loss.ne(0), 100.0))
            ma, sd = c.rolling(20, min_periods=20).mean(), c.rolling(20, min_periods=20).std(ddof=0)
            upper, lower = ma+2*sd, ma-2*sd
            add("BB_percent_B", (c-lower)/(upper-lower).replace(0, np.nan))
            add("BB_bandwidth", (upper-lower)/ma.replace(0, np.nan))
            for n in (20, 50): add(f"Price_ZScore_{n}", self._zscore(c, n))
            add("Distance_from_high_20", c/h.rolling(20, min_periods=20).max()-1)
            add("Distance_from_low_20", c/l.rolling(20, min_periods=20).min()-1)

        if "volatility" in self.enabled_groups:
            prev = c.shift(1)
            tr = pd.concat([h-l, (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
            add("ATR_14", tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean())
            for n in (10, 20, 30): add(f"Historical_Volatility_{n}", ret.rolling(n, min_periods=n).std()*np.sqrt(252))
            log_hl = np.log(h/l).replace([np.inf, -np.inf], np.nan)
            park = log_hl.pow(2)/(4*np.log(2))
            add("Parkinson_Volatility_20", np.sqrt(park.rolling(20, min_periods=20).mean()*252))
            add("High_Low_Range_pct", (h-l)/c.replace(0, np.nan))

        if "volume" in self.enabled_groups:
            for n in (10, 20):
                vol_sma = v.rolling(n, min_periods=n).mean()
                add(f"Volume_SMA_{n}", vol_sma)
                add(f"Volume_Ratio_{n}", v/vol_sma.replace(0, np.nan))
            add("OBV", (np.sign(c.diff()).fillna(0)*v).cumsum())
            add("Price_Volume_Trend", (ret.fillna(0)*v).cumsum())

        if "lags" in self.enabled_groups:
            for n in (1, 2, 3, 5): add(f"Return_lag_{n}", ret.shift(n))
            for n in range(1, 6): add(f"Close_lag_{n}", c.shift(n))

        # Targets use next close by definition; exclude these from predictors.
        next_return = c.shift(-1)/c-1
        result["target_return"] = next_return
        result["target_direction"] = (next_return > 0).astype(float).where(next_return.notna())
        result["target_direction_aggressive"] = (next_return > .005).astype(float).where(next_return.notna())
        self._feature_names = names
        return result

    def get_feature_names(self) -> list[str]:
        """Return latest feature names, excluding OHLCV and target columns."""
        return self._feature_names.copy()

    def select_features(self, df: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
        """Select requested generated feature columns or reject unknown names."""
        invalid = set(feature_list)-set(self._feature_names)
        if invalid: raise ValueError(f"Unknown feature(s): {sorted(invalid)}")
        return df.loc[:, feature_list].copy()



def generate_universe_features(
    config: dict[str, Any] | None = None,
    universe_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    enabled_groups: list[str] | None = None,
    file_format: str = "parquet",
) -> dict[str, Path]:
    """Create features for configured stocks; fall back to CSV without engines."""
    from src.data_loader import CSEDataLoader
    root = Path(__file__).resolve().parent.parent
    settings = config
    if settings is None:
        with (root / "config.yaml").open("r", encoding="utf-8") as stream:
            settings = yaml.safe_load(stream) or {}
    universe_file = Path(universe_path) if universe_path else root / "data" / "universe.yaml"
    with universe_file.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    symbols = [str(item["symbol"]) for item in document.get("stocks", []) if item.get("symbol")]
    if not symbols: raise ValueError(f"No symbols in {universe_file}")
    destination = Path(output_dir) if output_dir else root / str(settings.get("data", {}).get("processed_dir", "data/processed"))
    if not destination.is_absolute(): destination = root / destination
    destination.mkdir(parents=True, exist_ok=True)
    start = str(settings.get("universe", {}).get("start_date", "2015-01-01"))
    end = str(settings.get("universe", {}).get("end_date") or pd.Timestamp.now().strftime("%Y-%m-%d"))
    chosen = file_format.lower()
    if chosen not in ("parquet", "feather", "csv"): raise ValueError("file_format must be parquet, feather, or csv")
    loader = CSEDataLoader(root / "config.yaml")
    engineer = FeatureEngineer(settings, enabled_groups)
    outputs: dict[str, Path] = {}
    for symbol, frame in loader.get_multiple_stocks(symbols, start, end).items():
        if frame.empty:
            LOGGER.warning("Skipping %s: no history", symbol)
            continue
        features = engineer.create_features(frame, symbol)
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", symbol)
        path = destination / f"{safe}_features.{chosen}"
        try:
            if chosen == "parquet": features.to_parquet(path)
            elif chosen == "feather": features.reset_index().to_feather(path)
            else: features.to_csv(path, index=True)
        except (ImportError, ModuleNotFoundError) as exc:
            if chosen == "csv": raise
            LOGGER.warning("No %s engine; using CSV for %s: %s", chosen, symbol, exc)
            path = destination / f"{safe}_features.csv"
            features.to_csv(path, index=True)
        outputs[symbol] = path
    return outputs

