"""CSE market-data loading, normalization, source fallback, and local caching.

Free public providers can change or restrict access without notice; validate data.
"""
from __future__ import annotations
import json
import logging
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
import requests
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
CANONICAL_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
INTERVALS = {"1d": "1d", "15min": "15m", "30min": "30m", "1hour": "60m"}
USER_AGENT = "CSEQuantResearch/0.1 (academic research; contact: research@example.invalid)"


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load project YAML configuration."""
    import yaml
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return config


def resolve_data_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a configured data path relative to project root."""
    try:
        value = Path(str(config["data"][key]))
    except (KeyError, TypeError) as exc:
        raise KeyError(f"Missing data path configuration: data.{key}") from exc
    root = Path(__file__).resolve().parent.parent
    return value if value.is_absolute() else root / value


class CSEDataError(RuntimeError):
    """Critical data-loader configuration or source error."""


class CSEDataLoader:
    """Free-source OHLCV loader with cache and validated source fallbacks.

    The CSE endpoint is undocumented and free providers may not list CSE shares.
    Failures are logged; ASPI is labelled as a proxy, never presented as SL20.
    """
    def __init__(self, config_path: str | Path = "config.yaml") -> None:
        self.config = load_config(config_path)
        data = self.config.get("data", {})
        self.cache_enabled = bool(data.get("cache_enabled", True))
        self.cache_expiry_hours = float(data.get("cache_expiry_hours", 24))
        if self.cache_expiry_hours < 0:
            raise CSEDataError("data.cache_expiry_hours must be non-negative")
        self.raw_dir = resolve_data_path(self.config, "raw_dir")
        self.processed_dir = resolve_data_path(self.config, "processed_dir")
        self.cache_dir = self.processed_dir / "cache"
        for folder in (self.raw_dir, self.processed_dir, self.cache_dir):
            folder.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json,text/csv,*/*"})
        self._last_request_at = 0.0
        self._request_lock = threading.Lock()
        self.minimum_request_interval = 1.75

    @staticmethod
    def _safe_symbol(symbol: str) -> str:
        value = re.sub(r"[^A-Za-z0-9._-]", "_", symbol.strip())
        if not value:
            raise ValueError("symbol must not be empty")
        return value

    def _cache_paths(self, symbol: str, interval: str) -> tuple[Path, Path]:
        stem = f"{self._safe_symbol(symbol)}_{self._safe_symbol(interval)}"
        return self.cache_dir / f"{stem}.csv", self.cache_dir / f"{stem}.json"

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        """Make HTTP requests at least 1.75 seconds apart per loader."""
        with self._request_lock:
            wait = self.minimum_request_interval - (time.monotonic() - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
            self._last_request_at = time.monotonic()
        response = self.session.request(method, url, timeout=(10, 30), **kwargs)
        response.raise_for_status()
        return response

    def is_cache_valid(self, symbol: str, interval: str) -> bool:
        """Check cache existence and UTC retrieval timestamp against TTL."""
        if not self.cache_enabled:
            return False
        csv_path, meta_path = self._cache_paths(symbol, interval)
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            timestamp = datetime.fromisoformat(meta["downloaded_at"])
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)).total_seconds() / 3600
            return csv_path.is_file() and 0 <= age <= self.cache_expiry_hours
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return False

    def load_from_cache(self, symbol: str, interval: str) -> pd.DataFrame | None:
        """Load valid cached data or return None when missing, corrupt, or stale."""
        if not self.is_cache_valid(symbol, interval):
            return None
        csv_path, _ = self._cache_paths(symbol, interval)
        try:
            frame = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
            frame.index = pd.DatetimeIndex(frame.index, name="Date")
            source = str(frame["Source"].iloc[0]) if "Source" in frame and len(frame) else "cache"
            return self._clean_frame(frame, source)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            LOGGER.warning("Could not read cache for %s (%s): %s", symbol, interval, exc)
            return None

    def update_cache(self, symbol: str, df: pd.DataFrame, interval: str) -> None:
        """Write normalized observations and UTC retrieval timestamp atomically."""
        if not self.cache_enabled or df.empty:
            return
        csv_path, meta_path = self._cache_paths(symbol, interval)
        csv_tmp, meta_tmp = csv_path.with_suffix(".csv.tmp"), meta_path.with_suffix(".json.tmp")
        frame = df.copy()
        frame.index.name = "Date"
        source = str(frame["Source"].iloc[0]) if "Source" in frame else "unknown"
        try:
            frame.to_csv(csv_tmp, index=True, date_format="%Y-%m-%dT%H:%M:%S")
            meta_tmp.write_text(json.dumps({"symbol": symbol, "interval": interval,
                "downloaded_at": datetime.now(timezone.utc).isoformat(), "source": source}, indent=2), encoding="utf-8")
            csv_tmp.replace(csv_path)
            meta_tmp.replace(meta_path)
        except OSError:
            csv_tmp.unlink(missing_ok=True)
            meta_tmp.unlink(missing_ok=True)
            LOGGER.exception("Failed to update cache for %s (%s)", symbol, interval)

    @staticmethod
    def _clean_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
        """Normalize columns, dates, duplicates, numeric values, and small gaps."""
        if frame is None or frame.empty:
            return pd.DataFrame(columns=CANONICAL_COLUMNS + ["Source"], index=pd.DatetimeIndex([], name="Date"))
        result = frame.copy()
        aliases = {"open": "Open", "high": "High", "low": "Low", "close": "Close",
                   "adj_close": "Close", "adjusted_close": "Close", "volume": "Volume",
                   "date": "Date", "datetime": "Date", "timestamp": "Date"}
        rename = {column: aliases.get(str(column).strip().lower().replace(" ", "_"), column)
                  for column in result.columns}
        result = result.rename(columns=rename)
        if "Date" in result.columns:
            result["Date"] = pd.to_datetime(result["Date"], errors="coerce", utc=True).dt.tz_localize(None)
            result = result.set_index("Date")
        result.index = pd.to_datetime(result.index, errors="coerce", utc=True).tz_localize(None)
        result.index.name = "Date"
        result = result.loc[~result.index.isna()].sort_index()
        result = result.loc[~result.index.duplicated(keep="last")]
        for column in CANONICAL_COLUMNS:
            if column not in result:
                result[column] = pd.NA
            result[column] = pd.to_numeric(result[column], errors="coerce")
        result[CANONICAL_COLUMNS] = result[CANONICAL_COLUMNS].ffill(limit=2)
        result["Source"] = source
        return result[CANONICAL_COLUMNS + ["Source"]]

    @staticmethod
    def _date_bounds(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
        if start.tzinfo is not None: start = start.tz_convert(None)
        if end.tzinfo is not None: end = end.tz_convert(None)
        end = min(end, pd.Timestamp.now().tz_localize(None))
        if start > end:
            raise ValueError("start_date must be on or before end_date and not in the future")
        return start, end

    def _download_yahoo(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp, interval: str) -> pd.DataFrame:
        """Attempt Yahoo chart API; CSE coverage is not guaranteed."""
        ticker = symbol if symbol.upper().endswith(".CO") else f"{symbol}.CO"
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        response = self._request("GET", url, params={"period1": int(start.timestamp()), "period2": int((end + pd.Timedelta(days=1)).timestamp()), "interval": INTERVALS[interval], "events": "history"})
        payload = response.json()["chart"]
        if payload.get("error") or not payload.get("result"):
            raise CSEDataError(f"Yahoo returned no result: {payload.get('error')}")
        chart = payload["result"][0]
        quote = chart["indicators"]["quote"][0]
        dates = pd.to_datetime(chart["timestamp"], unit="s", utc=True).tz_localize(None)
        frame = pd.DataFrame({"Date": dates, **{key.capitalize(): quote.get(key) for key in ("open", "high", "low", "close", "volume")}})
        return self._clean_frame(frame, "Yahoo Finance chart")

    def _download_cse(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Attempt the CSE public historical-price endpoint for daily data."""
        response = self._request("POST", "https://www.cse.lk/api/stockPriceHistory",
            json={"symbol": symbol, "from": start.strftime("%Y-%m-%d"), "to": end.strftime("%Y-%m-%d")},
            headers={"Origin": "https://www.cse.lk", "Referer": "https://www.cse.lk/"})
        body = response.json()
        rows: Any = body
        if isinstance(body, dict):
            for key in ("data", "history", "prices", "records", "stockPriceHistory"):
                if isinstance(body.get(key), list): rows = body[key]; break
        if not isinstance(rows, list) or not rows:
            raise CSEDataError("CSE response contained no recognizable price rows")
        return self._clean_frame(pd.DataFrame(rows), "CSE public API")



    def get_stock_data(self, symbol: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
        """Get a symbol's OHLCV data using valid cache then free sources."""
        if interval not in INTERVALS:
            raise ValueError(f"Unsupported interval {interval!r}; choose {', '.join(INTERVALS)}")
        start, end = self._date_bounds(start_date, end_date)
        cached = self.load_from_cache(symbol, interval)
        if cached is not None:
            cached = cached.loc[(cached.index >= start) & (cached.index <= end)]
            if not cached.empty:
                LOGGER.info("Using cache for %s (%s)", symbol, interval)
                return cached
        names = (["CSE public API"] if interval == "1d" else []) + ["Yahoo Finance chart"]
        for name in names:
            try:
                LOGGER.info("Trying %s for %s (%s)", name, symbol, interval)
                frame = self._download_cse(symbol, start, end) if name == "CSE public API" else self._download_yahoo(symbol, start, end, interval)
                frame = self._clean_frame(frame, name)
                frame = frame.loc[(frame.index >= start) & (frame.index <= end)]
                frame = frame.loc[frame.index <= pd.Timestamp.now().tz_localize(None)]
                if frame.empty or frame["Close"].dropna().empty:
                    raise CSEDataError("Source returned no usable close prices")
                self.update_cache(symbol, frame, interval)
                LOGGER.info("Loaded %d rows for %s from %s", len(frame), symbol, name)
                return frame
            except Exception as exc:
                LOGGER.warning("Source %s failed for %s (%s): %s", name, symbol, interval, exc)
        LOGGER.error("All free data sources failed for %s (%s)", symbol, interval)
        return pd.DataFrame(columns=CANONICAL_COLUMNS + ["Source"], index=pd.DatetimeIndex([], name="Date"))

    def get_multiple_stocks(self, symbols: list[str], start_date: str, end_date: str, interval: str = "1d") -> dict[str, pd.DataFrame]:
        """Load multiple symbols independently; preserve failures as empty frames."""
        results: dict[str, pd.DataFrame] = {}
        for symbol in tqdm(symbols, desc="Loading CSE stocks", unit="stock"):
            try:
                results[symbol] = self.get_stock_data(symbol, start_date, end_date, interval)
            except Exception as exc:
                LOGGER.exception("Unexpected failure for %s; continuing: %s", symbol, exc)
                results[symbol] = pd.DataFrame(columns=CANONICAL_COLUMNS + ["Source"], index=pd.DatetimeIndex([], name="Date"))
        return results

    def get_benchmark_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Attempt S&P SL20, then explicitly label ASPI fallback as proxy."""
        start, end = self._date_bounds(start_date, end_date)
        candidates = (("^SL20", "Yahoo Finance (^SL20; S&P SL20 attempt)"),
                      ("^ASI", "Yahoo Finance (^ASI; ASPI proxy, not S&P SL20)"))
        for symbol, label in candidates:
            try:
                LOGGER.info("Trying benchmark source %s", label)
                frame = self._download_yahoo(symbol, start, end, "1d")
                frame = frame.loc[(frame.index >= start) & (frame.index <= end)]
                if frame.empty or frame["Close"].dropna().empty:
                    raise CSEDataError("No usable benchmark bars")
                frame["Source"] = label
                self.update_cache(f"benchmark_{symbol.replace('^', '')}", frame, "1d")
                return frame
            except Exception as exc:
                LOGGER.warning("Benchmark source %s failed: %s", label, exc)
        LOGGER.error("S&P SL20 and ASPI benchmark attempts failed")
        return pd.DataFrame(columns=CANONICAL_COLUMNS + ["Source"], index=pd.DatetimeIndex([], name="Date"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    loader = CSEDataLoader()
    end = pd.Timestamp.now().normalize()
    start = end - pd.DateOffset(months=3)
    stocks = loader.get_multiple_stocks(["JKH.N0000", "COMB.N0000", "SAMP.N0000"], start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    benchmark = loader.get_benchmark_data(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    LOGGER.info("Loaded %d stock frames; benchmark rows=%d", len(stocks), len(benchmark))
