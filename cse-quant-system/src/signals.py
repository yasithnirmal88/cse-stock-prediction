"""Causal combination of trend, ML, GARCH sizing, and a pairs sleeve."""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from src.models.garch_model import VolatilityManager
from src.models.pairs import PairsTradingEngine
from src.models.xgboost_model import DirectionProbabilityModel

LOGGER = logging.getLogger(__name__)
OUTPUT_COLUMNS = ["date", "symbol", "signal", "probability", "volatility_regime",
                  "suggested_position_size", "strategy_source", "confidence_score"]

class SignalEngine:
    """Combine trend-gated probabilities, GARCH sizing, and an independent pairs sleeve.

    Supply a trained DirectionProbabilityModel that was fitted only on data
    earlier than the signal date (prefer walk-forward predictions). Pair selection
    is date-bounded but should also be reselected walk-forward in backtests.
    """
    def __init__(self, price_data=None, model=None, capital=1_000_000,
                 probability_threshold=0.55, max_position_pct=0.10,
                 momentum_fast=20, momentum_slow=50, pair_capital_fraction=0.20,
                 model_path=None, signal_path=None):
        if capital <= 0 or not 0 <= probability_threshold <= 1 or not 0 < max_position_pct <= 1:
            raise ValueError("invalid capital or thresholds")
        self.price_data, self.model = price_data or {}, model
        self.capital, self.probability_threshold = float(capital), float(probability_threshold)
        self.max_position_pct = float(max_position_pct)
        self.momentum_fast, self.momentum_slow = int(momentum_fast), int(momentum_slow)
        self.pair_capital_fraction = float(pair_capital_fraction)
        self.model_path = Path(model_path) if model_path else None
        self.signal_path = Path(signal_path) if signal_path else Path("reports/latest_signals.csv")
        self._signals = pd.DataFrame(columns=OUTPUT_COLUMNS)
        if self.model is None and self.model_path and self.model_path.is_file():
            self.model = DirectionProbabilityModel.load(self.model_path)

    @staticmethod
    def _date_index(frame):
        out = frame.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            col = next((c for c in ("Date", "date", "Datetime") if c in out), None)
            if col is None: raise ValueError("Data needs DatetimeIndex or Date column")
            out[col] = pd.to_datetime(out[col], errors="coerce")
            out = out.dropna(subset=[col]).set_index(col)
        out.index = pd.to_datetime(out.index)
        return out.loc[~out.index.isna()].sort_index().loc[lambda x: ~x.index.duplicated(keep="last")]

    @staticmethod
    def _close(frame):
        col = next((c for c in ("Close", "Adj Close", "close") if c in frame), None)
        if col is None: raise ValueError("price frame must contain Close")
        return pd.to_numeric(frame[col], errors="coerce")

    def _momentum_uptrend(self, frame, date):
        close = self._close(frame.loc[:date]).dropna()
        if len(close) < self.momentum_slow: return False
        fast = close.rolling(self.momentum_fast, min_periods=self.momentum_fast).mean().iloc[-1]
        slow = close.rolling(self.momentum_slow, min_periods=self.momentum_slow).mean().iloc[-1]
        return bool(np.isfinite(fast) and np.isfinite(slow) and close.iloc[-1] > slow and fast > slow)

    def _probability(self, frame, date):
        if self.model is None or getattr(self.model, "model", None) is None:
            raise RuntimeError("A trained DirectionProbabilityModel is required")
        features = getattr(self.model, "feature_names_", [])
        if not features: raise RuntimeError("Trained model lacks feature_names_ metadata")
        row = frame.loc[:date].iloc[-1:]
        absent = set(features) - set(row.columns)
        if absent: raise ValueError(f"Missing model features: {sorted(absent)}")
        x = row.loc[:, features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if x.isna().any(axis=None): return float("nan")
        return float(self.model.predict_proba(x)[0])

    def generate_combined_signals(self, date: str = None) -> pd.DataFrame:
        """Trend-gate, probability-filter, then apply GARCH sizing."""
        if not self.price_data: raise ValueError("price_data is empty")
        frames = {s: self._date_index(f) for s, f in self.price_data.items()}
        cutoff = pd.Timestamp(date) if date else max(f.index.max() for f in frames.values())
        cutoff = cutoff.normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        rows, garch = [], VolatilityManager()
        for symbol, frame in frames.items():
            hist = frame.loc[:cutoff]
            if hist.empty: continue
            uptrend = self._momentum_uptrend(hist, cutoff)
            probability = self._probability(hist, cutoff) if uptrend and self.model is not None else float("nan")
            active = bool(uptrend and np.isfinite(probability) and probability >= self.probability_threshold)
            regime, size = "Unknown", 0.0
            if active:
                try:
                    returns = garch.calculate_returns(hist)
                    if len(returns) < 30: raise ValueError("fewer than 30 returns")
                    garch.fit_garch(returns)
                    forecast = garch.forecast_volatility()
                    sizing = garch.calculate_position_size(self.capital, float(self._close(hist).dropna().iloc[-1]),
                        forecast, max_position_pct=self.max_position_pct)
                    regime, size = sizing["vol_regime"], float(sizing["position_value"])
                except (ImportError, ValueError, RuntimeError) as exc:
                    LOGGER.warning("No GARCH size for %s: %s", symbol, exc)
                    active = False
            rows.append({"date": cutoff.date().isoformat(), "symbol": symbol, "signal": int(active),
                "probability": probability, "volatility_regime": regime, "suggested_position_size": size,
                "strategy_source": "Momentum+ML+GARCH" if active else "Momentum+ML (filtered/flat)",
                "confidence_score": float(probability) if active else 0.0})

        # Separate pairs sleeve; only observations through cutoff are supplied.
        prices = {s: f.loc[:cutoff] for s, f in frames.items() if not f.loc[:cutoff].empty}
        try:
            pair_engine = PairsTradingEngine()
            pairs = pair_engine.find_cointegrated_pairs(prices)
            if pairs:
                budget = self.capital * self.pair_capital_fraction / len(pairs)
                for item in pairs:
                    pair = item["pair"]
                    signal = pair_engine.generate_pairs_signals(pair_engine.calculate_spread(pair, prices))
                    pos = int(signal.signal.iloc[-1]) if len(signal) else 0
                    for symbol, direction in zip(pair, (pos, -pos)):
                        rows.append({"date": cutoff.date().isoformat(), "symbol": symbol, "signal": direction,
                            "probability": np.nan, "volatility_regime": "Pairs sleeve",
                            "suggested_position_size": budget / 2 if direction else 0.0,
                            "strategy_source": f"Pairs ({pair[0]}/{pair[1]})",
                            "confidence_score": max(0.0, 1.0 - item["p_value"]) if direction else 0.0})
        except ImportError as exc:
            LOGGER.warning("Skipping pairs sleeve: %s", exc)
        self._signals = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        return self._signals.copy()

    def get_current_signals(self) -> pd.DataFrame:
        """Return most recently generated in-memory signals."""
        return self._signals.copy()

    def save_signals_to_csv(self, path=None) -> Path:
        """Save current signals and return their CSV path."""
        destination = Path(path) if path else self.signal_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._signals.to_csv(destination, index=False)
        return destination

    def load_latest_signals(self, path=None) -> pd.DataFrame:
        """Load saved rows for the latest date; return an empty schema if absent."""
        source = Path(path) if path else self.signal_path
        if not source.is_file(): return pd.DataFrame(columns=OUTPUT_COLUMNS)
        frame = pd.read_csv(source)
        if "date" in frame and not frame.empty:
            frame = frame.loc[frame.date == frame.date.max()].reset_index(drop=True)
        self._signals = frame
        return frame.copy()

