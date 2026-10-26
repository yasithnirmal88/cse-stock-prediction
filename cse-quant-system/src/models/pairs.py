"""Statistical pairs and single-asset mean-reversion research tools."""
from __future__ import annotations
import logging
from itertools import combinations
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

class PairsTradingEngine:
    """Engle-Granger screening and market-neutral research.

    Full-history pair selection has look-ahead/selection bias. Re-screen per
    training window, then trade only subsequent dates. Uses all aligned input
    observations (default minimum 120). Set a positive median-dollar-turnover
    threshold to require volume-backed liquidity screening.
    """
    def __init__(self, min_observations=120, liquidity_lookback=60,
                 min_median_dollar_volume=0.0):
        if min_observations < 20 or liquidity_lookback < 1 or min_median_dollar_volume < 0:
            raise ValueError("invalid observations, liquidity lookback, or turnover threshold")
        self.min_observations = int(min_observations)
        self.liquidity_lookback = int(liquidity_lookback)
        self.min_median_dollar_volume = float(min_median_dollar_volume)
        self.pair_statistics_ = []
        self.last_pair_ = None

    @staticmethod
    def _close(frame):
        column = next((c for c in ("Close", "Adj Close", "close") if c in frame.columns), None)
        if column is None:
            raise ValueError("Each price frame must contain Close or Adj Close")
        values = pd.to_numeric(frame[column], errors="coerce")
        values = values.where(values > 0).dropna()
        values.name = "price"
        return values

    def _is_liquid(self, symbol, frame):
        if self.min_median_dollar_volume <= 0:
            return True
        if "Volume" not in frame:
            LOGGER.warning("Excluding %s: liquidity filter enabled but Volume is missing", symbol)
            return False
        turnover = (self._close(frame) * pd.to_numeric(frame["Volume"], errors="coerce")).dropna().tail(self.liquidity_lookback)
        return not turnover.empty and float(turnover.median()) >= self.min_median_dollar_volume

    def find_cointegrated_pairs(self, price_data: dict, significance=0.05) -> list:
        """Engle-Granger test on all aligned log-price history, min 120 points by default."""
        if not 0 < significance < 1:
            raise ValueError("significance must be between 0 and 1")
        try:
            from statsmodels.tsa.stattools import coint
        except ImportError as exc:
            raise ImportError("Pair screening requires statsmodels; install with `pip install statsmodels`") from exc
        eligible = {}
        for symbol, frame in price_data.items():
            if not isinstance(frame, pd.DataFrame) or frame.empty or not self._is_liquid(symbol, frame):
                continue
            try:
                eligible[symbol] = self._close(frame)
            except ValueError as exc:
                LOGGER.warning("Skipping %s: %s", symbol, exc)
        found = []
        for sy, sx in combinations(sorted(eligible), 2):
            aligned = pd.concat([eligible[sy].rename("y"), eligible[sx].rename("x")], axis=1).dropna()
            if len(aligned) < self.min_observations:
                continue
            y, x = np.log(aligned.y), np.log(aligned.x)
            if y.nunique() < 2 or x.nunique() < 2:
                continue
            beta = float(np.polyfit(x.to_numpy(), y.to_numpy(), 1)[0])
            try:
                _, p, _ = coint(y, x, trend="c", autolag="aic")
            except (ValueError, np.linalg.LinAlgError) as exc:
                LOGGER.warning("Cointegration failed for %s/%s: %s", sy, sx, exc)
                continue
            if np.isfinite(p) and p < significance:
                found.append({"pair": (sy, sx), "symbol_y": sy, "symbol_x": sx,
                              "p_value": float(p), "hedge_ratio": beta,
                              "observations": len(aligned), "start": aligned.index[0], "end": aligned.index[-1]})
        self.pair_statistics_ = sorted(found, key=lambda item: item["p_value"])
        LOGGER.info("Screened %d candidates; found %d cointegrated pairs", len(eligible), len(found))
        return [item.copy() for item in self.pair_statistics_]


    def calculate_spread(self, pair: tuple, price_data: dict) -> pd.DataFrame:
        """Return log-price residual spread and expanding causal z-score."""
        if len(pair) != 2 or any(s not in price_data for s in pair): raise ValueError("invalid pair")
        stat = next((p for p in self.pair_statistics_ if p["pair"] == tuple(pair)), None)
        if stat is None:
            stat = next((p for p in self.pair_statistics_ if p["pair"] == tuple(pair[::-1])), None)
            if stat is not None: pair = tuple(pair[::-1])
        if stat is None: raise ValueError("Pair has not been screened")
        y, x = (self._close(price_data[s]) for s in pair)
        out = pd.concat([y.rename("price_y"), x.rename("price_x")], axis=1).dropna()
        out["spread"] = np.log(out.price_y) - float(stat["hedge_ratio"]) * np.log(out.price_x)
        expanding = out.spread.expanding(min_periods=min(self.min_observations, len(out)))
        out["z_score"] = (out.spread - expanding.mean()) / expanding.std(ddof=1).replace(0, np.nan)
        self.last_pair_ = {**stat, "pair": tuple(pair)}
        return out

    @staticmethod
    def generate_pairs_signals(spread_df, entry_z=2.0, exit_z=0.5):
        """Persistent +1 long-spread / -1 short-spread / 0 flat signal."""
        if not 0 <= exit_z < entry_z or "z_score" not in spread_df: raise ValueError("invalid thresholds or missing z_score")
        position, signals = 0, []
        for value in pd.to_numeric(spread_df.z_score, errors="coerce"):
            if not np.isfinite(value): position = 0
            elif position == 0:
                if value < -entry_z: position = 1
                elif value > entry_z: position = -1
            elif abs(value) < exit_z: position = 0
            signals.append(position)
        out = spread_df.copy(); out["signal"] = signals; return out

    @staticmethod
    def single_stock_mean_reversion(df, lookback=20, entry_z=1.8):
        """Rolling price z-score: +1 buys oversold, -1 shorts overbought."""
        if lookback < 2 or entry_z <= 0: raise ValueError("invalid lookback or entry_z")
        close = PairsTradingEngine._close(df)
        mean = close.rolling(lookback, min_periods=lookback).mean()
        std = close.rolling(lookback, min_periods=lookback).std(ddof=1).replace(0, np.nan)
        out = pd.DataFrame({"price": close, "mean": mean, "z_score": (close - mean) / std})
        out["signal"] = np.select([out.z_score < -entry_z, out.z_score > entry_z], [1, -1], default=0)
        return out

    @staticmethod
    def allocate_pair_capital(pairs, capital: float, max_pair_fraction=0.2):
        """Equal-weight, dollar-neutral allocation with equal long and short legs."""
        if not np.isfinite(capital) or capital <= 0 or not 0 < max_pair_fraction <= 1: raise ValueError("invalid capital or cap")
        keys = [tuple(x["pair"] if isinstance(x, dict) else x) for x in pairs]
        if len(set(keys)) != len(keys): raise ValueError("pairs must be unique")
        if not keys: return {}
        budget = min(capital / len(keys), capital * max_pair_fraction)
        return {key: {"long_notional": budget / 2, "short_notional": budget / 2,
                       "gross_notional": budget, "net_notional": 0.0} for key in keys}

    def construct_portfolio(self, pairs, capital, max_pair_fraction=0.2):
        return self.allocate_pair_capital(pairs, capital, max_pair_fraction)
