"""GARCH volatility forecasting and volatility-aware position sizing."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class VolatilityManager:
    """Estimate conditional volatility and scale exposures to a target risk.

    GARCH models volatility clustering and persistence that a fixed-window
    historical standard deviation cannot represent. Its common Gaussian
    innovation assumption can understate tail risk and does not capture every
    regime jump; position sizing is therefore capped and is not a guarantee
    against losses. Risk-based exposure reduces risk concentration/drawdown
    potential, but does not eliminate market or gap risk.
    """

    def __init__(self) -> None:
        self.model: Any | None = None
        self.fit_result: Any | None = None
        self.returns_: pd.Series | None = None
        self.historical_volatility_: pd.Series = pd.Series(dtype=float)
        self.forecast_volatility_: float | None = None

    @staticmethod
    def calculate_returns(df: pd.DataFrame) -> pd.Series:
        """Calculate simple close-to-close returns from OHLCV or Close data."""
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("df must be a non-empty DataFrame")
        close_col = next((c for c in ("Close", "Adj Close", "close") if c in df.columns), None)
        if close_col is None:
            raise ValueError("df must contain a Close or Adj Close column")
        close = pd.to_numeric(df[close_col], errors="coerce")
        close = close.where(close > 0)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
        returns.name = "returns"
        return returns

    def fit_garch(self, returns: pd.Series, p: int = 1, q: int = 1) -> Any:
        """Fit a GARCH(p,q) model to percentage returns using ``arch``."""
        if p < 1 or q < 1:
            raise ValueError("p and q must be positive integers")
        series = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) < max(30, p + q + 5):
            raise ValueError("At least 30 finite return observations are required")
        if not isinstance(series.index, pd.DatetimeIndex):
            LOGGER.warning("Returns lack a DatetimeIndex; retaining their supplied order")
        self.returns_ = series.astype(float)
        # arch expects percentage-point returns; forecasts are converted back to decimal units.
        try:
            from arch import arch_model
        except ImportError as exc:
            raise ImportError("GARCH fitting requires the 'arch' package; install with `pip install arch`") from exc
        self.model = arch_model(self.returns_ * 100.0, mean="Constant", vol="GARCH", p=p, q=q,
                                dist="normal", rescale=False)
        try:
            result = self.model.fit(disp="off", show_warning=False, options={"maxiter": 1000})
            if not bool(getattr(result, "convergence_flag", 1) == 0):
                LOGGER.warning("GARCH optimizer did not converge (flag=%s); retrying with higher iteration limit",
                               getattr(result, "convergence_flag", "unknown"))
                result = self.model.fit(disp="off", show_warning=False, options={"maxiter": 3000, "ftol": 1e-8})
            self.fit_result = result
            if getattr(result, "convergence_flag", 1) != 0:
                LOGGER.warning("GARCH retry did not converge; retaining best available fit")
            cond = pd.Series(np.asarray(result.conditional_volatility, dtype=float) / 100.0,
                             index=self.returns_.index, name="conditional_volatility")
            self.historical_volatility_ = cond * np.sqrt(252.0)
            LOGGER.info("GARCH(%d,%d) fitted on %d returns (convergence=%s)",
                        p, q, len(series), getattr(result, "convergence_flag", "unknown"))
            return result
        except Exception as exc:
            self.fit_result = None
            LOGGER.exception("GARCH fit failed")
            raise RuntimeError(f"GARCH fitting failed: {exc}") from exc

    def forecast_volatility(self, horizon: int = 1) -> float:
        """Forecast annualized decimal volatility (e.g. 0.20 = 20%)."""
        if self.fit_result is None:
            raise RuntimeError("Fit a GARCH model before forecasting")
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        try:
            forecast = self.fit_result.forecast(horizon=horizon, reindex=False)
            variance = np.asarray(forecast.variance.iloc[-1], dtype=float)
            # Aggregate daily variances across horizon, then annualize by 252 sessions.
            vol = float(np.sqrt(np.sum(variance[:horizon]) / 10000.0 * 252.0 / horizon))
        except Exception as exc:
            LOGGER.exception("GARCH forecast failed")
            raise RuntimeError(f"GARCH forecasting failed: {exc}") from exc
        if not np.isfinite(vol) or vol <= 0:
            raise RuntimeError("GARCH produced a non-positive or non-finite volatility forecast")
        self.forecast_volatility_ = vol
        return vol


    def get_volatility_regime(self, current_vol: float, lookback: int = 60) -> str:
        """Classify annualized volatility by trailing 33rd/67th percentiles."""
        if not np.isfinite(current_vol) or current_vol < 0:
            raise ValueError("current_vol must be finite and non-negative")
        if lookback < 2:
            raise ValueError("lookback must be at least 2")
        history = self.historical_volatility_.dropna().tail(lookback)
        if len(history) < 3:
            LOGGER.warning("Insufficient volatility history; classifying forecast as Normal")
            return "Normal"
        low, high = history.quantile([1 / 3, 2 / 3])
        if current_vol < low:
            return "Low"
        if current_vol > high:
            return "High"
        return "Normal"

    def calculate_position_size(self, capital: float, current_price: float,
                                forecasted_vol: float, target_risk_per_trade: float = 0.01,
                                max_position_pct: float = 0.10) -> dict[str, float | int | str]:
        """Size whole shares to one-session risk budget, capped by capital percentage."""
        values = (capital, current_price, forecasted_vol, target_risk_per_trade, max_position_pct)
        if not all(np.isfinite(v) for v in values):
            raise ValueError("Sizing inputs must be finite")
        if capital <= 0 or current_price <= 0 or forecasted_vol <= 0:
            raise ValueError("capital, price and volatility must be positive")
        if not 0 < target_risk_per_trade <= 1 or not 0 < max_position_pct <= 1:
            raise ValueError("risk and max position percentages must be in (0, 1]")
        daily_vol = forecasted_vol / np.sqrt(252.0)
        risk_budget = capital * target_risk_per_trade
        value = min(risk_budget / daily_vol, capital * max_position_pct)
        shares = int(np.floor(value / current_price))
        value = shares * current_price
        return {"position_value": float(value), "shares": shares,
                "position_pct": float(value / capital), "risk_amount": float(value * daily_vol),
                "vol_regime": self.get_volatility_regime(forecasted_vol)}

    def apply_volatility_sizing(self, signals_df: pd.DataFrame, price_df: pd.DataFrame,
                                capital: float) -> pd.DataFrame:
        """Attach causal per-date GARCH sizing; every fit sees prices through signal date only."""
        for name, frame in (("signals_df", signals_df), ("price_df", price_df)):
            if not isinstance(frame.index, pd.DatetimeIndex):
                raise TypeError(f"{name} must have DatetimeIndex")
            if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
                raise ValueError(f"{name} index must be sorted and unique")
        close_col = next((c for c in ("Close", "Adj Close", "close") if c in price_df), None)
        signal_col = next((c for c in ("signal", "prediction", "predicted_signal") if c in signals_df), None)
        if close_col is None or signal_col is None:
            raise ValueError("price_df needs Close; signals_df needs signal/prediction")
        rows = []
        for date, signal in signals_df[signal_col].items():
            prices = pd.to_numeric(price_df.loc[price_df.index <= date, close_col], errors="coerce").dropna()
            if prices.empty or pd.isna(signal):
                rows.append({"forecasted_vol": np.nan, "position_value": 0., "shares": 0,
                             "position_pct": 0., "risk_amount": 0., "vol_regime": "Unknown"})
                continue
            returns = self.calculate_returns(prices.to_frame("Close"))
            if len(returns) < 30:
                LOGGER.warning("Insufficient returns as of %s; assigning zero exposure", date)
                rows.append({"forecasted_vol": np.nan, "position_value": 0., "shares": 0,
                             "position_pct": 0., "risk_amount": 0., "vol_regime": "Unknown"})
                continue
            # Reinitialize each as-of fit so future dates cannot inform historical estimates.
            self.model = self.fit_result = None
            self.fit_garch(returns)
            vol = self.forecast_volatility()
            sized = self.calculate_position_size(capital, float(prices.iloc[-1]), vol)
            if float(signal) <= 0:
                sized.update(position_value=0., shares=0, position_pct=0., risk_amount=0.)
            rows.append({"forecasted_vol": vol, **sized})
        output = signals_df.copy()
        sizes = pd.DataFrame(rows, index=signals_df.index)
        for column in sizes:
            output[column] = sizes[column]
        return output

