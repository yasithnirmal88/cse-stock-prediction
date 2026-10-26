"""Simple long-only trend-following baseline for daily OHLCV data."""
from __future__ import annotations

from typing import Literal
import pandas as pd


class SimpleMomentumStrategy:
    """20/50 moving-average trend strategy with no short exposure.

    Signal formation uses the current close; position is shifted one bar so
    returns are earned only after the signal could have been observed.
    """

    def __init__(
        self,
        lookback_period: int = 20,
        moving_average_type: Literal["sma", "ema"] = "sma",
        volatility_filter: float | None = None,
    ) -> None:
        if lookback_period < 1:
            raise ValueError("lookback_period must be positive")
        if moving_average_type not in ("sma", "ema"):
            raise ValueError("moving_average_type must be 'sma' or 'ema'")
        if volatility_filter is not None and volatility_filter < 0:
            raise ValueError("volatility_filter must be non-negative")
        self.lookback_period = lookback_period
        self.moving_average_type = moving_average_type
        self.volatility_filter = volatility_filter

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with momentum score, close signal, and next-bar position.

        MA20 uses ``lookback_period`` (20 by default); MA50 is fixed to satisfy
        the Phase 1 trend rule. Optional volatility_filter is a maximum rolling
        daily-return standard deviation, expressed as a decimal per session.
        """
        if "Close" not in df.columns:
            raise ValueError("df must contain a 'Close' column")
        result = df.copy()
        close = pd.to_numeric(result["Close"], errors="coerce")
        if self.moving_average_type == "sma":
            ma20 = close.rolling(self.lookback_period, min_periods=self.lookback_period).mean()
            ma50 = close.rolling(50, min_periods=50).mean()
        else:
            ma20 = close.ewm(span=self.lookback_period, adjust=False, min_periods=self.lookback_period).mean()
            ma50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
        result["momentum_score"] = ma20.div(ma50).sub(1.0)
        condition = (close > ma20) & (ma20 > ma50)
        if self.volatility_filter is not None:
            realized_vol = close.pct_change(fill_method=None).rolling(
                self.lookback_period, min_periods=self.lookback_period
            ).std()
            condition &= realized_vol <= self.volatility_filter
        result["signal"] = condition.fillna(False).astype("int8")
        result["position"] = result["signal"].shift(1, fill_value=0).astype("int8")
        return result
