"""Feature engineering interfaces for future market-data research."""

from typing import TypeVar

import pandas as pd

FrameT = TypeVar("FrameT", bound=pd.DataFrame)


def validate_price_frame(frame: FrameT) -> FrameT:
    """Validate that a price table has a datetime index and at least one column."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Price data must use a pandas DatetimeIndex.")
    if frame.empty or len(frame.columns) == 0:
        raise ValueError("Price data must contain at least one row and one column.")
    return frame
