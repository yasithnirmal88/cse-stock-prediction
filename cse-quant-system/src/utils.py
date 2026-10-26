"""Shared project utilities."""

import logging
from typing import Any

import numpy as np
from scipy import stats


def configure_logging(config: dict[str, Any]) -> None:
    """Configure console logging from the project's logging settings."""
    settings = config.get("logging", {})
    level_name = str(settings.get("level", "INFO")).upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid logging level: {level_name}")
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def diebold_mariano_test(loss_a: Any, loss_b: Any, h: int = 1) -> tuple[float, float]:
    """Two-sided Diebold-Mariano test of equal predictive accuracy.

    Uses Harvey-Leybourne-Newbold (HLN) small-sample correction and Bartlett kernel
    for long-run covariance estimation across forecast horizon h.

    Parameters:
    -----------
    loss_a : array-like
        Per-period forecast error or loss from model A (e.g. squared errors or absolute errors).
    loss_b : array-like
        Per-period forecast error or loss from model B.
    h : int, default=1
        Forecast horizon in periods. For overlapping h-step-ahead forecasts,
        uses Bartlett weights over h-1 lags.

    Returns:
    --------
    statistic : float
        Diebold-Mariano test statistic. Negative value indicates Model A has lower loss.
    p_value : float
        Two-sided p-value from Student's t distribution with (n - 1) degrees of freedom.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 10 or np.allclose(d, 0.0):
        return float("nan"), float("nan")

    dbar = d.mean()
    lrv = np.mean((d - dbar) ** 2)
    for k in range(1, h):
        weight = 1.0 - (k / h)
        lrv += 2.0 * weight * np.mean((d[k:] - dbar) * (d[:-k] - dbar))

    lrv = max(lrv, 1e-18)
    stat = dbar / np.sqrt(lrv / n)
    # Harvey-Leybourne-Newbold (HLN) small sample correction
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p_value = 2.0 * (1.0 - stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p_value)
