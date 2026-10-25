"""Daily, equal-weighted, long-only portfolio backtesting with explicit costs."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import numpy as np
import pandas as pd


class SimpleBacktester:
    """Backtest per-symbol binary positions with costs on every change."""
    def __init__(self, initial_capital: float = 1_000_000.0, entry_cost: float = 0.0056, exit_cost: float = 0.0056, periods_per_year: int = 252) -> None:
        if initial_capital <= 0 or min(entry_cost, exit_cost) < 0 or max(entry_cost, exit_cost) >= 1 or periods_per_year < 1:
            raise ValueError("invalid backtester parameters")
        self.initial_capital, self.entry_cost, self.exit_cost = float(initial_capital), float(entry_cost), float(exit_cost)
        self.periods_per_year = int(periods_per_year)

    @staticmethod
    def _mapping(data: Mapping[str, pd.DataFrame] | pd.DataFrame, name: str) -> dict[str, pd.DataFrame]:
        if isinstance(data, pd.DataFrame): return {name: data}
        if not isinstance(data, Mapping): raise TypeError(f"{name} must be a DataFrame or mapping")
        return {str(k): v for k, v in data.items() if isinstance(v, pd.DataFrame) and not v.empty}

    @staticmethod
    def _date_index(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        if "Date" in result.columns:
            result.index = pd.to_datetime(result.pop("Date"), errors="coerce", utc=True).dt.tz_localize(None)
        else:
            result.index = pd.to_datetime(result.index, errors="coerce", utc=True).tz_localize(None)
        result.index.name = "Date"
        result = result.loc[~result.index.isna()].sort_index()
        return result.loc[~result.index.duplicated(keep="last")]

    def _metrics(self, returns: pd.Series, equity: pd.Series, trades: int, wins: int) -> dict[str, Any]:
        clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
        total = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) else 0.0
        annual = float((equity.iloc[-1] / equity.iloc[0]) ** (self.periods_per_year / max(len(clean), 1)) - 1) if len(equity) and equity.iloc[-1] >= 0 else -1.0
        drawdown = equity / equity.cummax() - 1
        volatility = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
        sharpe = float(clean.mean() / volatility * np.sqrt(self.periods_per_year)) if volatility > 0 else 0.0
        return {"total_return": total, "annualized_return": annual, "maximum_drawdown": float(drawdown.min()) if len(drawdown) else 0.0, "sharpe_ratio": sharpe, "number_of_trades": int(trades), "win_rate": float(wins / trades) if trades else 0.0}

    def run_backtest(self, price_data: Mapping[str, pd.DataFrame] | pd.DataFrame, signals: Mapping[str, pd.DataFrame] | pd.DataFrame | None = None, benchmark: pd.DataFrame | None = None, benchmark_label: str = "Benchmark") -> dict[str, Any]:
        """Run equal-weight holdings; positions must be lagged before returns."""
        prices = self._mapping(price_data, "price_data")
        if not prices: raise ValueError("no price data supplied")
        signal_map = self._mapping(signals, "signals") if signals is not None else prices
        per_symbol, positions = {}, {}
        for symbol, raw in prices.items():
            frame = self._date_index(raw)
            if "Close" not in frame: continue
            close = pd.to_numeric(frame["Close"], errors="coerce")
            sf = self._date_index(signal_map.get(symbol, frame))
            column = "position" if "position" in sf else "signal" if "signal" in sf else None
            if column is None: raise ValueError(f"no position/signal for {symbol}")
            position = pd.to_numeric(sf[column], errors="coerce")
            if column == "signal": position = position.shift(1)
            position = position.reindex(frame.index).fillna(0).clip(0, 1)
            changes = position.diff().fillna(position)
            costs = changes.clip(lower=0) * self.entry_cost + (-changes).clip(lower=0) * self.exit_cost
            per_symbol[symbol] = (close.pct_change(fill_method=None).fillna(0) * position - costs).rename(symbol)
            positions[symbol] = position
        if not per_symbol: raise ValueError("no usable Close series supplied")
        asset_returns = pd.concat(per_symbol, axis=1).sort_index().fillna(0)
        strategy_returns = asset_returns.mean(axis=1)
        strategy_equity = self.initial_capital * (1 + strategy_returns).cumprod()
        trade_count = wins = 0
        for symbol, position in positions.items():
            changes = position.diff().fillna(position)
            entries, exits = list(position.index[changes > 0]), list(position.index[changes < 0])
            used_exits = set()
            for entry in entries:
                candidates = [d for d in exits if d > entry and d not in used_exits]
                if candidates:
                    exit_date = candidates[0]; used_exits.add(exit_date); trade_count += 1
                    wins += int(per_symbol[symbol].loc[entry:exit_date].sum() > 0)
            if len(entries) > len(exits):
                trade_count += 1; wins += int(per_symbol[symbol].loc[entries[-1]:].sum() > 0)
        curves = pd.DataFrame({"Strategy": strategy_equity})
        metrics = {"Strategy": self._metrics(strategy_returns, strategy_equity, trade_count, wins)}
        if benchmark is not None and not benchmark.empty:
            bf = self._date_index(benchmark)
            if "Close" not in bf: raise ValueError("benchmark must contain Close")
            bc = pd.to_numeric(bf["Close"], errors="coerce").dropna().reindex(strategy_returns.index).ffill().dropna()
            br = bc.pct_change(fill_method=None).fillna(0)
            be = self.initial_capital * (1 + br).cumprod()
            curves = curves.join(be.rename(benchmark_label), how="left").ffill()
            metrics[benchmark_label] = self._metrics(br, be, 0, 0)
        table = pd.DataFrame(metrics).T
        table.index.name = "portfolio"
        return {"daily_returns": strategy_returns.rename("Strategy"), "equity_curves": curves, "metrics": table, "asset_returns": asset_returns}

    backtest = run_backtest
