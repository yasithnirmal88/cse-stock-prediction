"""Causal walk-forward backtesting with fold refits and GARCH sizing."""
from __future__ import annotations
import logging
from collections.abc import Callable, Mapping
from typing import Any
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from src.models.xgboost_model import DirectionProbabilityModel
from src.signals import SignalEngine
from .performance import PerformanceAnalyzer

LOGGER=logging.getLogger(__name__)
ROUND_TRIP_COST=0.0112
ENTRY_COST=EXIT_COST=ROUND_TRIP_COST/2
TARGETS={"target_direction","target_direction_aggressive","target_return"}
PRICE_COLS={"open","high","low","close","adj close","volume","date","datetime","timestamp"}

class WalkForwardBacktester:
    """Fold-wise evaluation with a fresh model, SignalEngine and causal GARCH per fold."""
    def __init__(self,config:dict[str,Any],analyzer:PerformanceAnalyzer,signal_engine_factory:Callable[...,Any]|None=None)->None:
        self.config,self.analyzer=config,analyzer; self.signal_engine_factory=signal_engine_factory or SignalEngine
        wf,bt=config.get("walk_forward",{}),config.get("backtest",{})
        self.window_type=str(wf.get("window_type","expanding")).lower(); self.min_train_days=int(wf.get("min_train_days",504))
        self.test_days=int(wf.get("test_days",63)); self.step_days=int(wf.get("step_days",21)); self.rolling_train_days=int(wf.get("rolling_train_days",self.min_train_days))
        self.periods_per_year=int(wf.get("periods_per_year",252)); self.seed=int(wf.get("random_seed",42))
        self.initial_capital=float(wf.get("initial_capital",bt.get("initial_capital",1_000_000)))
        self.target_col=str(wf.get("target_column","target_direction")); self.model_type=str(wf.get("model_type","xgboost"))
        self.threshold=float(wf.get("probability_threshold",.55)); self.max_position_pct=float(wf.get("max_position_pct",.10))
        if self.window_type not in {"expanding","rolling"}: raise ValueError("window_type must be expanding or rolling")
        if min(self.min_train_days,self.test_days,self.step_days,self.periods_per_year)<1 or self.initial_capital<=0: raise ValueError("Invalid walk-forward config")
        np.random.seed(self.seed)

    @staticmethod
    def _dated(frame:pd.DataFrame)->pd.DataFrame:
        out=frame.copy()
        if not isinstance(out.index,pd.DatetimeIndex):
            col=next((c for c in ("Date","date","Datetime","datetime") if c in out),None)
            if col is None: raise ValueError("Frame requires a date index or date column")
            out.index=pd.to_datetime(out.pop(col),errors="coerce",utc=True).tz_localize(None)
        else: out.index=pd.to_datetime(out.index,errors="coerce",utc=True).tz_localize(None)
        out.index.name="Date"; out=out.loc[~out.index.isna()].sort_index()
        return out.loc[~out.index.duplicated(keep="last")]

    def _features(self,frames:Mapping[str,pd.DataFrame])->list[str]:
        common=set.intersection(*(set(f.columns) for f in frames.values())); explicit=self.config.get("walk_forward",{}).get("feature_columns")
        if explicit: return [str(c) for c in explicit if c in common]
        first=next(iter(frames.values()))
        return sorted(c for c in common if c not in TARGETS and c.lower() not in PRICE_COLS and pd.api.types.is_numeric_dtype(first[c]))

    def _make_model(self)->DirectionProbabilityModel:
        return DirectionProbabilityModel(model_type=self.model_type,random_state=self.seed,probability_threshold=self.threshold)

    def _make_engine(self,history:dict[str,pd.DataFrame],model:DirectionProbabilityModel)->Any:
        try: return self.signal_engine_factory(price_data=history,model=model,capital=self.initial_capital,probability_threshold=self.threshold,max_position_pct=self.max_position_pct)
        except TypeError: return self.signal_engine_factory(price_data=history,model=model)


    def run(self,data_dict:dict[str,pd.DataFrame])->dict[str,Any]:
        """Run full walk-forward and return portfolio, fold, prediction and sizing results."""
        if not data_dict: raise ValueError("data_dict must not be empty")
        raw={str(k):self._dated(v) for k,v in data_dict.items() if isinstance(v,pd.DataFrame) and not v.empty}
        target=self.target_col if any(self.target_col in f for f in raw.values()) else "target_direction_aggressive"
        frames={s:f for s,f in raw.items() if target in f and any(c in f for c in ("Close","Adj Close","close"))}
        if not frames: raise ValueError(f"No usable frames with {target}")
        features=self._features(frames)
        if not features: raise ValueError("No usable common numeric features")
        dates=pd.DatetimeIndex(sorted(set().union(*(set(f.index) for f in frames.values()))))
        predictions=[]; sizes=[]; fold_metrics=[]; importances=[]
        for fold,start in enumerate(tqdm(range(self.min_train_days,len(dates),self.step_days),desc="Walk-forward folds",unit="fold")):
            oos=dates[start:min(start+self.test_days,len(dates))]
            train_dates=dates[:start] if self.window_type=="expanding" else dates[max(0,start-self.rolling_train_days):start]
            cutoff=train_dates[-1]-pd.Timedelta(nanoseconds=1)
            xs=[]; ys=[]
            for frame in frames.values():
                tr=frame.loc[:cutoff]; x=tr.reindex(columns=features).apply(pd.to_numeric,errors="coerce"); y=pd.to_numeric(tr[target],errors="coerce")
                valid=x.replace([np.inf,-np.inf],np.nan).notna().all(axis=1)&y.isin([0,1])
                if valid.any(): xs.append(x.loc[valid]); ys.append(y.loc[valid].astype("int8"))
            if not xs: LOGGER.warning("Skipping fold %d: no training rows",fold); continue
            model=self._make_model()
            try: model.train(pd.concat(xs),pd.concat(ys))
            except (ValueError,RuntimeError) as exc: LOGGER.warning("Skipping fold %d: %s",fold,exc); continue
            imp=getattr(model,"feature_importance_",None)
            if isinstance(imp,pd.DataFrame) and {"feature","importance"}<=set(imp):
                block=imp.copy(); block["fold"]=fold; importances.extend(block.to_dict("records"))
            engine=self._make_engine({s:f.loc[:cutoff].copy() for s,f in frames.items()},model)
            fold_start=len(predictions)
            for date in oos:
                asof={s:f.loc[:date].copy() for s,f in frames.items() if not f.loc[:date].empty}
                try:
                    engine.price_data=asof
                    signals=engine.generate_combined_signals(date=date.strftime("%Y-%m-%d"))
                except (ValueError,RuntimeError,ImportError,KeyError) as exc:
                    LOGGER.warning("No signal at %s: %s",date.date(),exc); continue
                for _,row in signals.iterrows():
                    symbol=str(row.get("symbol",""))
                    if symbol not in frames or date not in frames[symbol].index: continue
                    frame=frames[symbol]; rawrow=frame.loc[date]; rawrow=rawrow.iloc[-1] if isinstance(rawrow,pd.DataFrame) else rawrow
                    close_col=next(c for c in ("Close","Adj Close","close") if c in frame)
                    close=pd.to_numeric(rawrow[close_col],errors="coerce")
                    forward=pd.to_numeric(rawrow.get("target_return",np.nan),errors="coerce")
                    if pd.isna(forward):
                        close_s=pd.to_numeric(frame[close_col],errors="coerce"); forward=close_s.shift(-1).div(close_s).sub(1).get(date,np.nan)
                    size=float(row.get("suggested_position_size",0) or 0); signal=float(row.get("signal",0) or 0)
                    item={"date":date,"symbol":symbol,"signal":signal,"probability":row.get("probability",np.nan),"actual":rawrow.get(target,np.nan),"forward_return":forward,"close":close,"position_size":size,"fold":fold}
                    predictions.append(item); sizes.append({"date":date,"symbol":symbol,"position_size":size,"signal":signal,"fold":fold})
            if len(predictions)>fold_start:
                fold_frame=pd.DataFrame(predictions[fold_start:]); fold_metrics.append({"fold":fold,"start":oos[0],"end":oos[-1],"observations":len(fold_frame)})
        if not predictions: raise RuntimeError("Walk-forward produced no out-of-sample predictions")
        pred=pd.DataFrame(predictions); returns=self._portfolio_returns(pred); equity=self.initial_capital*(1+returns).cumprod()
        curve=pd.DataFrame({"Strategy":equity}); positions=pd.DataFrame(sizes)
        for entry in fold_metrics:
            part=pred.loc[pd.to_numeric(pred["fold"],errors="coerce")==entry["fold"]]
            fold_dates=pd.to_datetime(part["date"],errors="coerce").dropna().drop_duplicates().sort_values()
            fold_returns=returns.loc[returns.index.isin(fold_dates)]
            fold_equity=self.initial_capital*(1+fold_returns).cumprod()
            summary=self.analyzer.calculate_metrics(part,portfolio_returns=fold_returns,equity=fold_equity)
            entry.update({"total_return":summary["total_return"],"sharpe_ratio":summary["sharpe_ratio"],"max_drawdown":summary["max_drawdown"],"accuracy":summary["accuracy"],"number_of_trades":summary["number_of_trades"]})

        overall=self.analyzer.calculate_metrics(pred,portfolio_returns=returns,equity=equity,positions=positions.set_index("date")["position_size"])
        for entry in fold_metrics:
            part=pred.loc[pd.to_numeric(pred["fold"],errors="coerce")==entry["fold"]]
            fold_dates=pd.to_datetime(part["date"],errors="coerce").dropna().drop_duplicates().sort_values()
            fold_returns=returns.loc[returns.index.isin(fold_dates)]
            fold_equity=self.initial_capital*(1+fold_returns).cumprod()
            summary=self.analyzer.calculate_metrics(part,portfolio_returns=fold_returns,equity=fold_equity)
            entry.update({"total_return":summary["total_return"],"sharpe_ratio":summary["sharpe_ratio"],"max_drawdown":summary["max_drawdown"],"accuracy":summary["accuracy"],"number_of_trades":summary["number_of_trades"]})

        return {"equity_curve":curve,"equity_curves":curve,"portfolio_returns":returns,"fold_metrics":pd.DataFrame(fold_metrics),"all_predictions":pred,"predictions":pred,"position_sizes":positions,"positions":positions,"feature_importance":pd.DataFrame(importances),"metrics":overall,"round_trip_cost":ROUND_TRIP_COST}

    def _portfolio_returns(self,pred:pd.DataFrame)->pd.Series:
        frame=pred.copy(); frame["date"]=pd.to_datetime(frame["date"]); frame["weight"]=frame.position_size.fillna(0)*frame.signal.fillna(0)/self.initial_capital
        weights=frame.pivot_table(index="date",columns="symbol",values="weight",aggfunc="sum").fillna(0).sort_index(); returns=frame.pivot_table(index="date",columns="symbol",values="forward_return",aggfunc="last").reindex(weights.index).fillna(0)
        gross=weights.abs().sum(axis=1).clip(lower=1); held=weights.div(gross,axis=0).shift(1).fillna(0); turnover=weights.div(gross,axis=0).diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1).clip(upper=1)*0+weights.abs().sum(axis=1))
        out=held.mul(returns.reindex(columns=held.columns)).sum(axis=1)-turnover*ENTRY_COST
        return out.replace([np.inf,-np.inf],np.nan).fillna(0).rename("strategy_return")

