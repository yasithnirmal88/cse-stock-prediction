"""Run Phase 3 walk-forward backtest on processed feature CSV files."""
from __future__ import annotations
import logging
from pathlib import Path
import sys
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.backtester import WalkForwardBacktester,PerformanceAnalyzer
from src.features import FeatureEngineer
from src.data_loader import load_config

def main()->None:
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config=load_config(ROOT/"config.yaml"); processed=ROOT/config.get("data",{}).get("processed_dir","data/processed")
    frames={}
    for path in sorted(processed.glob("*.csv")):
        frame=pd.read_csv(path)
        if frame.empty: continue
        dated=FeatureEngineer._dated(frame)
        symbol=path.stem.split("_")[0]
        if {"Open","High","Low","Close","Volume"}<=set(dated.columns):
            frames[symbol]=FeatureEngineer(config).create_features(dated,symbol=symbol)
    if not frames: raise FileNotFoundError(f"No usable OHLCV CSV files in {processed}")
    analyzer=PerformanceAnalyzer(int(config.get("walk_forward",{}).get("periods_per_year",252)))
    results=WalkForwardBacktester(config,analyzer).run(frames)
    analyzer.generate_reports(results,ROOT/"reports")
    logging.info("Walk-forward metrics: %s",results["metrics"])
if __name__=="__main__": main()
