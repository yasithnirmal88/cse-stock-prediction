"""Run full walk-forward from local processed features."""
from __future__ import annotations
import argparse, logging, re, sys
from pathlib import Path
import pandas as pd, yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data_loader import load_config, resolve_data_path
from src.backtester.walk_forward import WalkForwardBacktester, WalkForwardConfig

def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=ROOT/"config.yaml"); p.add_argument("--data-dir",type=Path); p.add_argument("--universe",type=Path,default=ROOT/"data"/"universe.yaml"); p.add_argument("--output-dir",type=Path,default=ROOT/"reports"); a=p.parse_args()
    logging.basicConfig(level=logging.INFO); cfg=load_config(a.config); directory=a.data_dir or resolve_data_path(cfg,"processed_dir"); items=(yaml.safe_load(a.universe.read_text(encoding="utf-8")) or {}).get("stocks",[]); data={}
    for item in items:
        symbol=str(item.get("symbol","")); stem=re.sub(r"[^A-Za-z0-9._-]","_",symbol); paths=[directory/f"{stem}_features.csv",directory/f"{stem}_features.parquet",directory/f"{stem}_features.feather"]; file=next((x for x in paths if x.exists()),None)
        if file is None: continue
        frame=pd.read_csv(file) if file.suffix==".csv" else (pd.read_parquet(file) if file.suffix==".parquet" else pd.read_feather(file)); dc=next((c for c in ("Date","date","Datetime") if c in frame),None)
        if dc is None: continue
        frame[dc]=pd.to_datetime(frame[dc],errors="coerce"); data[symbol]=frame.dropna(subset=[dc]).set_index(dc).sort_index()
    if not data: raise SystemExit(f"No processed feature data under {directory}")
    excluded={"Open","High","Low","Close","Volume","target_direction","target_return","target_direction_aggressive"}; features=[c for c in data[next(iter(data))].select_dtypes(include="number").columns if c not in excluded]
    params={k:v for k,v in cfg.get("walk_forward",{}).items() if k in WalkForwardConfig.__dataclass_fields__}; result=WalkForwardBacktester(WalkForwardConfig(**params)).run(data,features,output_dir=str(a.output_dir)); logging.info("Complete: %s",result["metrics"])
if __name__=="__main__": main()
