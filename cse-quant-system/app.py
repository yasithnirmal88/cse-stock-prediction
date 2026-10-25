"""Interactive dashboard for viewing saved CSE walk-forward results."""
"""Interactive dashboard for viewing saved CSE walk-forward results."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.dashboard_utils import (load_equity_curve,load_fold_metrics,load_metrics,load_predictions,load_run_metadata,load_signals,model_status)
REPORTS=ROOT/"reports"
st.set_page_config(page_title="CSE Quant Dashboard",page_icon="ðŸ“ˆ",layout="wide")
st.markdown("<style>.block-container{padding-top:1.5rem;max-width:1480px}[data-testid='stMetric']{background:#f4f7fb;padding:14px;border-radius:10px;border:1px solid #e4e9f2}h1,h2{color:#142b4a}</style>",unsafe_allow_html=True)
def pct(value):
    try:
        number=float(value)
        return f"{number:.2%}" if pd.notna(number) else "â€”"
    except (TypeError,ValueError): return "â€”"
def table(frame):
    frame=frame.copy()
    for col in frame.columns:
        name=str(col).lower()
        if name in {"date","start_date","end_date","fold_start","fold_end"}:
            dates=pd.to_datetime(frame[col],errors="coerce")
            if dates.notna().any(): frame[col]=dates.dt.strftime("%Y-%m-%d").fillna("â€”")
        if any(x in name for x in ("return","drawdown","accuracy","precision","recall","win_rate","volatility","exposure")):
            values=pd.to_numeric(frame[col],errors="coerce")
            if values.notna().any(): frame[col]=values.map(pct)
    return frame
with st.sidebar:
    st.title("CSE Quant"); st.caption("Walk-forward research dashboard")
    st.subheader("View")
    symbols=[x.strip() for x in st.text_input("Symbols","JKH.N0000, COMB.N0000, SAMP.N0000").split(",") if x.strip()]
    st.radio("Jump to section",["Overview","Latest Signals","Equity & Drawdown","Walk-forward Summary","Prediction vs Actual","Data Status"])
    st.divider(); st.subheader("Refresh Signals")
    st.caption("Load is instant. Optional smoke test uses synthetic or local cache and never fetches data.")
    if st.button("Load Latest Results",use_container_width=True,help="Reload existing reports/latest artifacts only"):
        st.session_state["refresh_status"]=("success","Latest saved results reloaded."); st.rerun()
    quick_mode=st.selectbox("Quick test data",["synthetic","auto (local cache)"])
    if st.button("Run Quick Smoke Test",type="primary",use_container_width=True,help="Explicitly runs the limited smoke-test script"):
        try:
            mode="synthetic" if quick_mode=="synthetic" else "auto"
            result=subprocess.run([sys.executable,str(ROOT/"scripts"/"smoke_test_walk_forward.py"),"--data-mode",mode],cwd=ROOT,capture_output=True,text=True,timeout=300,check=False)
            if result.returncode==0: st.session_state["refresh_status"]=("success","Smoke test passed; latest exports refreshed.")
            else: st.session_state["refresh_status"]=("error",f"Smoke test failed ({result.returncode}): {(result.stderr or result.stdout)[-1500:]}")
        except subprocess.TimeoutExpired: st.session_state["refresh_status"]=("error","Smoke test exceeded its 5-minute timeout.")
        except OSError as exc: st.session_state["refresh_status"]=("error",f"Could not start smoke test: {exc}")
        st.rerun()
    if "refresh_status" in st.session_state:
        level,message=st.session_state["refresh_status"]; getattr(st,level)(message)
    st.caption("No analysis runs automatically on dashboard startup.")
metrics=load_metrics(REPORTS); equity=load_equity_curve(REPORTS); folds=load_fold_metrics(REPORTS)
predictions=load_predictions(REPORTS); signals=load_signals(REPORTS); status=model_status(REPORTS); metadata=load_run_metadata(REPORTS)

last_update=metadata.get("generated_at")
try: last_update=pd.Timestamp(last_update).strftime("%Y-%m-%d %H:%M UTC") if last_update else "Not available"
except (TypeError,ValueError): last_update="Not available"
st.title("CSE Quantitative Research Dashboard")
st.caption(f"Walk-forward performance, portfolio signals, and model monitoring · Last updated: **{last_update}**")
st.header("Overview")
st.caption("Headline out-of-sample performance metrics from the latest saved run.")
if metrics:
    cards=[("Total return",metrics.get("total_return"),"percent"),("CAGR",metrics.get("cagr"),"percent"),("Sharpe ratio",metrics.get("sharpe_ratio"),"number"),("Max drawdown",metrics.get("max_drawdown"),"percent"),("Trades",metrics.get("number_of_trades"),"number"),("Accuracy",metrics.get("accuracy"),"percent")]
    for col,(label,value,kind) in zip(st.columns(6),cards):
        shown=pct(value) if kind=="percent" else (f"{float(value):.2f}" if kind=="number" and value is not None and label=="Sharpe ratio" else str(value if value is not None else "—"))
        delta="normal" if label in {"Total return","CAGR"} and value is not None and float(value)>=0 else ("inverse" if label in {"Total return","CAGR","Max drawdown"} else "off")
        col.metric(label,shown,delta_color=delta)
    with st.expander("Additional metrics"): st.json({k:metrics[k] for k in ("observations","precision","recall","f1","brier_score","annualized_volatility","win_rate","profit_factor","exposure","turnover") if k in metrics})
else: st.info("No walk-forward metrics are available. Load results or run the optional smoke test.")
st.header("Latest Signals")
st.caption("Most recent out-of-sample rows with signal probability and position sizing where available.")
if not signals.empty:
    shown=signals.copy()
    if "symbol" in shown and symbols: shown=shown[shown.symbol.astype(str).isin(symbols)]
    cols=[c for c in ("date","symbol","signal","probability","position_size","suggested_position_size","close","fold") if c in shown]
    st.dataframe(table(shown[cols] if cols else shown),use_container_width=True,hide_index=True)
else: st.info("No saved signal rows are available.")
st.header("Equity & Drawdown")
st.caption("Cumulative portfolio value and peak-to-trough drawdown over the out-of-sample period.")
if not equity.empty:
    chart=equity.select_dtypes(include="number")
    if not chart.empty:
        try:
            import plotly.express as px
            frame=chart.copy(); frame.index.name=frame.index.name or "date"
            st.plotly_chart(px.line(frame.reset_index(),x=frame.index.name,y=list(chart.columns),title="Portfolio equity"),use_container_width=True)
        except ImportError: st.line_chart(chart,use_container_width=True)
        key="Strategy" if "Strategy" in chart else chart.columns[0]
        st.area_chart((chart[key]/chart[key].cummax()-1).rename("Drawdown"),use_container_width=True)
    else: st.info("Equity artifact contains no numeric series.")
else: st.info("No equity curve is available.")
st.header("Walk-forward Summary")
st.caption("Per-fold portfolio performance and prediction-quality results.")
if not folds.empty: st.dataframe(table(folds),use_container_width=True,hide_index=True)
else: st.info("No fold metrics are available.")
st.header("Prediction vs Actual")
st.caption("Compare model probabilities with realized outcomes; expand the table for observations.")
if not predictions.empty:
    if {"probability","actual"}.issubset(predictions.columns):
        valid=predictions.dropna(subset=["probability","actual"])
        if not valid.empty:
            try:
                import plotly.express as px
                st.plotly_chart(px.scatter(valid,x="probability",y="actual",color="symbol" if "symbol" in valid else None,hover_data=[c for c in ("date","fold") if c in valid],opacity=.7),use_container_width=True)
            except ImportError: st.scatter_chart(valid,x="probability",y="actual",color="symbol" if "symbol" in valid else None)
    with st.expander("Prediction records"): st.dataframe(table(predictions),use_container_width=True,hide_index=True)
else: st.info("No prediction observations are available.")
st.header("Data Status & Model Health")
st.caption("Data provenance, coverage, and availability of saved dashboard artifacts.")
source=metadata.get("data_source","Unknown (older run metadata)")
source_label="Synthetic" if "synthetic" in str(source).lower() else ("Real / local cache" if source!="Unknown (older run metadata)" else source)
items=[("Data source",source_label),("Date range",f"{str(status.get('data_start') or '—')[:10]} – {str(status.get('data_end') or '—')[:10]}"),("Symbols",str(metadata.get("symbol_count","—"))),("Folds",str(status.get("fold_count") or "—")),("Predictions",str(status.get("prediction_count") or "—"))]
for col,(label,value) in zip(st.columns(5),items): col.metric(label,value)
st.caption(f"Metrics: {'available' if status['metrics_available'] else 'missing'} · Signals: {'available' if status['signals_available'] else 'missing'} · Universe filter: {', '.join(symbols) or 'None'}")
if status.get("metrics_path"): st.caption(f"Metrics source: `{status['metrics_path']}`")
