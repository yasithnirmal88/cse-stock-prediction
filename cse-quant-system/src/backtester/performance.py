"""Metrics and interactive report generation for walk-forward backtests."""
from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score
LOGGER=logging.getLogger(__name__)

def _write_minimal_pdf(path: Path, lines: list[str]) -> None:
    """Write a readable one-page text PDF without optional renderers."""
    import textwrap
    wrapped = [part for line in lines for part in textwrap.wrap(str(line), width=92) or [""]]
    wrapped = wrapped[:52]
    commands = ["BT", "/F1 9 Tf", "42 800 Td"]
    for i, line in enumerate(wrapped):
        safe = line.encode("ascii", "replace").decode("ascii").replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if i:
            commands.append("0 -14 Td")
        commands.append(f"({safe}) Tj")
    commands.append("ET")
    stream = "\n".join(commands).encode("ascii")
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>", b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>", b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"]
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(pdf)); pdf.extend(f"{number} 0 obj\n".encode()); pdf.extend(obj); pdf.extend(b"\nendobj\n")
    xref = len(pdf); pdf.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]: pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(pdf)

class PerformanceAnalyzer:
    """Calculate classification/financial metrics and save HTML/PDF reports."""
    def __init__(self, periods_per_year:int=252, risk_free_rate:float=0.) -> None:
        self.periods_per_year=int(periods_per_year); self.risk_free_rate=float(risk_free_rate)
    def calculate_metrics(self, predictions:pd.DataFrame, portfolio_returns:pd.Series|None=None, equity:pd.Series|None=None, positions:pd.Series|None=None, transaction_costs:pd.Series|None=None) -> dict[str,Any]:
        y=pd.to_numeric(predictions.get("actual",pd.Series(dtype=float)),errors="coerce"); p=pd.to_numeric(predictions.get("probability",pd.Series(dtype=float)),errors="coerce").clip(0,1); valid=y.notna()&p.notna(); y=y[valid].astype(int); p=p[valid]; yh=(p>=.5).astype(int)
        r=pd.to_numeric(portfolio_returns if portfolio_returns is not None else pd.Series(dtype=float),errors="coerce").replace([np.inf,-np.inf],np.nan).dropna(); eq=pd.to_numeric(equity,errors="coerce").dropna() if equity is not None else (1+r).cumprod(); dd=eq/eq.cummax()-1 if len(eq) else pd.Series(dtype=float)
        sd=float(r.std()) if len(r)>1 else 0.; down=float(np.sqrt(np.mean(np.minimum(r.to_numpy(),0)**2))) if len(r) else 0.; total=float(eq.iloc[-1]/eq.iloc[0]-1) if len(eq)>1 and eq.iloc[0] else 0.; cagr=float((eq.iloc[-1]/eq.iloc[0])**(self.periods_per_year/max(len(r),1))-1) if len(eq)>1 and eq.iloc[0]>0 and eq.iloc[-1]>0 else 0.; w,l=r[r>0],r[r<0]
        return {"observations":len(y),"accuracy":float(accuracy_score(y,yh)) if len(y) else np.nan,"precision":float(precision_score(y,yh,zero_division=0)) if len(y) else np.nan,"recall":float(recall_score(y,yh,zero_division=0)) if len(y) else np.nan,"f1":float(f1_score(y,yh,zero_division=0)) if len(y) else np.nan,"brier_score":float(brier_score_loss(y,p)) if len(y) else np.nan,"total_return":total,"cagr":cagr,"annualized_volatility":sd*np.sqrt(self.periods_per_year),"sharpe_ratio":float((r.mean()-self.risk_free_rate/self.periods_per_year)/sd*np.sqrt(self.periods_per_year)) if sd else 0.,"sortino_ratio":float((r.mean()-self.risk_free_rate/self.periods_per_year)/down*np.sqrt(self.periods_per_year)) if down else 0.,"max_drawdown":float(dd.min()) if len(dd) else 0.,"calmar_ratio":cagr/abs(float(dd.min())) if len(dd) and dd.min()<0 else 0.,"profit_factor":float(w.sum()/-l.sum()) if l.sum()<0 else (float("inf") if w.sum()>0 else 0.),"win_rate":len(w)/(len(w)+len(l)) if len(w)+len(l) else 0.,"average_win":float(w.mean()) if len(w) else 0.,"average_loss":float(l.mean()) if len(l) else 0.,"number_of_trades":int((r!=0).sum()),"exposure":float(pd.to_numeric(positions,errors="coerce").fillna(0).ne(0).mean()) if positions is not None and len(positions) else 0.,"turnover":float(pd.to_numeric(transaction_costs,errors="coerce").fillna(0).sum()) if transaction_costs is not None else 0.}
    def generate_reports(self,result:dict[str,Any],output_dir:str|Path="reports")->dict[str,Path]:
        """Generate interactive charts, metrics CSV/JSON, HTML and PDF."""
        out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); curve=result.get("equity_curve",result.get("equity_curves",pd.DataFrame())); pred=result.get("all_predictions",result.get("predictions",pd.DataFrame())); folds=result.get("fold_metrics",pd.DataFrame()); pos=result.get("position_sizes",result.get("positions",pd.DataFrame())); metrics=result.get("metrics",{}); imp=result.get("feature_importance",pd.DataFrame())
        (out/"walk_forward_metrics.json").write_text(json.dumps(metrics,indent=2,default=str,allow_nan=True),encoding="utf-8"); pd.DataFrame([metrics]).to_csv(out/"walk_forward_metrics.csv",index=False)
        charts=[]
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            from plotly.offline import plot
        except ImportError:
            go=make_subplots=plot=None
            LOGGER.warning("Plotly unavailable; generating a static HTML report")
        if go is not None:
            if not curve.empty:
                fig=go.Figure()
                for col in curve: fig.add_scatter(x=curve.index,y=curve[col],name=str(col))
                fig.update_layout(title="Equity curve vs benchmark"); charts.append(("Equity",fig))
                fig=go.Figure()
                for col in curve: fig.add_scatter(x=curve.index,y=curve[col]/curve[col].cummax()-1,name=str(col))
                fig.update_layout(title="Drawdown"); charts.append(("Drawdown",fig))
                if isinstance(curve.index,pd.DatetimeIndex):
                    ret=curve["Strategy"].pct_change().dropna(); mon=((1+ret).resample("ME").prod()-1).to_frame("r"); mon["year"]=mon.index.year; mon["month"]=mon.index.month; grid=mon.pivot(index="year",columns="month",values="r")
                    charts.append(("Monthly heatmap",go.Figure(go.Heatmap(z=grid.values,x=grid.columns,y=grid.index,colorscale="RdYlGn",zmid=0))))
                    yr=(1+ret).groupby(ret.index.year).prod()-1; charts.append(("Yearly heatmap",go.Figure(go.Heatmap(z=[yr.values],x=yr.index,y=["Strategy"],colorscale="RdYlGn",zmid=0))))
            if not pred.empty and {"probability","actual"}<=set(pred):
                fig=make_subplots(rows=1,cols=2,subplot_titles=("Prediction vs actual","Calibration")); q=pred.dropna(subset=["probability","actual"]); fig.add_scatter(x=q.probability,y=q.actual,mode="markers",name="OOS",row=1,col=1); q=q.copy(); q["bin"]=pd.cut(q.probability,np.linspace(0,1,11),include_lowest=True); cal=q.groupby("bin",observed=True).agg(prob=("probability","mean"),actual=("actual","mean")); fig.add_scatter(x=cal.prob,y=cal.actual,mode="lines+markers",name="Calibration",row=1,col=2); charts.append(("Calibration",fig))
            if not pos.empty and "position_size" in pos: charts.append(("GARCH sizes",go.Figure(go.Histogram(x=pd.to_numeric(pos.position_size,errors="coerce").abs(),nbinsx=30))) )
            if not folds.empty:
                fig=go.Figure(); fig.add_table(header={"values":list(folds.columns)},cells={"values":[folds[c] for c in folds]}); charts.append(("Fold table",fig))
            if not imp.empty and {"feature","importance"}<=set(imp):
                fig=go.Figure()
                for name,g in imp.groupby("feature"): fig.add_box(y=g.importance,name=str(name))
                charts.append(("Feature stability",fig))
        html=out/"walk_forward_report.html"
        import html as html_module
        def table(frame:pd.DataFrame, max_rows:int=100)->str:
            if frame.empty: return "<p class='empty'>No data available.</p>"
            return frame.head(max_rows).to_html(index=False,escape=True,border=0,classes="data-table")
        cards=[]
        for label,key,fmt in (("Total return","total_return",".2%"),("Sharpe ratio","sharpe_ratio",".2f"),("Max drawdown","max_drawdown",".2%"),("Trades","number_of_trades","d"),("OOS observations","observations","d")):
            value=metrics.get(key,float("nan"))
            try: shown=format(float(value),fmt) if fmt!="d" else format(int(value),fmt)
            except (TypeError,ValueError): shown="N/A"
            cards.append(f"<div class='card'><span>{label}</span><strong>{shown}</strong></div>")
        sections=["<section><h2>Key metrics</h2><div class='cards'>"+"".join(cards)+"</div>"+table(pd.DataFrame([metrics]))+"</section>"]
        if not curve.empty:
            equity=curve.select_dtypes(include=[np.number])
            if not equity.empty:
                strategy=equity.iloc[:,0].dropna(); values=strategy.to_numpy(dtype=float)
                if len(values)>1:
                    lo,hi=float(np.min(values)),float(np.max(values)); span=hi-lo or 1.; points=[]
                    for idx,value in enumerate(values):
                        x=idx*600/max(len(values)-1,1); y=130-(value-lo)*110/span; points.append(f"{x:.1f},{y:.1f}")
                    svg=f"<svg viewBox='0 0 600 150' role='img' aria-label='Equity curve'><polyline fill='none' stroke='#2563eb' stroke-width='2' points='{' '.join(points)}'/><text x='0' y='148'>{html_module.escape(str(strategy.index[0]))}</text><text x='500' y='148'>{html_module.escape(str(strategy.index[-1]))}</text></svg>"
                    sections.append("<section><h2>Equity curve</h2><div class='chart'>"+svg+"</div></section>")
            sections.append("<section><h2>Equity data</h2>"+table(curve.reset_index())+"</section>")
        if not folds.empty: sections.append("<section><h2>Fold summary</h2>"+table(folds)+"</section>")
        if not pred.empty:
            preview=pred.copy()
            if "date" in preview: preview["date"]=pd.to_datetime(preview["date"],errors="coerce").astype(str)
            sections.append("<section><h2>Out-of-sample predictions</h2>"+table(preview,50)+"</section>")
        if not imp.empty: sections.append("<section><h2>Feature importance</h2>"+table(imp,100)+"</section>")
        static="" if plot is not None else "<p class='note'>Static report: Plotly is unavailable; the equity chart is rendered as inline SVG.</p>"
        body="".join(sections)
        if plot is not None:
            for title,fig in charts: body+=f"<section><h2>{html_module.escape(title)}</h2>"+plot(fig,output_type="div",include_plotlyjs="cdn")+"</section>"
        document="<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>CSE Quant | Walk-forward Report</title><style>body{font:15px/1.5 system-ui,Segoe UI,sans-serif;background:#f3f6fb;color:#172033;margin:0}.wrap{max-width:1120px;margin:auto;padding:32px}header{background:#102341;color:white;padding:28px 32px;border-radius:14px;margin-bottom:22px}header h1{margin:0;font-size:28px}header p{color:#c9d7ed;margin:6px 0 0}section{background:white;border:1px solid #e2e8f0;border-radius:12px;padding:22px;margin:18px 0;overflow:auto}h2{font-size:19px;margin:0 0 16px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:18px}.card{background:#edf4ff;border-radius:9px;padding:14px}.card span{display:block;color:#54647a;font-size:12px}.card strong{font-size:22px}.data-table{border-collapse:collapse;width:100%;font-size:13px}.data-table th,.data-table td{padding:8px 10px;text-align:left;border-bottom:1px solid #e8edf4}.data-table th{background:#f5f8fc;color:#3d4c62}.chart svg{width:100%;max-height:260px}.note,.empty{color:#66758b;font-size:13px}footer{color:#718096;font-size:12px;padding:12px}</style></head><body><div class='wrap'><header><h1>CSE Quant System</h1><p>Walk-forward backtest report · generated from out-of-sample results</p></header>"+static+body+"<footer>Metrics are calculated by PerformanceAnalyzer; this report does not imply future performance.</footer></div></body></html>"
        html.write_text(document,encoding="utf-8")
        pdf=out/"walk_forward_report.pdf"
        try:
            from weasyprint import HTML
            HTML(string=document,base_url=str(out)).write_pdf(str(pdf))
        except Exception as exc:
            LOGGER.warning("WeasyPrint failed (%s); trying available PDF fallback",exc)
            lines=["CSE QUANT SYSTEM - WALK-FORWARD REPORT","", "KEY METRICS"]
            for key,value in metrics.items(): lines.append(f"{key.replace('_',' ').title()}: {value}")
            lines.extend(["", "EQUITY SUMMARY"])
            if not curve.empty and "Strategy" in curve:
                s=pd.to_numeric(curve["Strategy"],errors="coerce").dropna()
                if len(s): lines.extend([f"Points: {len(s)}",f"Start equity: {s.iloc[0]:,.2f}",f"End equity: {s.iloc[-1]:,.2f}",f"Low / high: {s.min():,.2f} / {s.max():,.2f}",f"Dates: {s.index[0]} to {s.index[-1]}"])
            lines.extend(["", "FOLD SUMMARY"])
            if not folds.empty:
                lines.append(" | ".join(map(str,folds.columns)))
                lines.extend(" | ".join(map(str,row)) for row in folds.head(12).itertuples(index=False,name=None))
            else: lines.append("No fold summary available.")
            _write_minimal_pdf(pdf,lines)
        return {"html":html,"pdf":pdf,"metrics_csv":out/"walk_forward_metrics.csv","metrics_json":out/"walk_forward_metrics.json"}
