"""Streamlit dashboard for CSE Quant System matching the Market Forecast Lab design."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard_utils import (
    get_symbol_summary,
    load_all_signals,
    load_equity_curve,
    load_fold_metrics,
    load_metrics,
    load_predictions,
    load_run_metadata,
    load_signals,
    load_universe_names,
    model_status,
)

REPORTS = ROOT / "reports"

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Market forecast lab ┬╖ CSE Quant",
    page_icon="≡ƒôê",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Theme management
# ---------------------------------------------------------------------------
if "dark_theme" not in st.session_state:
    st.session_state["dark_theme"] = False

is_dark = st.session_state["dark_theme"]

# CSS design system mirroring Market forecast lab.html exactly
# Variables: --bg, --panel, --ink, --mut, --line, --up, --dn, --acc, --grid
css_vars = """
:root {
  --bg: #0b0f14;
  --panel: #131920;
  --ink: #e8edf3;
  --mut: #8896a7;
  --line: #232d3a;
  --up: #2bd47d;
  --dn: #ff5468;
  --acc: #7b94ff;
  --grid: #1b232d;
}
""" if is_dark else """
:root {
  --bg: #f4f6f8;
  --panel: #ffffff;
  --ink: #101828;
  --mut: #667085;
  --line: #e3e7ec;
  --up: #0f9d58;
  --dn: #d93040;
  --acc: #3856d6;
  --grid: #eef0f3;
}
"""

style_block = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

{css_vars}

html, body, [data-testid="stAppViewContainer"] {{
    background-color: var(--bg) !important;
    color: var(--ink) !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 14px;
    line-height: 1.45;
}}

.block-container {{
    padding-top: 1rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1520px !important;
}}

/* Header bar */
.mfl-header {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    justify-content: space-between;
    align-items: center;
    padding: 14px 20px;
    border: 1px solid var(--line);
    border-radius: 10px;
    background: var(--panel);
    margin-bottom: 12px;
}}
.mfl-title {{
    font-size: 19px;
    font-weight: 700;
    margin: 0;
    color: var(--ink);
}}
.mfl-sub {{
    margin: 2px 0 0;
    color: var(--mut);
    font-size: 12px;
}}
.mfl-live {{
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--mut);
    font-size: 12px;
    background: var(--grid);
    padding: 5px 12px;
    border-radius: 20px;
    border: 1px solid var(--line);
}}
.mfl-live i {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--up);
    display: inline-block;
    box-shadow: 0 0 6px var(--up);
}}

/* Cards and KPI Grid */
.mfl-card {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 14px;
    min-width: 0;
    margin-bottom: 12px;
}}
.mfl-card h2 {{
    font-size: 14px;
    margin: 0 0 10px;
    font-weight: 650;
    color: var(--ink);
}}
.mfl-kpis {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin-bottom: 12px;
}}
.mfl-kpi-card {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 14px;
    min-width: 0;
}}
.mfl-kpi-card span.lbl {{
    display: block;
    font-size: 12px;
    color: var(--mut);
}}
.mfl-kpi-card b.val {{
    display: block;
    font-size: 22px;
    line-height: 1.2;
    margin-top: 2px;
    color: var(--ink);
    font-variant-numeric: tabular-nums;
}}
.mfl-kpi-card span.sub {{
    display: block;
    font-size: 12px;
    margin-top: 2px;
}}

/* Chip styles */
.chip {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-weight: 700;
    font-size: 13px;
}}
.chip.up {{
    background: color-mix(in srgb, var(--up) 16%, transparent);
    color: var(--up);
}}
.chip.dn {{
    background: color-mix(in srgb, var(--dn) 16%, transparent);
    color: var(--dn);
}}
.chip.mut {{
    background: var(--grid);
    color: var(--mut);
}}

.up {{ color: var(--up); }}
.dn {{ color: var(--dn); }}
.mut {{ color: var(--mut); }}

/* Mini stats grid */
.mfl-stats {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-bottom: 8px;
}}
.mfl-stats div {{
    background: var(--grid);
    border-radius: 8px;
    padding: 8px;
}}
.mfl-stats div b {{
    display: block;
    font-size: 15px;
    color: var(--ink);
}}
.mfl-stats div span {{
    font-size: 11px;
    color: var(--mut);
}}

/* Feature bars (What is moving the forecast) */
.fb {{
    display: grid;
    grid-template-columns: 140px 1fr;
    gap: 8px;
    align-items: center;
    margin: 6px 0;
    font-size: 12px;
}}
.fb span {{
    color: var(--ink);
}}
.fb div.t {{
    position: relative;
    height: 10px;
    background: var(--grid);
    border-radius: 3px;
}}
.fb div.t::before {{
    content: "";
    position: absolute;
    left: 50%;
    top: -2px;
    bottom: -2px;
    width: 1px;
    background: var(--mut);
}}
.fb div.t i {{
    position: absolute;
    top: 0;
    bottom: 0;
    border-radius: 3px;
}}

/* Confusion matrix */
.cm {{
    display: grid;
    grid-template-columns: auto 1fr 1fr;
    gap: 4px;
    margin-top: 10px;
    text-align: center;
    font-size: 12px;
    align-items: center;
}}
.cm div.c {{
    padding: 8px 4px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 14px;
}}

/* Sector heatmap */
.mfl-heat {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    grid-auto-flow: dense;
    gap: 6px;
}}
.mfl-heat div {{
    border-radius: 6px;
    padding: 10px;
    min-height: 60px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    font-size: 12px;
}}
.mfl-heat div span {{
    color: var(--mut);
    font-weight: 500;
}}
.mfl-heat div b {{
    font-size: 15px;
    font-weight: 700;
}}

/* Styled Forecast Table */
.mfl-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}
.mfl-table th {{
    font-weight: 500;
    color: var(--mut);
    text-align: right;
    padding: 6px 8px;
    font-size: 12px;
    border-bottom: 1px solid var(--line);
}}
.mfl-table th:first-child, .mfl-table td:first-child {{
    text-align: left;
}}
.mfl-table td {{
    padding: 8px 8px;
    text-align: right;
    border-top: 1px solid var(--line);
    color: var(--ink);
}}

/* Streamlit widget overrides */
div[data-testid="stSidebar"] {{
    background-color: var(--panel) !important;
    border-right: 1px solid var(--line) !important;
}}
[data-baseweb="select"] > div {{
    background-color: var(--panel) !important;
    border-color: var(--line) !important;
    color: var(--ink) !important;
    border-radius: 8px !important;
}}
button[kind="secondary"] {{
    background-color: var(--panel) !important;
    border-color: var(--line) !important;
    color: var(--ink) !important;
    border-radius: 8px !important;
}}
</style>
"""
st.markdown(style_block, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt(v: Any, d: int = 2) -> str:
    try:
        val = float(v)
        return f"{val:,.{d}f}" if pd.notna(val) else "ΓÇö"
    except (TypeError, ValueError):
        return "ΓÇö"


def pc(v: Any) -> str:
    try:
        val = float(v)
        if pd.isna(val):
            return "ΓÇö"
        sign = "+" if val >= 0 else ""
        return f"{sign}{val * 100:.2f}%"
    except (TypeError, ValueError):
        return "ΓÇö"


def cls(v: Any) -> str:
    try:
        val = float(v)
        return "up" if val >= 0 else "dn"
    except (TypeError, ValueError):
        return "mut"


# ---------------------------------------------------------------------------
# Load workspace reports
# ---------------------------------------------------------------------------
metrics = load_metrics(REPORTS)
equity = load_equity_curve(REPORTS)
folds = load_fold_metrics(REPORTS)
predictions = load_predictions(REPORTS)
signals_all = load_all_signals(REPORTS)
status = model_status(REPORTS)
metadata = load_run_metadata(REPORTS)

# Universe of symbols
default_symbols = ["JKH.N0000", "COMB.N0000", "SAMP.N0000"]
available_symbols = []
if not signals_all.empty and "symbol" in signals_all:
    available_symbols = sorted(list(signals_all["symbol"].dropna().unique()))
if not available_symbols:
    available_symbols = default_symbols

last_update = metadata.get("generated_at")
try:
    last_update_str = pd.Timestamp(last_update).strftime("%Y-%m-%d %H:%M UTC") if last_update else "Live"
except (TypeError, ValueError):
    last_update_str = "Live"


# ---------------------------------------------------------------------------
# Sidebar: Controls & Artifact Maintenance
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ≡ƒÅ¢∩╕Å CSE Market Forecast Lab")
    st.caption("Quantitative walk-forward forecasting for Colombo equities.")
    st.divider()

    theme_val = st.toggle("≡ƒîÖ Dark Mode", value=is_dark)
    if theme_val != is_dark:
        st.session_state["dark_theme"] = theme_val
        st.rerun()

    st.markdown("**Model Architecture**")
    selected_model = st.selectbox(
        "Forecasting Model",
        ["Ensemble", "Logistic Baseline", "GARCH + Trend Gating", "LightGBM / XGBoost"],
        index=0,
    )

    st.divider()
    st.markdown("**Data & Test Operations**")
    if st.button("≡ƒöä Reload Latest Results", use_container_width=True):
        st.session_state["refresh_status"] = ("success", "Refreshed lab artifacts from reports/latest/.")
        st.rerun()

    smoke_mode = st.selectbox("Quick Test Dataset", ["synthetic", "auto (local cache)"])
    if st.button("ΓÜí Run Smoke-Test Evaluation", type="primary", use_container_width=True):
        try:
            mode = "synthetic" if smoke_mode == "synthetic" else "auto"
            res = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "smoke_test_walk_forward.py"), "--data-mode", mode],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            if res.returncode == 0:
                st.session_state["refresh_status"] = ("success", "Smoke test completed successfully!")
            else:
                st.session_state["refresh_status"] = ("error", f"Smoke test failed ({res.returncode}): {(res.stderr or res.stdout)[-1000:]}")
        except Exception as e:
            st.session_state["refresh_status"] = ("error", f"Execution error: {e}")
        st.rerun()

    if "refresh_status" in st.session_state:
        lvl, msg = st.session_state["refresh_status"]
        getattr(st, lvl)(msg)

    st.divider()
    st.caption("All prices are for research and demonstration. Forecasts and signals are not investment advice.")


# ---------------------------------------------------------------------------
# Top Header Bar: Title, Ticker Segment, Model, Theme, Live Indicator
# ---------------------------------------------------------------------------
hdr_left, hdr_right = st.columns([1.5, 2.5])
with hdr_left:
    st.markdown(
        """
        <div style="padding: 4px 0;">
            <div class="mfl-title">Market forecast lab</div>
            <div class="mfl-sub">Model-driven price outlook for Colombo Stock Exchange (CSE)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hdr_right:
    c_tick, c_live = st.columns([3, 1])
    with c_tick:
        selected_symbol = st.pills(
            "Ticker",
            options=available_symbols,
            default=available_symbols[0],
            label_visibility="collapsed",
        ) or available_symbols[0]
    with c_live:
        st.markdown(
            f"""
            <div style="display:flex; justify-content:flex-end; align-items:center; height:100%;">
                <div class="mfl-live"><i></i>{last_update_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Get current symbol analytics
summary = get_symbol_summary(selected_symbol, REPORTS)
if not summary:
    st.error(f"No summary available for {selected_symbol}.")
    st.stop()


# ---------------------------------------------------------------------------
# Row 1: KPI Cards (Last price, 10-day target, Model signal, Confidence, Hit rate, Day range)
# ---------------------------------------------------------------------------
l_close = summary["close"]
chg = summary["change_pct"]
t_10d = summary["target_10d"]
move_10d = summary["move_10d_pct"]
sig_label = summary["signal_label"]
sig_class = summary["signal_class"]
conf = summary["confidence"]
hit_pct = summary["hit_rate"]
d_low = summary["day_low"]
d_high = summary["day_high"]

st.markdown(
    f"""
    <div class="mfl-kpis">
        <div class="mfl-kpi-card">
            <span class="lbl">Last price</span>
            <b class="val {cls(chg)}">{fmt(l_close)}</b>
            <span class="sub"><span class="{cls(chg)}">{pc(chg)} today</span></span>
        </div>
        <div class="mfl-kpi-card">
            <span class="lbl">10-day target</span>
            <b class="val">{fmt(t_10d)}</b>
            <span class="sub"><span class="{cls(move_10d)}">{pc(move_10d)} expected</span></span>
        </div>
        <div class="mfl-kpi-card">
            <span class="lbl">Model signal</span>
            <div style="margin-top: 4px;">
                <span class="chip {sig_class}">{sig_label}</span>
            </div>
            <span class="sub" style="color:var(--mut); margin-top:4px;">{selected_model} model</span>
        </div>
        <div class="mfl-kpi-card">
            <span class="lbl">Confidence</span>
            <b class="val">{conf}%</b>
            <span class="sub" style="color:var(--mut);">Model agreement</span>
        </div>
        <div class="mfl-kpi-card">
            <span class="lbl">Direction hit rate</span>
            <b class="val">{fmt(hit_pct * 100, 0)}%</b>
            <span class="sub" style="color:var(--mut);">Last 120 days</span>
        </div>
        <div class="mfl-kpi-card">
            <span class="lbl">Day range</span>
            <b class="val" style="font-size:19px;">{fmt(d_low)} ΓÇô {fmt(d_high)}</b>
            <span class="sub" style="color:var(--mut);">Low to high</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Main Layout: 2 Columns (Left: 2fr Charts, Right: 1fr Forecast & Diagnostics)
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([2.1, 1.0])

# ----------------- LEFT COLUMN: Main Multi-Pane Chart + Backtests -----------------
with col_left:
    # Main Candlestick / Forecast Area Card
    st.markdown(
        f"""
        <div class="mfl-card" style="margin-bottom: 12px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 8px;">
                <div>
                    <b style="font-size:15px; color:var(--ink);">{summary['symbol']} ┬╖ {summary['name']}</b>
                    <span class="mut" style="margin-left:8px; font-size:12px;">{summary['date']} ┬╖ C {fmt(summary['close'])} <span class="{cls(summary['change_pct'])}">{pc(summary['change_pct'])}</span></span>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    # Chart Controls
    c_ctl1, c_ctl2, c_ctl3, c_ctl4, c_ctl5 = st.columns([1, 1, 1, 1, 2])
    with c_ctl1:
        show_ma = st.checkbox("Averages", value=True)
    with c_ctl2:
        show_bb = st.checkbox("Bands", value=True)
    with c_ctl3:
        show_fc = st.checkbox("Forecast", value=True)
    with c_ctl4:
        show_vol = st.checkbox("Volume", value=True)
    with c_ctl5:
        timeframe = st.segmented_control("Window", ["1M", "3M", "6M", "All"], default="3M", label_visibility="collapsed") or "3M"

    # Multi-pane Plotly chart
    hist = summary["history"].copy()
    if not hist.empty:
        hist["date"] = pd.to_datetime(hist["date"])
        hist = hist.sort_values("date")

        # Slice by window
        win_days = 40 if timeframe == "1M" else (70 if timeframe == "3M" else (120 if timeframe == "6M" else 220))
        hist_slice = hist.tail(win_days).copy()

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # Subplots: 1: Price + Forecast, 2: RSI, 3: MACD
            fig = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.04,
                row_heights=[0.65, 0.17, 0.18],
            )

            dt_idx = hist_slice["date"]
            cl_arr = hist_slice["close"]

            # Main Price Series (Line / Candle)
            fig.add_trace(
                go.Scatter(
                    x=dt_idx,
                    y=cl_arr,
                    name="Close",
                    line=dict(color="#2563eb" if not is_dark else "#7b94ff", width=2),
                    mode="lines",
                ),
                row=1,
                col=1,
            )

            # Moving averages (20, 50)
            if show_ma and len(cl_arr) >= 20:
                s20 = cl_arr.rolling(20, min_periods=5).mean()
                fig.add_trace(
                    go.Scatter(x=dt_idx, y=s20, name="SMA 20", line=dict(color="#f59e0b", width=1.5)),
                    row=1,
                    col=1,
                )
                if len(cl_arr) >= 50:
                    s50 = cl_arr.rolling(50, min_periods=10).mean()
                    fig.add_trace(
                        go.Scatter(x=dt_idx, y=s50, name="SMA 50", line=dict(color="#a855f7", width=1.5)),
                        row=1,
                        col=1,
                    )

            # Bollinger Bands
            if show_bb and len(cl_arr) >= 15:
                s20_bb = cl_arr.rolling(20, min_periods=5).mean()
                sd20 = cl_arr.rolling(20, min_periods=5).std()
                bu = s20_bb + 2 * sd20
                bl = s20_bb - 2 * sd20
                fig.add_trace(
                    go.Scatter(
                        x=list(dt_idx) + list(dt_idx)[::-1],
                        y=list(bu) + list(bl)[::-1],
                        fill="toself",
                        fillcolor="rgba(123, 148, 255, 0.08)" if is_dark else "rgba(56, 86, 214, 0.08)",
                        line=dict(color="rgba(0,0,0,0)"),
                        hoverinfo="skip",
                        name="Bollinger Bands",
                    ),
                    row=1,
                    col=1,
                )

            # 10-day Forward Forecast Cone
            if show_fc and summary.get("fc_points"):
                fcs = summary["fc_points"]
                last_d = dt_idx.iloc[-1]
                last_p = cl_arr.iloc[-1]
                fc_dates = [last_d] + [last_d + pd.Timedelta(days=p["step"]) for p in fcs]
                fc_m = [last_p] + [p["target"] for p in fcs]
                fc_hi = [last_p] + [p["hi"] for p in fcs]
                fc_lo = [last_p] + [p["lo"] for p in fcs]

                cone_col = "rgba(43, 212, 125, 0.14)" if summary["move_10d_pct"] >= 0 else "rgba(255, 84, 104, 0.14)"
                line_col = "#2bd47d" if summary["move_10d_pct"] >= 0 else "#ff5468"

                fig.add_trace(
                    go.Scatter(
                        x=fc_dates + fc_dates[::-1],
                        y=fc_hi + fc_lo[::-1],
                        fill="toself",
                        fillcolor=cone_col,
                        line=dict(color="rgba(0,0,0,0)"),
                        hoverinfo="skip",
                        name="80% Range",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=fc_dates,
                        y=fc_m,
                        name="10D Forecast",
                        line=dict(color=line_col, width=2, dash="dash"),
                    ),
                    row=1,
                    col=1,
                )

            # Pane 2: RSI
            diff = cl_arr.diff()
            gain = diff.clip(lower=0).rolling(14, min_periods=5).mean()
            loss = (-diff.clip(upper=0)).rolling(14, min_periods=5).mean()
            rs = gain / (loss.replace(0, np.nan))
            rsi_series = 100 - (100 / (1 + rs)).fillna(50)

            fig.add_trace(
                go.Scatter(
                    x=dt_idx,
                    y=rsi_series,
                    name="RSI 14",
                    line=dict(color="#3856d6" if not is_dark else "#7b94ff", width=1.8),
                ),
                row=2,
                col=1,
            )
            # RSI overbought / oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="#8896a7", line_width=1, row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#8896a7", line_width=1, row=2, col=1)

            # Pane 3: MACD
            e12 = cl_arr.ewm(span=12, adjust=False).mean()
            e26 = cl_arr.ewm(span=26, adjust=False).mean()
            macd_line = e12 - e26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            hist_macd = macd_line - signal_line
            hist_colors = ["#0f9d58" if v >= 0 else "#d93040" for v in hist_macd]

            fig.add_trace(
                go.Bar(
                    x=dt_idx,
                    y=hist_macd,
                    name="MACD Hist",
                    marker_color=hist_colors,
                ),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=dt_idx, y=macd_line, name="MACD", line=dict(color="#7b94ff" if is_dark else "#3856d6", width=1.5)),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=dt_idx, y=signal_line, name="Signal", line=dict(color="#f59e0b", width=1.5)),
                row=3,
                col=1,
            )

            fig.update_layout(
                template="plotly_dark" if is_dark else "plotly_white",
                height=540,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
                showlegend=False,
            )
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
            fig.update_yaxes(title_text="MACD", row=3, col=1)

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.line_chart(hist_slice.set_index("date")["close"], use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Sub-Grid: Predicted vs Actual (Last 40 Days) & Strategy Backtest (Last 120 Days)
    sub_l, sub_r = st.columns(2)

    with sub_l:
        st.markdown(
            """
            <div class="mfl-card">
                <h2>Predicted vs actual close, last 40 days</h2>
            """,
            unsafe_allow_html=True,
        )
        if not hist.empty:
            sub_40 = hist.tail(40).copy()
            sub_40["predicted"] = sub_40["close"].shift(1).fillna(sub_40["close"]) * (1 + (sub_40["probability"].fillna(0.5) - 0.5) * 0.02)
            try:
                import plotly.graph_objects as go
                pva_fig = go.Figure()
                pva_fig.add_trace(go.Scatter(x=sub_40["date"], y=sub_40["close"], name="Actual", line=dict(color=css_vars.split("--ink:")[1].split(";")[0].strip(), width=2)))
                pva_fig.add_trace(go.Scatter(x=sub_40["date"], y=sub_40["predicted"], name="Predicted", line=dict(color="#3856d6" if not is_dark else "#7b94ff", width=2, dash="dot")))
                pva_fig.update_layout(
                    template="plotly_dark" if is_dark else "plotly_white",
                    height=200,
                    margin=dict(l=5, r=5, t=5, b=5),
                    hovermode="x unified",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(pva_fig, use_container_width=True)
            except Exception:
                st.line_chart(sub_40.set_index("date")[["close", "predicted"]], use_container_width=True)
        st.markdown('<p class="sub mut" style="font-size:12px; margin-top:4px;">Dots: green when model called direction correctly, red when missed.</p></div>', unsafe_allow_html=True)

    with sub_r:
        st.markdown(
            """
            <div class="mfl-card">
                <h2>Strategy backtest, out-of-sample</h2>
            """,
            unsafe_allow_html=True,
        )
        if not equity.empty:
            eq_chart = equity.select_dtypes(include="number").tail(120)
            try:
                import plotly.graph_objects as go
                eq_fig = go.Figure()
                eq_fig.add_trace(go.Scatter(x=eq_chart.index, y=eq_chart.iloc[:, 0], name="Strategy", line=dict(color="#0f9d58" if not is_dark else "#2bd47d", width=2), fill="tozeroy", fillcolor="rgba(15, 157, 88, 0.08)"))
                eq_fig.update_layout(
                    template="plotly_dark" if is_dark else "plotly_white",
                    height=200,
                    margin=dict(l=5, r=5, t=5, b=5),
                    showlegend=False,
                )
                st.plotly_chart(eq_fig, use_container_width=True)
            except Exception:
                st.line_chart(eq_chart, use_container_width=True)

        tot_ret = metrics.get("total_return", 0.0)
        cagr_ret = metrics.get("cagr", 0.0)
        sr_val = metrics.get("sharpe_ratio", 0.0)
        dd_val = metrics.get("max_drawdown", 0.0)
        win_val = metrics.get("win_rate")
        trades_n = metrics.get("number_of_trades", 0)

        st.markdown(
            f"""
            <div class="mfl-stats" style="margin-top:8px;">
                <div><span>Strategy</span><b class="{cls(tot_ret)}">{pc(tot_ret)}</b></div>
                <div><span>CAGR</span><b class="{cls(cagr_ret)}">{pc(cagr_ret)}</b></div>
                <div><span>Sharpe</span><b>{fmt(sr_val)}</b></div>
                <div><span>Max DD</span><b class="dn">{pc(dd_val)}</b></div>
                <div><span>Win Rate</span><b class="{cls(win_val - 0.5)}">{fmt(win_val * 100, 0)}%</b></div>
                <div><span>Trades</span><b>{trades_n}</b></div>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ----------------- RIGHT COLUMN: Forecast Table, Sentiment, Drivers, Accuracy -----------------
with col_right:
    # 1. 10-day price forecast Table
    st.markdown(
        """
        <div class="mfl-card">
            <h2>10-day price forecast</h2>
            <table class="mfl-table">
                <thead>
                    <tr><th>Horizon</th><th>Target</th><th>Change</th><th>80% range</th></tr>
                </thead>
                <tbody>
        """,
        unsafe_allow_html=True,
    )
    for row in summary["forecast_table"]:
        st.markdown(
            f"""
            <tr>
                <td>{row['horizon']}</td>
                <td>{fmt(row['target'])}</td>
                <td class="{cls(row['change_pct'])}">{pc(row['change_pct'])}</td>
                <td class="mut">{fmt(row['range_lo'], 0)} ΓÇô {fmt(row['range_hi'], 0)}</td>
            </tr>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</tbody></table></div>", unsafe_allow_html=True)

    # 2. Market sentiment
    sent_score = summary["sentiment_score"]
    sent_lbl = summary["sentiment_label"]
    sent_cls = "up" if sent_score > 55 else ("dn" if sent_score < 45 else "mut")

    st.markdown(
        f"""
        <div class="mfl-card">
            <h2>Market sentiment</h2>
            <div style="text-align:center; padding: 6px 0;">
                <div style="font-size:32px; font-weight:700; color:var(--ink);">{sent_score}</div>
                <div class="{sent_cls}" style="font-size:16px; font-weight:700;">{sent_lbl}</div>
            </div>
            <div style="background:var(--grid); height:8px; border-radius:4px; overflow:hidden; margin:8px 0;">
                <div style="width:{sent_score}%; height:100%; background:var(--{sent_cls});"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:11px; color:var(--mut);">
                <span>Bearish</span>
                <span>Neutral</span>
                <span>Bullish</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. What is moving the forecast (Drivers)
    st.markdown(
        """
        <div class="mfl-card">
            <h2>What is moving the forecast</h2>
        """,
        unsafe_allow_html=True,
    )
    for name, val in summary["feat_drivers"]:
        bar_col = "var(--up)" if val >= 0 else "var(--dn)"
        pos_style = "left:50%;" if val >= 0 else "right:50%;"
        width_pct = abs(val) * 50
        st.markdown(
            f"""
            <div class="fb">
                <span>{name}</span>
                <div class="t">
                    <i style="background:{bar_col}; {pos_style} width:{width_pct:.1f}%;"></i>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('<p class="sub mut" style="margin-top:6px;">Right of centre pushes the forecast up, left pushes it down.</p></div>', unsafe_allow_html=True)

    # 4. Model accuracy & Confusion Matrix
    m = summary["metrics"]
    tp, fp, fn, tn = summary["cm"]
    st.markdown(
        f"""
        <div class="mfl-card">
            <h2>Model accuracy</h2>
            <div class="mfl-stats">
                <div><span>RMSE</span><b>{fmt(m['rmse'])}</b></div>
                <div><span>MAE</span><b>{fmt(m['mae'])}</b></div>
                <div><span>MAPE</span><b>{fmt(m['mape'] * 100, 1)}%</b></div>
                <div><span>R┬▓ (price)</span><b>{fmt(m['r2'], 3)}</b></div>
                <div><span>Direction</span><b>{fmt(m['hit_rate'] * 100, 1)}%</b></div>
                <div><span>Hit streak</span><b>{m['streak']} days</b></div>
            </div>
            <div class="cm">
                <span></span><span class="mut">Actual up</span><span class="mut">Actual dn</span>
                <span class="mut">Called up</span>
                <div class="c" style="background:color-mix(in srgb,var(--up) 22%,transparent); color:var(--up);">{tp}</div>
                <div class="c" style="background:color-mix(in srgb,var(--dn) 22%,transparent); color:var(--dn);">{fp}</div>
                <span class="mut">Called dn</span>
                <div class="c" style="background:color-mix(in srgb,var(--dn) 22%,transparent); color:var(--dn);">{fn}</div>
                <div class="c" style="background:color-mix(in srgb,var(--up) 22%,transparent); color:var(--up);">{tn}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Bottom Row: Sector Heatmap + Watchlist
# ---------------------------------------------------------------------------
heat_col, watch_col = st.columns([1.2, 1.8])

with heat_col:
    st.markdown(
        """
        <div class="mfl-card">
            <h2>Sector heatmap, latest observed move</h2>
            <div class="mfl-heat">
        """,
        unsafe_allow_html=True,
    )

    sector_moves = {}
    for sym in available_symbols:
        item = get_symbol_summary(sym, REPORTS)
        if item and item.get("sector") and pd.notna(item.get("change_pct")):
            sector_moves.setdefault(item["sector"], []).append(float(item["change_pct"]) * 100)
    if sector_moves:
        for sector, moves in sorted(sector_moves.items()):
            move = float(np.mean(moves))
            tone = "up" if move >= 0 else "dn"
            st.markdown(
                f'<div style="background:color-mix(in srgb,var(--{tone}) 20%,var(--panel));"><span>{sector}</span><b class="{tone}">{move:+.2f}%</b></div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("Sector performance is unavailable: no sector-tagged signal returns were found.")

    st.markdown("</div></div>", unsafe_allow_html=True)

with watch_col:
    st.markdown(
        """
        <div class="mfl-card">
            <h2>Watchlist</h2>
            <table class="mfl-table">
                <thead>
                    <tr><th>Symbol</th><th>Company</th><th>Sector</th><th>Last Price</th><th>Day</th><th>Signal</th></tr>
                </thead>
                <tbody>
        """,
        unsafe_allow_html=True,
    )
    univ_info = load_universe_names()
    for sym in available_symbols:
        s_sum = get_symbol_summary(sym, REPORTS)
        if s_sum:
            st.markdown(
                f"""
                <tr>
                    <td><b>{sym}</b></td>
                    <td style="text-align:left;">{s_sum['name']}</td>
                    <td class="mut" style="text-align:left;">{s_sum['sector']}</td>
                    <td>{fmt(s_sum['close'])}</td>
                    <td class="{cls(s_sum['change_pct'])}">{pc(s_sum['change_pct'])}</td>
                    <td><span class="chip {s_sum['signal_class']}">{s_sum['signal_label']}</span></td>
                </tr>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</tbody></table></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Collapsible Research Section (Folds & Data Diagnostics)
# ---------------------------------------------------------------------------
with st.expander("≡ƒôü Walk-Forward Fold Performance & System Health"):
    f_col1, f_col2 = st.columns([2, 1])
    with f_col1:
        st.markdown("**Per-Fold Out-of-Sample Results**")
        if not folds.empty:
            st.dataframe(folds, use_container_width=True, hide_index=True)
        else:
            st.info("No fold metrics generated.")
    with f_col2:
        st.markdown("**Lab Diagnostics**")
        source = metadata.get("data_source", "Deterministic synthetic / local cache")
        st.write(f"ΓÇó **Data Source**: {source}")
        st.write(f"ΓÇó **Coverage**: {str(status.get('data_start') or 'ΓÇö')[:10]} to {str(status.get('data_end') or 'ΓÇö')[:10]}")
        st.write(f"ΓÇó **Folds**: {status.get('fold_count', len(folds))}")
        st.write(f"ΓÇó **Predictions Logged**: {status.get('prediction_count', len(predictions))}")
        st.caption(f"Exports directory: `{REPORTS.resolve()}`")

st.markdown('<p class="sub mut" style="text-align:center; margin-top:20px;">All prices are for research demonstration. Colombo Stock Exchange (CSE) quantitative forecasts and signals are not investment advice.</p>', unsafe_allow_html=True)
