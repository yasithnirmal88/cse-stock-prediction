# CSE Quant System

An educational quantitative-research pipeline for Colombo Stock Exchange (CSE) equities: OHLCV preparation, causal features, probability models, signal generation, volatility-aware sizing, walk-forward evaluation, reporting, and a Streamlit results dashboard.

> **Academic use only.** Not investment advice or a production trading system. Historical performance does not predict future returns.

## Implemented features

- Free-source data loading/cache and OHLCV validation; deterministic synthetic data for software checks. Full walk-forward runner reads processed CSV files and does not download data.
- Causal indicators; scikit-learn direction model and optional XGBoost/LightGBM; moving-average and pairs research components.
- Trend-gated signals, GARCH volatility sizing, and exposure limits.
- Expanding/rolling walk-forward folds, fresh model fitting, out-of-sample predictions, portfolio results, and transaction costs.
- JSON/CSV metrics, interactive HTML (Plotly), and PDF via available renderer or minimal fallback.
- Streamlit dashboard for saved results. Reload and quick smoke test are explicit controls; no automatic analysis/download on startup.

## Project structure

```text
.
├── app.py                         # Streamlit dashboard
├── config.yaml                    # Paths, costs, walk-forward settings
├── requirements.txt               # Core, reporting and optional dependencies
├── data/                          # Local/raw/processed data (not committed)
├── reports/                       # Generated reports/dashboard exports
├── scripts/
│   ├── run_full_walk_forward.py   # Full analysis entry point
│   ├── smoke_test_walk_forward.py # Synthetic/local-cache functional test
│   ├── check_data_quality.py      # Data quality check
│   ├── run_phase1_baseline.py     # Baseline workflow
│   ├── run_phase2_signals.py      # Signal workflow
│   ├── run_pairs_analysis.py      # Pairs research
│   ├── test_garch_sizing.py       # GARCH sizing check
│   └── train_xgboost_baseline.py  # Optional model experiment
└── src/
    ├── data_loader.py, data_validator.py, features.py
    ├── signals.py, dashboard_utils.py
    ├── models/                    # Direction, momentum, pairs, volatility
    └── backtester/                # Walk-forward, simple backtest, reports
```

Supply processed CSVs under `data/processed` (or configured path); data and optional universe files may not be in the checkout. Analysis/smoke tests generate `reports/latest` artifacts.

## Install

Python 3.10+; run from the repository root (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file groups core, reporting, and optional model packages. PDF options can need OS libraries or separately installed `wkhtmltopdf`; a minimal PDF fallback is included.

## Configure and run
The optional `check_data_quality.py` workflow specifically requires `data/universe.yaml` with a `stocks` list of symbol mappings; that file is not included in this checkout by default.


Edit `config.yaml`. Full analysis requires daily OHLCV CSVs (`Open`, `High`, `Low`, `Close`, `Volume`) in the processed directory; one file per symbol, with filenames used to derive symbols. Run from repository root:

```powershell
python scripts/check_data_quality.py        # optional data validation
python scripts/run_full_walk_forward.py     # empirical run using local processed CSVs
python scripts/smoke_test_walk_forward.py --data-mode synthetic
python scripts/smoke_test_walk_forward.py --data-mode real
```

Smoke-test modes: `synthetic` requires no market data; `real` requires sufficient local cache, without network fallback; `auto` uses sufficient cache or falls back to synthetic. Synthetic output verifies code paths only. Free sources can be delayed, incomplete, revised, or inconsistent in corporate-action treatment.

## Dashboard

```powershell
streamlit run app.py
```

Dashboard reads `reports/latest/`. **Load Latest Results** reloads saved artifacts. **Run Quick Smoke Test** is opt-in (local cache or synthetic) and refreshes exports. Startup neither fetches data nor runs analysis; absent artifacts show a missing-results state. Reports and CSV/JSON artifacts are generated under `reports/`, including `reports/smoke_test/` and `reports/latest/`.

## Academic notes

- `config.yaml` sets entry and exit commission to 0.56% each (1.12% round trip). Walk-forward cost is applied on position changes and split equally between entry/exit. This fixed-rate model does not fully capture spread, slippage, market impact, liquidity, taxes, or all exchange/broker charges. Verify actual CSE fees for the study period.
- Default methodology: expanding window, 504 minimum training observations, 63-observation test windows, 21-observation steps. Each fold fits on past data and tests subsequent dates. Results depend on labels, features, thresholds, and missing-data handling.
- Synthetic input is not CSE data. Free sources may be delayed/incomplete. No verified survivorship-free point-in-time security universe is supplied; retrospective lists risk selection bias. Daily-bar execution is simplified; this is not live-trading software.
- Fixed costs, limited models, and compact risk/reporting choices were intentional university-project simplifications. Avoid tuning on the final test period. Disclose data sources, dates, preprocessing, exclusions, costs, and configuration. No financial advice is provided.

## Phase overview

1. **Foundations/baseline:** configuration, data access/validation building blocks.
2. **Features/signals:** causal indicators, model/strategy components, signal generation.
3. **Evaluation/reporting:** walk-forward, volatility sizing, costs, metrics, HTML/PDF output.
4. **Dashboard:** saved-result visualization and explicit reload/quick-test controls.
5. **Submission readiness:** documentation, dependency grouping, repository hygiene, verification.

## Verification

Run from project root:

```powershell
python -m compileall -q app.py src scripts
python -c "import src.data_loader, src.data_validator, src.features, src.signals; import src.backtester, src.dashboard_utils; print('Imports OK')"
python scripts/smoke_test_walk_forward.py --data-mode synthetic
streamlit run app.py
```

Smoke test regenerates reports and latest artifacts. With Streamlit running, check `http://localhost:8501/_stcore/health`. A full empirical run also requires prepared processed market data.
