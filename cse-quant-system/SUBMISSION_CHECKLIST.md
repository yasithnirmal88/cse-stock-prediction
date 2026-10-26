# CSE Quant System — Submission Checklist

## Implemented

- CSE OHLCV loading, local cache access, validation, and deterministic synthetic test data.
- Causal feature engineering, directional model interfaces, momentum/pairs research, and `SignalEngine` signal generation.
- Fold-based expanding/rolling walk-forward evaluation with GARCH-based sizing, position limits, and simplified transaction costs.
- Metrics and HTML/PDF report generation, latest-result exports, and Streamlit dashboard with explicit reload and optional smoke-test controls.
- Configuration, dependency groups, repository ignore rules, academic notes, and user instructions.

## Reproduce and verify

Run commands from the repository root, using the Python environment installed from `requirements.txt`.

1. **Software smoke test (no market-data files required):**
   ```powershell
   python scripts/smoke_test_walk_forward.py --data-mode synthetic
   ```
   Uses deterministic synthetic OHLCV for three symbols, evaluates walk-forward folds, and writes reports and latest artifacts. These are software-validation outputs, not empirical CSE results.

2. **Main empirical run (requires prepared local data):**
   Put one suitable daily OHLCV CSV per symbol in `data/processed` (or the configured `data.processed_dir`), with a date column/index and `Open`, `High`, `Low`, `Close`, `Volume` columns. Then run:
   ```powershell
   python scripts/run_full_walk_forward.py
   ```
   The script engineers features, executes the configured walk-forward analysis, and writes reports under `reports/` and dashboard artifacts under `reports/latest/`. It does not download source data.

3. **Dashboard:**
   ```powershell
   streamlit run app.py
   ```
   Health endpoint when running locally: `http://localhost:8501/_stcore/health`.

4. **Code checks:**
   ```powershell
   python -m compileall -q app.py src scripts
   python -c "import src.data_loader, src.data_validator, src.features, src.signals; import src.backtester, src.dashboard_utils; print('Imports OK')"
   ```

Optional `python scripts/check_data_quality.py` requires a local `data/universe.yaml`, which is not included by default. Smoke-test alternatives are `--data-mode auto` (local cache or synthetic fallback) and `--data-mode real` (requires at least three cached symbols with sufficient recent rows; no network fetch).

## Known limitations

- Synthetic data is generated for repeatable software testing only; it is not market evidence.
- Free/delayed data may be incomplete, revised, or inconsistent in corporate-action treatment. No survivorship-free, point-in-time CSE universe is supplied.
- Default costs (0.56% entry and 0.56% exit) are simplified and exclude complete spread, slippage, impact, liquidity, tax, and broker-fee effects.
- Daily-bar fills, limited model selection, and risk/reporting choices are academic simplifications. Results can be affected by selection bias, overfitting, data quality, and execution assumptions; the system is not a live trading platform or financial advice.

## Main deliverables

- **Code:** `app.py`, `src/` modules, `scripts/`, `config.yaml`, `requirements.txt`.
- **Reports and data products:** generated HTML/PDF research reports plus CSV/JSON metrics, equity, folds, predictions, signals, and run metadata under `reports/`.
- **Dashboard:** Streamlit application at `app.py`, consuming saved exports from `reports/latest/`.
- **Documentation:** `README.md`, this checklist, and `.gitignore`.

Generated reports are reproducible outputs, not guaranteed to be present in a clean checkout; regenerate them using the commands above. Review and include only the intended run artifacts with the academic submission.
