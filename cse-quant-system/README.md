# CSE Quant System

## Project goal

This university research project provides a transparent, reproducible foundation for quantitative analysis of equities listed on the Colombo Stock Exchange (CSE). It is intended for educational and research purposes, not as investment advice or an automated trading service.

## Phase 1 scope

Phase 1 establishes project configuration, a manually curated initial watchlist, data and report directories, and Python module boundaries for later research. It deliberately does not download market data, train or implement models, or place trades. The universe in `data/universe.yaml` is a starting point, not a verified or point-in-time liquidity ranking.

## Install

Use Python 3.10 or later and run commands from this project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Configure and run

Edit `config.yaml` to set the research date range and other project settings. Data and output paths are project-relative; run scripts from this directory so those relative paths resolve consistently.

```powershell
python app.py
```

The initial application validates and displays the project configuration. No data is fetched by running it.

## Academic and research notes

- Transaction costs are material: the configured entry and exit commission rates are each 0.56%, or 1.12% round trip before slippage, taxes, market impact, and other applicable charges. Confirm current CSE fees before any empirical study.
- The initial universe focuses on liquid securities, which can reduce (but does not eliminate) liquidity and execution concerns. It can introduce survivorship and selection bias, especially when applied retrospectively.
- Use point-in-time constituents and corporate-action-adjusted prices where possible. Document data sources, missing observations, delistings, assumptions, and date ranges.
- Backtests are sensitive to costs, execution assumptions, look-ahead bias, and overfitting. Historical performance does not establish future returns.
- This project is for academic research only and does not constitute financial advice.
