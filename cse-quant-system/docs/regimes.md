# Sri Lanka CSE Market Regimes and Structural Breaks

This document outlines structural regimes and market breakpoints identified in the Colombo Stock Exchange (CSE) S&P SL20 multi-decade dataset (2001–2025). Pre-specifying and adhering to these regimes prevents lookahead bias and cherry-picking when evaluating quantitative models.

---

## 1. Identified Structural Regimes

### A. Pre-Crisis Era & Civil Conflict Period (2001 – May 2009)
- Characterized by civil conflict, wartime liquidity constraints, higher inflation risk premia, and intermittent market trading volumes.
- Post-war surge occurred from May 2009 onwards.

### B. Post-War Expansion & Normalization (June 2009 – 2019)
- Rebound in foreign investment, infrastructure expansion, and regular trading sessions.
- Relatively stable baseline market conditions across the primary banking and blue-chip conglomerate universe (JKH, COMB, SAMP, HNB).

### C. COVID-19 Disruption Window (March 2020 – December 2020)
- **Market Closure**: CSE suspended active trading between March 20, 2020 and May 11, 2020.
- **Reopening Shock**: Extreme volatility upon trading resumption, circuit breakers triggered, and rapid monetary easing.
- **Evaluation Rule**: Models evaluated over 2020 should account for the mandatory trading halt; returns across the closure window should not be treated as a single continuous daily trading session.

### D. Economic Crisis & Currency Devaluation (2021 – 2023)
- Severe foreign exchange liquidity constraints, debt restructuring announcements, sovereign rating downgrades, and major data gaps.
- Extensive price adjustments and high domestic inflation.

### E. Post-Gap & Recovery Regime (2024 – 2025)
- Stabilization under IMF program conditions, banking sector recovery, and renewed equity turnover.
- Reflects modernized trading rules and current valuation multiples.

---

## 2. Volatility Regimes (Pre-registered Terciles)

To avoid lookahead bias during walk-forward cross-validation, volatility regimes should be defined strictly on the training period:
- **Calm**: 22-day rolling annualized volatility $\le 10\%$
- **Normal**: 22-day rolling annualized volatility between $10\%$ and $25\%$
- **Turbulent / Stressed**: 22-day rolling annualized volatility $> 25\%$

---

## 3. Data Cleaning and Handling Guidelines

1. **Ordinary Voting Shares ('N' Series)**:
   - Always isolate ordinary voting shares (e.g. `JKH.N0000`, `COMB.N0000`) from non-voting (`X`), rights (`R`), warrants (`W`), and debentures (`D`) to prevent artificial return contamination.

2. **Gaps Across Breaks**:
   - Calendar gaps exceeding 10 trading/calendar days break continuous technical indicator windows. Rolling windows (e.g., 20-day SMA, RSI) should be reset rather than interpolated across month-long halts.

3. **Corporate Actions & Split Thresholds**:
   - Single-day absolute log returns $|r_t| > 0.35$ (35%) in historical unadjusted prints typically indicate stock splits, bonus issues, or rights distributions.
