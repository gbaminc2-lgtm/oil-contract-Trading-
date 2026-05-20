# Quant Energy Commodity Pipeline

**Institutional-Grade Quantitative Commodity Trading System**  
WTI Crude Oil · Brent · RBOB Gasoline · ULSD · Options & Futures

[![CI](https://github.com/gbaminc2-lgtm/oil-contract-Trading-/actions/workflows/ci.yml/badge.svg)](https://github.com/gbaminc2-lgtm/oil-contract-Trading-/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://www.python.org/)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

A complete, four-module institutional trading pipeline implementing:

| Strategy | Model | Reference |
|----------|-------|-----------|
| Storage Arbitrage | Cost-of-Carry F=S·e^(r+u−y)·T | Gkinis, NYMEX Ch.200 |
| Crack Spread Trading | 3-2-1 / 5-3-2 mean reversion | Trafigura, Oil Trader Academy |
| Basis Trading | Physical–futures differential | Oil Contracts book |
| Directional Futures | Term structure + GARCH + seasonal | Gkinis, Hull |
| Options Vol Strategies | Black-76 (energy standard) | Bittman, Sherbin |
| ML Signal | GradientBoosting, 15 features | QuantStart |
| Monte Carlo | GBM + Ornstein-Uhlenbeck | Gkinis Ch.5–6 |
| Risk Management | VaR + ES + Basel III + 8 stress tests | Hull 4th Ed. |

---

## Architecture

```
quant-energy-pipeline/
├── CLAUDE.md              ← Claude Code project instructions
├── data_agent.py          ← Market data layer
├── strategy_agent.py      ← Pricing engine + signal generation
├── risk_agent.py          ← Risk limits + VaR + approval gate
├── main.py                ← 10-step orchestration pipeline
├── tests/
│   ├── test_data_agent.py
│   ├── test_strategy_agent.py
│   └── test_risk_agent.py
├── requirements.txt
├── .gitignore
├── .vscode/
│   ├── settings.json      ← Python formatting + linting config
│   ├── extensions.json    ← Recommended extensions
│   ├── launch.json        ← 15 debug/run configurations
│   └── tasks.json         ← Build + test + lint tasks
└── .github/
    └── workflows/
        └── ci.yml         ← 9-job CI/CD pipeline
```

---

## Quick Start

### 1. Clone & set up

```bash
git clone https://github.com/gbaminc2-lgtm/oil-contract-Trading-.git
cd oil-contract-Trading-

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
# Full 10-step institutional pipeline
python main.py

# Black-76 Greeks table (energy options standard)
python main.py --bsm-demo --spot 72.0 --sigma 0.32

# Monte Carlo: GBM vs Ornstein-Uhlenbeck (50,000 paths each)
python main.py --mc-demo

# 8-scenario stress test report
python main.py --stress-only

# PPM (Private Placement Memorandum)
python main.py --ppm

# Risk state report
python main.py --risk-report

# Brent crude
python main.py --ticker BZ=F
```

### 3. Open in VS Code

```bash
code .
# Ctrl+Shift+P → "Claude: Open" to launch Claude Code
```

All 15 debug configurations are in `.vscode/launch.json` (F5 to run).

---

## Module Reference

### `data_agent.py` — Market Data Layer

```python
from data_agent import (
    fetch_spot,              # Latest WTI/Brent spot price
    fetch_forward_curve,     # 12-month cost-of-carry curve
    fetch_crack_spreads,     # 3-2-1 / 5-3-2 refinery margin
    fetch_storage_economics, # Cushing storage arb economics
    fetch_garch_vol,         # GARCH(1,1) volatility term structure
    fetch_seasonal_pattern,  # 12-month seasonal factors
    fetch_iv_surface,        # IV surface (negative skew convention)
    build_ml_features,       # GBM feature matrix for ML
    fetch_risk_free_rate,    # FRED 3M T-bill
)
```

All functions have **synthetic data fallbacks** — runs fully offline.

---

### `strategy_agent.py` — Pricing + Signal Engine

```python
from strategy_agent import (
    # Black-76 (use for ALL energy/futures options)
    black76,                 # Greeks: Δ, Γ, Θ, V, ρ, vanna, charm
    black_scholes,           # BSM for equity options only
    implied_vol_bisection,   # IV solver

    # Monte Carlo
    monte_carlo_gbm,         # GBM paths (short-dated, options VaR)
    monte_carlo_ou,          # Ornstein-Uhlenbeck (long-run mean reversion)

    # Commodity strategies
    generate_storage_arb,    # Buy spot + store + sell forward
    generate_crack_spread_signal,
    generate_basis_trade,
    generate_futures_signal,
    generate_options_signal,

    # Financials
    build_nav_model,
    build_three_statement,   # P&L + Balance Sheet + Cash Flow
    generate_ppm_sections,   # PPM report (7 sections)

    # ML
    train_ml_signal,
    ml_predict_direction,
)
```

**Pricing convention for energy options:**
```python
# CORRECT — Black-76 with futures price as underlying
g = black76(F=72.50, K=75.0, T=45/365, r=0.053, sigma=0.32, right=OptionRight.CALL)

# WRONG for energy — BSM assumes spot, not futures
# g = black_scholes(S=72.50, ...)   ← Do not use for WTI/Brent/RBOB/ULSD
```

---

### `risk_agent.py` — Risk Management Engine

**Section 1 (hardcoded — never modified without risk-committee sign-off):**

| Parameter | Value |
|-----------|-------|
| `FUND_EQUITY_USD` | $10,000,000 |
| `MAX_RISK_PER_TRADE_PCT` | 1% |
| `MAX_PORTFOLIO_HEAT_PCT` | 8% |
| `MAX_WTI_CONTRACTS` | 50 lots |
| `MAX_VAR_1D_99_USD` | $250,000 |
| `MAX_DAILY_LOSS_USD` | $100,000 |
| `MAX_ABSOLUTE_DRAWDOWN_PCT` | 15% |
| `MIN_DTE` | 21 days |
| `VAR_CONFIDENCE` | 99% |

```python
from risk_agent import (
    evaluate_trade,     # Pre-trade gate → APPROVED / FLAGGED / REJECTED
    parametric_var,     # Variance-covariance VaR (Hull Ch.12)
    historical_var,     # Historical simulation VaR (Hull Ch.13)
    monte_carlo_var,    # MC VaR with Student-t(5) fat tails
    expected_shortfall, # CVaR / ES (Basel III / FRTB)
    run_stress_tests,   # 8 named scenarios
    check_greeks_limits,
    basel_iii_capital,
    record_pnl,
    performance_report,
)
```

---

### `main.py` — 10-Step Pipeline

```
Step 1:  Data ingestion (spot, curves, GARCH, crack, storage, IV, ML)
Step 2:  ML direction signal (GradientBoosting, 15 commodity features)
Step 3:  Generate candidate signals (all 5 strategies)
Step 4:  Risk evaluation (pre-trade gate on every signal)
Step 5:  NAV model + Three-Statement Financial Model
Step 6:  Monte Carlo scenarios (GBM + OU, 50K paths each)
Step 7:  Stress testing (8 named scenarios)
Step 8:  PPM report generation (7 sections)
Step 9:  Performance report (Sharpe, Sortino, VaR, ES, Basel III)
Step 10: Session summary
```

---

## CI/CD Pipeline (GitHub Actions)

| Job | Trigger | Description |
|-----|---------|-------------|
| `lint` | every push/PR | black + ruff + isort |
| `typecheck` | after lint | mypy static analysis |
| `test` | after lint | pytest × Python 3.10/3.11/3.12 |
| `model-smoke` | after tests | Black-76 + MC offline smoke test |
| `pipeline-offline` | after smoke | Full pipeline synthetic data run |
| `risk-guard` | after lint | Verifies hardcoded risk params unchanged |
| `security` | after lint | bandit + pip-audit |
| `claude-code-check` | every push | CLAUDE.md section validation |
| `notify` | on failure | GitHub step summary |

**Schedule:** Daily at 09:00 AM EST (Monday–Friday) via cron.

---

## Claude Code Integration

This project is optimized for **Claude Code for VS Code** (`anthropic.claude-code`).

The `CLAUDE.md` file at the repository root provides Claude Code with:
- Full system role definition (Expert Quantitative Commodity Strategist)
- Architecture overview and module responsibilities
- Strict constraints (never modify hardcoded risk params, always use Black-76)
- Pricing conventions (Black-76 for energy, BSM for equity)
- Complete data flow diagram
- All 21 knowledge base references

**Opening with Claude Code:**
```bash
# Install Claude Code extension in VS Code, then:
code .
# Claude Code automatically reads CLAUDE.md on project open
```

---

## Knowledge Base

21 professional texts integrated across all modules:

**Oil & Commodity:** Oil Trader Academy · Hedging Strategies in Crude Oil Futures · Hedging Strategies for Oil End-Users · Commodities Demystified (Trafigura) · Oil Contracts: How to Read Petroleum Contracts · NYMEX Chapter 200 · CME Customer Center Manual  
**Energy Derivatives:** Modelling Energy Markets & Pricing Energy Derivatives (Gkinis)  
**Options:** Trading Options as a Professional (Bittman) · How to Price and Trade Options (Sherbin) · Options Strategies (Danes)  
**Risk:** Risk Management & Financial Institutions (Hull 4th Ed.) · Handbook of Risk (IMCA/Wiley)  
**Algo/Quant:** Successful Algorithmic Trading (QuantStart)  
**Technical:** Art & Science of Technical Analysis (Grimes) · Technical Trading Tactics  
**Psychology:** Mastering Trading Psychology (Aziz)  
**Day Trading:** Complete Guide to Day Trading (Heitkoetter) · Master the Markets (VSA — Williams)

---

## Disclaimer

This software is for **research and educational purposes only**.  
It does not constitute financial advice or a solicitation to trade.  
All live trading requires compliance review of every hardcoded risk parameter  
and sign-off from a qualified risk officer before deployment.  
Past model performance does not guarantee future results.

---

## License

MIT — see `LICENSE`.
