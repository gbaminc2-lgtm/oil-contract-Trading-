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

## Baum-Welch HMM Regime Engine (`hmm_regime.py`)

The system uses a **Gaussian Hidden Markov Model** trained via the Baum-Welch EM algorithm
(Hasegawa-Johnson, ECE 417 Lecture 15) to classify the WTI crude oil market into 4 hidden regimes.
This replaces the legacy 50MA/200MA heuristic across every agent in the pipeline.

### Mathematical Framework

```
Hidden states  q_t ∈ {BULL, BEAR, VOLATILE, SIDEWAYS}        N = 4
Observations   x_t = [daily_return%, rolling_vol%, rsi_norm, log_vol_norm]  D = 4
Emission       b_i(x) = N(x; μ_i, Σ_i)   ← Gaussian pdf per state
```

**Forward algorithm** (log-space):
```
α_1(i)  = π_i · b_i(x_1)
α_t(j)  = [Σ_i α_{t-1}(i) · a_ij] · b_j(x_t)
```

**E-step posteriors**:
```
γ_t(i)    = α_t(i) · β_t(i) / Σ_k α_t(k) · β_t(k)     ← state occupation
ξ_t(i,j)  = α_t(i) · a_ij · b_j(x_{t+1}) · β_{t+1}(j) / p(X|Λ)   ← transition
```

**Baum-Welch M-step**:
```
π'_i  = γ_1(i)
a'_ij = Σ_t ξ_t(i,j) / Σ_j Σ_t ξ_t(i,j)
μ'_i  = Σ_t γ_t(i) x_t / Σ_t γ_t(i)
Σ'_i  = Σ_t γ_t(i)(x_t − μ'_i)(x_t − μ'_i)ᵀ / Σ_t γ_t(i) + εI
```

### Regime States & Position Sizing

| State | Economic Meaning | Kelly Size Multiplier |
|-------|----------------|-----------------------|
| **BULL** | Supply tightening, backwardation, positive returns | 0.8–1.0 (γ_BULL weighted) |
| **BEAR** | Oversupply, contango, negative returns | 0.8–1.0 (γ_BEAR weighted) |
| **VOLATILE** | Crisis — OPEC shock / geopolitical event, high vol | 0.25 (75% size reduction) |
| **SIDEWAYS** | Balanced demand/supply, range-bound, low vol | 0.50 (half size) |

---

## MAP-HMM Next-Bar Predictor (`hmm_regime.py`)

A second, independent Gaussian HMM runs alongside the regime engine to predict next-bar **price direction** using the **Maximum a Posteriori (MAP)** approach from:

> Gupta & Dhingra, *"Stock Market Prediction Using Hidden Markov Models"*, IEEE 2012

### Algorithm

**D=3 OHLC fractional features** (equation 3, paper):
```
fracChange = (Close − Open) / Open
fracHigh   = (High  − Open) / Open
fracLow    = (Open  − Low)  / Open
```

**MAP prediction formula**:
```
Ô_{d+1} = argmax_O  P(O₁, O₂, ..., O_d, O | λ)
```

**Efficient computation** (log-space, fully vectorised):
```
log_trans[j]    = lse_i[ log α_d(i) + log A(i,j) ]           ← precomputed once, O(N²)
log P(O_{d+1})  = lse_j[ log_trans[j] + log b_j(O_{d+1}) ]   ← O(N) per candidate
```

**5 000-point grid** (Table II, paper):

| Dimension | Range | Steps |
|-----------|-------|-------|
| fracChange | [−0.05, +0.05] | 50 |
| fracHigh | [0.0, 0.04] | 10 |
| fracLow | [0.0, 0.04] | 10 |

**Direction threshold**: `|fracChange| > 0.002` → UP / DOWN, else FLAT

### Integration

| File | MAP Usage |
|------|-----------|
| `signal_engine.py` | ±0.05 alignment bonus on ensemble score |
| `vsa_agents.py` | Agent 1 logs MAP direction alongside HMM trend state |
| `crew_agent.py` | `fetch_hmm_regime_context()` includes MAP fields |
| `autonomous_agent.py` | `--status` shows MAP prediction; `risk_monitor` logs it |
| `global_ecosystem.py` | ClaudeLeadershipAgent prompt enriched with MAP direction |

### Quick Usage

```python
from hmm_regime import get_hmm_regime

result = get_hmm_regime("CL=F")

# Regime (Baum-Welch)
print(result.regime)          # OilRegime.BULL
print(result.probabilities)   # {'BULL': 0.78, ...}

# MAP next-bar prediction (Gupta & Dhingra 2012)
print(result.map_direction)   # "UP"
print(result.map_frac_change) # +0.0072
print(result.map_explanation) # "[MAP-HMM] Next-bar prediction: UP ..."
```

---

## Fallon Likelihood-Similarity Predictor (`hmm_regime.py`)

Third HMM predictor based on:

> Fallon, *"Making Profit in the Stock Market Using HMMs"*, University of Massachusetts Lowell (2012)  
> — Achieved **26%+ profit** over one year trading 10 stocks

### Algorithm

**D=1 observation**: daily close-to-close fractional return ≈ (Close − Open) / Open

**4 hidden states**: HIGH_INCREASE | LOW_INCREASE | LOW_DECREASE | HIGH_DECREASE

**Likelihood-similarity nearest-neighbour** (rolling 20-day window):
```
1. Compute log P(O_{t−19}...O_t | λ) for every historical day
2. Find the historical window with the closest log-likelihood to today
3. Predicted next-day return = actual return that followed that window
4. BUY if predicted_return > 0 else SKIP
```

### Integration

| File | Fallon Usage |
|------|-------------|
| `signal_engine.py` | +0.03 alignment bonus when Fallon=BUY and ensemble>0 |
| `vsa_agents.py` | Agent 1 logs `fallon_signal` every 15 min |
| `crew_agent.py` | `fetch_hmm_regime_context()` includes Fallon fields |
| `autonomous_agent.py` | `--status` shows Fallon signal; `risk_monitor` logs it |
| `global_ecosystem.py` | ClaudeLeadershipAgent prompt enriched with Fallon direction |

### Quick Usage

```python
from hmm_regime import get_hmm_regime

result = get_hmm_regime("CL=F")

# Fallon likelihood-similarity signal
print(result.fallon_direction)        # "BUY"
print(result.fallon_predicted_return) # +0.0083
print(result.fallon_explanation)      # "[Fallon-HMM] BUY | pred_return=+0.0083 ..."
```

---

### Why HMM Beats Simple Moving Averages

1. **Regime persistence** — the transition matrix A captures that markets don't jump between states instantly
2. **Soft assignment** — γ_t(i) posteriors give probability weights for Kelly position sizing
3. **Multivariate** — D=4 features capture correlated price+volume+momentum structure
4. **Data-driven** — parameters optimised from actual WTI data via EM (no hand-tuned thresholds)
5. **Mathematically guaranteed** — Baum-Welch monotonically improves p(X|Λ) each iteration

### Integration Across All Agents

```
hmm_regime.py
  ├── signal_engine.py      → replaces _long_term_trend_regime() (50/200MA heuristic)
  ├── vsa_agents.py         → Agent 1 (Macro Trend) uses get_hmm_regime() every 15 min
  ├── crew_agent.py         → fetch_hmm_regime_context() fed to IngestionOfficer + RiskOfficer
  ├── autonomous_agent.py   → risk_monitor() refreshes every 5 min; shown in --status
  └── global_ecosystem.py   → ClaudeLeadershipAgent prompt enriched; VOLATILE blocks ML entries
```

### Quick Usage

```python
from hmm_regime import get_hmm_regime, regime_size_multiplier

# Fit and classify current WTI regime
result = get_hmm_regime(ticker="CL=F", close=wti_close_series)

print(result.regime)          # OilRegime.BULL
print(result.probabilities)   # {'BULL': 0.78, 'BEAR': 0.12, 'VOLATILE': 0.06, 'SIDEWAYS': 0.04}
print(result.explanation)     # "BULL: strong carry & positive return (P=0.78)"

# Kelly position size multiplier
mult = regime_size_multiplier(result)   # 0.87
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
