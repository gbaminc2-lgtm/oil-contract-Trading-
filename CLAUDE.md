# CLAUDE.md — Quant Energy Commodity Pipeline

## Project Role
You are operating as an **Expert Quantitative Commodity Strategist** (Institutional Grade).  
This codebase implements a full institutional options, futures, and commodity derivatives  
trading system for WTI crude oil, Brent, RBOB, and ULSD.

---

## Architecture — 4-File Split

| File | Responsibility | Do Not Modify |
|------|---------------|---------------|
| `data_agent.py` | Market data: spot prices, forward curves, GARCH, crack spreads, storage economics, IV surface, ML features | Ticker constants, contract specs |
| `strategy_agent.py` | Black-76 / BSM engine, Monte Carlo (GBM + OU), NAV model, all 5 trade strategies, Three-Statement model, PPM generator | Pricing formulae |
| `risk_agent.py` | **ALL hardcoded risk limits**, VaR (parametric/historical/MC), ES/CVaR, stress tests, Greeks limits, Basel III capital | Section 1 constants |
| `main.py` | 10-step orchestration pipeline, CLI entry point, demo modes | Pipeline sequence |

---

## Critical Constraints for Claude Code

### NEVER DO:
- Override or modify hardcoded risk parameters in `risk_agent.py` Section 1
- Remove the pre-trade approval gate (`evaluate_trade`) from the pipeline
- Change Black-76 to BSM for energy/futures options pricing
- Add `execute_trade()` or any live order-routing functions
- Bypass the `ApprovalStatus.REJECTED` gate
- Introduce `time.sleep()` loops or infinite polling

### ALWAYS DO:
- Use `black76()` (not `black_scholes()`) for WTI/Brent/RBOB/ULSD options
- Lag all ML features by ≥1 period (no look-ahead bias)
- Run `evaluate_trade()` before appending any signal to `approved[]`
- Keep all dollar values in USD
- Keep `q=r` when calling `black_scholes()` for futures (Black-76 convention)
- Use `FUND_EQUITY_USD` from `risk_agent.py` as the single source of account size

---

## Pricing Conventions

```python
# Energy options (WTI, Brent, RBOB, ULSD) — ALWAYS use Black-76
greeks = black76(F=futures_price, K=strike, T=dte/365, r=r, sigma=iv, right=OptionRight.CALL)

# Equity options only — use BSM with dividend yield
greeks = black_scholes(S=spot, K=strike, T=dte/365, r=r, sigma=iv, right=OptionRight.CALL, q=div_yield)

# Futures options (Black-76 = BSM with q=r, underlying=F)
# q=r collapses forward price to F: S·exp((r-q)·T) = F
```

---

## Running the Pipeline

```bash
# Full pipeline (all 5 strategies + PPM + stress tests)
python main.py

# Black-76 Greeks table demo
python main.py --bsm-demo --spot 72.0 --sigma 0.32

# Monte Carlo (GBM vs Ornstein-Uhlenbeck)
python main.py --mc-demo

# Stress test report only (8 named scenarios)
python main.py --stress-only

# Full PPM (Private Placement Memorandum)
python main.py --ppm

# Risk state report
python main.py --risk-report

# Brent crude
python main.py --ticker BZ=F --fund "Brent Alpha Fund"
```

---

## Dependencies

```bash
pip install yfinance pandas numpy scipy scikit-learn requests
```

All imports have graceful degradation (`_YF`, `_SKL`, `_REQ` guards).  
The system runs fully offline with synthetic data fallbacks.

---

## Data Flow

```
data_agent.py
  ├── fetch_spot()            → float
  ├── fetch_forward_curve()   → ForwardCurve (contango/backwardation)
  ├── fetch_crack_spreads()   → CrackSpread (3-2-1, 5-3-2)
  ├── fetch_storage_economics()→ StorageEconomics (Cushing arb)
  ├── fetch_garch_vol()       → HistoricalVolRegime (GARCH 1,1)
  ├── fetch_seasonal_pattern()→ SeasonalPattern (12-month factors)
  ├── fetch_iv_surface()      → Dict[(dte,moneyness) → IV]
  └── build_ml_features()     → MLSignalData (GBM features, lagged)

strategy_agent.py
  ├── black76() / black_scholes()  → Greeks (Δ,Γ,Θ,V,ρ,vanna,charm)
  ├── monte_carlo_gbm()            → MCSimResult
  ├── monte_carlo_ou()             → MCSimResult (mean-reverting)
  ├── generate_storage_arb()       → TradeSignal | None
  ├── generate_crack_spread_signal()→ TradeSignal
  ├── generate_basis_trade()       → TradeSignal
  ├── generate_futures_signal()    → TradeSignal
  ├── generate_options_signal()    → TradeSignal (Black-76)
  ├── build_nav_model()            → NAVModel
  ├── build_three_statement()      → ThreeStatementModel
  └── generate_ppm_sections()      → str (PPM report)

risk_agent.py
  ├── evaluate_trade()        → RiskAssessment (APPROVED/FLAGGED/REJECTED)
  ├── parametric_var()        → float
  ├── historical_var()        → float
  ├── monte_carlo_var()       → float  (Student-t fat tails)
  ├── expected_shortfall()    → float  (Basel III / FRTB)
  ├── run_stress_tests()      → List[StressTestResult]
  ├── check_greeks_limits()   → List[str]
  ├── basel_iii_capital()     → Dict
  └── performance_report()    → str

main.py  (orchestrator)
  └── run_full_pipeline()     → dict (data, approved, nav, mc, ppm)
```

---

## Knowledge Base (21 Books)

| Domain | Books |
|--------|-------|
| Oil Trading | Oil Trader Academy, Hedging Strategies in Crude Oil Futures, Hedging Strategies for Oil End-Users, Commodities Demystified (Trafigura), Oil Contracts (Petroleum Contract), NYMEX Chapter 200 |
| Energy Derivatives | Modelling Energy Markets & Pricing Derivatives (Gkinis), CME Customer Center Manual |
| Options | Trading Options as a Professional (Bittman), How to Price and Trade Options (Sherbin), Options Strategies (Danes) |
| Risk | Risk Management & Financial Institutions (Hull 4th Ed.), Handbook of Risk (IMCA/Wiley) |
| Algo/Quant | Successful Algorithmic Trading (QuantStart) |
| Technical | Art & Science of Technical Analysis (Grimes), Technical Trading Tactics |
| Psychology | Mastering Trading Psychology (Aziz) |
| Day Trading | Complete Guide to Day Trading (Heitkoetter), Master the Markets (VSA) |

---

## Hardcoded Risk Limits (risk_agent.py Section 1)

| Parameter | Value | Source |
|-----------|-------|--------|
| Fund equity | $10,000,000 | — |
| Max risk / trade | 1% ($100K) | Hull Ch.16 |
| Max portfolio heat | 8% ($800K) | IMCA |
| Max WTI contracts | 50 lots | NYMEX Ch.200 |
| Max 1d 99% VaR | $250,000 | Basel III |
| Max daily loss | $100,000 | Heitkoetter |
| Max drawdown gate | 15% | PPM constraint |
| Min DTE (options) | 21 days | Bittman |
| Max net delta | ±20,000 bbls | Bittman Ch.12 |
| Max net vega | ±$50,000/1%IV | Sherbin |

---

## Git Workflow

```bash
# Feature branch → PR → main
git checkout -b feature/strategy-name
# ... edit ...
git push origin feature/strategy-name
# Open PR → CI runs → merge after review
```

Never commit directly to `main`.  
Never commit API keys, account credentials, or live broker configs.
