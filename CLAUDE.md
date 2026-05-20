# CLAUDE.md — Quant Energy Commodity Pipeline

## Project Role
You are operating as an **Expert Quantitative Commodity Strategist** (Institutional Grade).  
This codebase implements a full institutional options, futures, and commodity derivatives  
trading system for WTI crude oil, Brent, RBOB, and ULSD.

---

## Architecture — 12-File Split

| File | Responsibility | Do Not Modify |
|------|---------------|---------------|
| `autonomous_agent.py` | **Master 24/7 AI controller** — coordinates ALL 8 agent tasks across 4 market phases | Phase schedule, session limits |
| `global_ecosystem.py` | **7-agent global system** — IB bracket orders, XGBoost ML, Claude leadership, OPEC/IEA scraper, SPAN margin, FastAPI webhook | `GLOBAL_EXCHANGE_REGISTRY`, agent topology |
| `crew_agent.py` | **CrewAI 4-agent trading team** — IngestionOfficer, FundamentalAnalyst, RiskOfficer, ExecutionBroker; ChromaDB memory, EIA API, Alpaca paper, SQLite ledger | Agent roles, evaluate_trade() gate, paper-only URL |
| `dashboard.py` | **Streamlit telemetry UI** — real-time PnL, Black-76 Greeks surface, latency profile, Discord/Slack alert simulator | DB schema paths |
| `data_agent.py` | Market data: spot prices, forward curves, GARCH, crack spreads, storage economics, IV surface, ML features | Ticker constants, contract specs |
| `strategy_agent.py` | Black-76 / BSM engine, Monte Carlo (GBM + OU), NAV model, all 5 trade strategies, Three-Statement model, PPM generator | Pricing formulae |
| `risk_engine.py` | **ALL hardcoded risk limits**, VaR (parametric/historical/MC), ES/CVaR, stress tests, Greeks limits, Basel III capital | Section 1 constants |
| `main.py` | 10-step orchestration pipeline, CLI entry point, demo modes | Pipeline sequence |
| `vsa_agents.py` | Async 4-agent VSA team: trend filter, sharpshooter scanner, context/execution, quant risk | `VSA_THRESHOLDS` keys, agent topology |
| `micro_futures.py` | Micro & E-mini energy futures agent: SMA crossover, $5K/day target, backtrader cerebro backtest | `INSTRUMENTS` specs, daily target constant |
| `hmm_regime.py` | **Baum-Welch HMM** — 4-state Gaussian HMM (BULL/BEAR/VOLATILE/SIDEWAYS) fitted to WTI features via EM; outputs soft posteriors γ_t(i) for position sizing | Forward/backward math, N=4 states, D=4 features, log-space numerics |
| `market_architecture.py` | **Market Microstructure Math Engine** — 4-phase: microwave/fiber latency, contract notional/tick, BSM (equity only), 1%-rule position sizing & parametric VaR | `EXCHANGE_ROUTES`, `ALL_CONTRACTS`, `MCL_SPEC` values; Phase 3 is equity only — energy options MUST use `black76()` |

---

## Critical Constraints for Claude Code

### NEVER DO:
- Override or modify hardcoded risk parameters in `risk_engine.py` Section 1
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
- Use `FUND_EQUITY_USD` from `risk_engine.py` as the single source of account size

---

## HMM Regime Engine (`hmm_regime.py`)

### Baum-Welch Algorithm (ECE 417 / Hasegawa-Johnson, 2021)

| Concept | Implementation |
|---------|---------------|
| Hidden states N=4 | BULL, BEAR, VOLATILE, SIDEWAYS |
| Observation D=4 | daily_return%, rolling_vol%, rsi_norm, log_volume_norm |
| Emission | Gaussian b_i(x) = N(x; μ_i, Σ_i) |
| E-step | γ_t(i) = α_t(i)β_t(i) / Σ_k α_t(k)β_t(k) |
| M-step | Re-estimate π, A, μ, Σ via weighted MLE |
| Numerics | Log-space throughout (prevents underflow) |
| Retraining | Every 63 bars (~quarterly) with `_RETRAIN_EVERY` |

### HMM Position Size Multipliers

| Regime | Multiplier | Rationale |
|--------|-----------|-----------|
| BULL | 0.8–1.0 (γ_BULL weighted) | High confidence trend → full Kelly |
| BEAR | 0.8–1.0 (γ_BEAR weighted) | Clear downtrend → full short sizing |
| VOLATILE | 0.25 | Crisis/OPEC shock → 75% size reduction |
| SIDEWAYS | 0.50 | Range-bound → half size |

### HMM Integration Points

| File | HMM Usage |
|------|-----------|
| `signal_engine.py` | Replaces 50/200MA heuristic in `_long_term_trend_regime()` |
| `vsa_agents.py` | Agent 1 (Macro Trend) uses `get_hmm_regime()` every 15 min |
| `crew_agent.py` | `fetch_hmm_regime_context()` injected into Ingestion + Risk tasks |
| `autonomous_agent.py` | `risk_monitor()` refreshes every 5 min; shown in `--status` |
| `global_ecosystem.py` | ClaudeLeadershipAgent prompt enriched; VOLATILE blocks ML entries |

### MAP-HMM Next-Bar Predictor (Gupta & Dhingra, IEEE 2012)

A second, independent D=3 HMM runs alongside the regime HMM to predict next-bar direction.

| Concept | Implementation |
|---------|---------------|
| Paper | "Stock Market Prediction Using Hidden Markov Models," IEEE 2012 |
| Features D=3 | fracChange=(C-O)/O, fracHigh=(H-O)/O, fracLow=(O-L)/O |
| MAP formula | `Ô_{d+1} = argmax_O P(O₁,...,O_d,O\|λ)` |
| Grid | 50 × 10 × 10 = 5 000 candidates (vectorised via meshgrid + logsumexp) |
| Direction | fracChange > 0.002 → UP; < −0.002 → DOWN; else FLAT |
| Cache | Separate `_ohlc_hmm_model`, retrains every `_RETRAIN_EVERY=63` bars |

**MAP computation (log-space)**:
```
log_trans[j]  = lse_i[ log α_d(i) + log A(i,j) ]          ← O(N²) once
log P(O_{d+1}) = lse_j[ log_trans[j] + log b_j(O_{d+1}) ]  ← O(N) per candidate
```

**`RegimeResult` MAP fields** (added with defaults — backward compatible):
- `map_direction`   — "UP" | "DOWN" | "FLAT"
- `map_frac_change` — best fracChange from 5 000-point grid
- `map_explanation` — formatted string with logP and Gupta & Dhingra attribution

**Integration — MAP direction used in**:

| File | Usage |
|------|-------|
| `signal_engine.py` | ±0.05 alignment bonus on ensemble score when MAP agrees |
| `vsa_agents.py` | Agent 1 logs `map_direction` and `map_frac_change` |
| `crew_agent.py` | `fetch_hmm_regime_context()` includes MAP fields |
| `autonomous_agent.py` | `--status` shows MAP prediction; `risk_monitor` logs MAP |
| `global_ecosystem.py` | ClaudeLeadershipAgent prompt includes MAP direction |

### Fallon Likelihood-Similarity Predictor (Fallon, UMass Lowell, 2012)

A third, independent D=1 HMM runs to generate a BUY/SKIP trading signal via nearest-neighbor likelihood matching.

| Concept | Implementation |
|---------|---------------|
| Paper | "Making Profit in the Stock Market Using HMMs," UMass Lowell (2012) |
| Feature D=1 | close-to-close fractional return (approximates Fallon's (close−open)/open) |
| States N=4 | HIGH_INCREASE, LOW_INCREASE, LOW_DECREASE, HIGH_DECREASE |
| Algorithm | Rolling 20-day log P(window\|λ) → find nearest historical day → use its next-day return |
| Signal | BUY if predicted_return > 0; SKIP if ≤ 0 |
| Cache | Separate `_fallon_hmm_model`, retrains every `_RETRAIN_EVERY=63` bars |
| Validation | Fallon achieved 26%+ profit over 1 year on 10 stocks using this method |

**`RegimeResult` Fallon fields** (added with defaults — backward compatible):
- `fallon_direction`        — "BUY" | "SKIP"
- `fallon_predicted_return` — nearest-neighbour next-day return prediction
- `fallon_explanation`      — formatted string with logL and Fallon attribution

**Integration — Fallon signal used in**:

| File | Usage |
|------|-------|
| `signal_engine.py` | +0.03 alignment bonus on ensemble score when Fallon=BUY and ensemble>0 |
| `vsa_agents.py` | Agent 1 logs `fallon_signal` alongside HMM and MAP |
| `crew_agent.py` | `fetch_hmm_regime_context()` includes Fallon fields |
| `autonomous_agent.py` | `--status` shows Fallon signal; `risk_monitor` logs it |
| `global_ecosystem.py` | ClaudeLeadershipAgent prompt includes Fallon direction |

### HMM Constraints for Claude Code

#### NEVER DO:
- Change N (number of states) without updating `OilRegime` enum and all integration points
- Remove the `_RETRAIN_EVERY = 63` cache — retraining every bar would over-fit noise
- Use HMM posteriors to directly set dollar position sizes — always multiply through Kelly
- Bypass the `_HMM` guard — all 5 integration files have graceful fallback to MA heuristic
- Change the MAP grid dimensions (50×10×10) without re-validating direction thresholds
- Merge the OHLC HMM (`_ohlc_hmm_model`) with the regime HMM — they have different D
- Merge the Fallon HMM (`_fallon_hmm_model`) with either other model — D=1 vs D=3 vs D=4

#### ALWAYS DO:
- Keep log-space forward/backward to prevent float underflow on long series
- Add `1e-6 * np.eye(D)` covariance regularization in M-step (prevents singular Σ)
- Return `RegimeResult` from `get_hmm_regime()` — callers depend on `.regime`, `.probabilities`, `.explanation`, `.map_direction`, `.map_frac_change`, `.fallon_direction`, `.fallon_predicted_return`
- Source `regime_size_multiplier()` from `hmm_regime.py` — do not re-implement in other modules
- Treat `regime_size_multiplier()` as returning `float` — it no longer returns a dict

---

## Market Architecture Math Engine (`market_architecture.py`)

### Four-Phase Calculator

| Phase | Function | Purpose |
|-------|----------|---------|
| 1 — Latency | `calculate_latency_advantage(distance_km)` | Microwave vs fiber µs advantage per route |
| 1 — All routes | `all_exchange_latencies()` | Dict of all `EXCHANGE_ROUTES` with computed µs values |
| 2 — Notional | `calculate_notional_and_tick(price, multiplier, tick_size)` | Contract exposure + $/tick |
| 2 — Shortcut | `contract_notional(symbol, price, contracts)` | Looks up `ALL_CONTRACTS` by symbol |
| 2 — Crack | `calculate_crack_spread(crude, gasoline, heating_oil)` | 3:2:1 spread $/bbl |
| 3 — BSM | `black_scholes_call(S, K, T, r, sigma)` | **EQUITY ONLY** — call price + Greeks |
| 4 — Sizing | `calculate_position_size(balance, risk_pct, stop_ticks, tick_value)` | 1%-rule max contracts |
| 4 — VaR | `calculate_parametric_var(portfolio_value, daily_mean, daily_vol, confidence, days)` | Parametric VaR |

### EXCHANGE_ROUTES (Phase 1)

| Key | Route | Distance |
|-----|-------|---------|
| `NYC_CHICAGO` | NYSE → CME/NYMEX (WTI CL, S&P ES) | 1 180 km |
| `CHICAGO_LONDON` | CME → ICE London (Brent BRN, TTF) | 7 500 km |
| `NYC_HOUSTON` | NYSE → Cushing physical hub | 2 200 km |
| `LONDON_FRANKFURT` | ICE → Eurex (European equity derivatives) | 650 km |

### ALL_CONTRACTS (Phase 2)

| Symbol | Name | Multiplier | Tick Size |
|--------|------|-----------|----------|
| `CL` | WTI Crude Oil | 1 000 bbl | $0.01/bbl |
| `MCL` | Micro WTI | 100 bbl | $0.01/bbl |
| `BRN` | Brent Crude | 1 000 bbl | $0.01/bbl |
| `RB` | RBOB Gasoline | 42 000 gal | $0.0001/gal |
| `HO` | Heating Oil (ULSD) | 42 000 gal | $0.0001/gal |

### Integration Points

| File | MAM Usage |
|------|-----------|
| `data_agent.py` | `fetch_exchange_latency()` calls `all_exchange_latencies()` |
| `risk_engine.py` | `mam_position_size()` cross-checks sizing via Phase 4 |
| `micro_futures.py` | `size_for_daily_target()` uses MAM sizing cross-check |
| `strategy_agent.py` | `generate_crack_spread_signal()` uses `calculate_crack_spread()` as reference |
| `vsa_agents.py` | Agent 4 uses `calculate_position_size()` as secondary validation |
| `autonomous_agent.py` | `--status` shows Phase-1 latency table for all routes |
| `crew_agent.py` | `fetch_market_arch_context()` included in IngestionOfficer + RiskOfficer tasks |
| `global_ecosystem.py` | `exchange_latency` injected into ClaudeLeadershipAgent context |
| `main.py` | `--market-arch` flag runs full 4-phase demo |

### Module Singleton

```python
from market_architecture import get_market_arch
mam = get_market_arch()   # lazily initialised, shared instance

lat  = mam.all_exchange_latencies()["NYC_CHICAGO"]
# lat["microwave_microseconds"], lat["fiber_microseconds"], lat["advantage_microseconds"]

pos  = mam.calculate_position_size(500, 2.0, 20, 1.00)
# pos["max_contracts"], pos["max_loss_allowed_usd"], pos["risk_per_contract_usd"]
```

### Market Architecture Math Constraints for Claude Code

#### NEVER DO:
- Use `black_scholes_call()` from `market_architecture.py` for energy options — it is **EQUITY ONLY**
- Modify `EXCHANGE_ROUTES` distances or `ALL_CONTRACTS` multipliers/tick sizes — sourced from CME/ICE/NYMEX specs
- Let MAM sizing override the hard `MAX_WTI_CONTRACTS` cap in `risk_engine.py`
- Use `MCL_SPEC["multiplier"]` (100 bbl) for standard CL contracts — they are separate entries

#### ALWAYS DO:
- Use `black76()` from `strategy_agent.py` for all WTI/Brent/RBOB/ULSD options pricing
- Treat MAM position sizing as a **cross-check**, never the primary gate — `evaluate_trade()` is always primary
- Source account equity from `risk_engine.ACCOUNT_EQUITY_USD`, not from a MAM constant
- Call `get_market_arch()` (singleton) — never instantiate `MarketArchitectureMath()` directly in integration code
- Keep `run_demo()` as the only entry point from `main.py --market-arch` — do not call `if __name__ == "__main__"` block from other modules

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

## Python Setup (Windows)

### Install Python 3.12 (one-time)
```powershell
# Option A — Windows Package Manager (recommended, run in PowerShell as admin)
winget install --id Python.Python.3.12 --source winget --accept-package-agreements --accept-source-agreements

# Option B — Direct download
# https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe
# During install: check "Add Python to PATH" and "Add Python to environment variables"
```

Python 3.12 installs to:
```
C:\Users\alvin\AppData\Local\Programs\Python\Python312\python.exe
C:\Users\alvin\AppData\Local\Programs\Python\Python312\Scripts\pip.exe
```
These paths are pre-wired into Claude Code's global settings (`~/.claude/settings.json`).  
After installing Python, Claude Code can run `python` and `pip` commands without any PATH changes.

### Verify installation
```powershell
python --version          # Python 3.12.x
pip --version             # pip 24.x
```

### Disable Microsoft Store Python stub (if python still opens Store)
```powershell
# Settings → Apps → Advanced app settings → App execution aliases
# Toggle OFF: python.exe and python3.exe
```

## Dependencies

```bash
# Core pipeline (all 10 modules)
pip install yfinance pandas numpy scipy scikit-learn requests loguru pydantic backtrader python-dotenv

# Global ecosystem additions (global_ecosystem.py)
pip install xgboost beautifulsoup4 fastapi uvicorn ib_insync anthropic

# CrewAI trading team + dashboard (crew_agent.py + dashboard.py)
pip install crewai langchain-anthropic langchain-openai langchain-community
pip install chromadb pypdf alpaca-trade-api feedparser streamlit plotly
```

All imports have graceful degradation guards:

| Guard | Package | Fallback if missing |
|-------|---------|---------------------|
| `_YF` | `yfinance` | Synthetic price data |
| `_SKL` | `scikit-learn` | Disabled ML features |
| `_BT` | `backtrader` | No cerebro backtest |
| `_REQ` | `requests` | No HTTP scraping |
| `_IB` | `ib_insync` | No IB bracket orders |
| `_XGB` | `xgboost` | LinearRegression fallback |
| `_BS4` | `beautifulsoup4` | No OPEC/IEA scraping |
| `_FASTAPI` | `fastapi` + `uvicorn` | No webhook server |
| `_ANTHROPIC` | `anthropic` | No Claude leadership agent |
| `_CREW` | `crewai` | No CrewAI team (crew_agent.py gracefully exits) |
| `_LC_ANTHROPIC` | `langchain-anthropic` | Falls back to OpenAI LLM |
| `_CHROMA` | `chromadb` | Agents run without vector memory |
| `_PDF` | `pypdf` | No PDF ingestion |
| `_ALPACA` | `alpaca-trade-api` | Paper orders simulated locally |
| `_FEEDPARSER` | `feedparser` | Mock RSS headlines |

The core 6-module pipeline (`data_agent` → `strategy_agent` → `risk_engine` → `main` → `vsa_agents` → `micro_futures`) runs fully offline.  
`global_ecosystem.py` and `crew_agent.py` each degrade gracefully — every optional dependency is guarded independently.

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

risk_engine.py
  ├── evaluate_trade()        → RiskAssessment (APPROVED/FLAGGED/REJECTED)
  ├── parametric_var()        → float
  ├── historical_var()        → float
  ├── monte_carlo_var()       → float  (Student-t fat tails)
  ├── expected_shortfall()    → float  (Basel III / FRTB)
  ├── run_stress_tests()      → List[StressScenario]
  ├── check_greeks_limits()   → List[str]
  ├── basel_iii_capital()     → Dict
  └── performance_report()    → str

main.py  (orchestrator)
  └── run_full_pipeline()     → dict (data, approved, nav, mc, ppm)

micro_futures.py  (micro/E-mini agent)
  ├── micro_futures_agent()   → async live agent (SMA crossover, yfinance feed)
  ├── run_backtest()          → backtrader cerebro result
  ├── SMACrossEngine          → pure-Python 10/30 SMA crossover engine
  ├── DailySession            → tracks P&L vs $5K target + loss circuit-breaker
  └── size_for_daily_target() → int  (position sizing by daily target)

vsa_agents.py  (VSA 4-agent system)
  ├── macro_trend_agent()     → async, sets LONG_ONLY|SHORT_ONLY|FLAT every 15 min
  ├── vsa_sharpshooter_agent()→ async, noise filter → SOS/SOW signals
  ├── context_execution_agent()→ async, trend gate → OrderRequest
  └── quant_risk_agent()      → async, 1% sizing → execution stub

autonomous_agent.py  (master 24/7 controller)
  ├── run_autonomous()        → 24/7 asyncio loop — coordinates ALL 8 tasks
  ├── risk_monitor()          → async, 10s checks, enforces $100 loss limit + $5K target
  ├── pre_market_runner()     → async, 08:00–09:00 ET, runs full main.py pipeline
  ├── vsa_coordinator()       → async, launches/shuts VSA 4-agent team at market open/close
  ├── micro_coordinator()     → async, launches/shuts MCL micro agent at market open/close
  ├── ecosystem_coordinator() → async, launches global_ecosystem 7-agent system at market open
  ├── crew_coordinator()      → async, launches CrewAI 4-agent team every 30 min at MARKET_OPEN
  ├── post_market_reporter()  → async, 16:30 ET, session report + JSON archive → ./logs/
  ├── overnight_monitor()     → async, 30-min heartbeat during off-hours
  ├── AutoSession             → master shared state (P&L, phase, flat flag, agent list)
  └── MarketPhase             → PRE_MARKET | MARKET_OPEN | POST_MARKET | OVERNIGHT

crew_agent.py  (CrewAI 4-agent decentralized trading team)
  ├── run_crew_cycle()        → async, single ingest→analyze→risk→execute cycle
  ├── build_knowledge_base()  → ChromaDB seeded from CLAUDE.md + 16 institutional sources
  ├── fetch_eia_data()        → EIA V2 API weekly crude storage (live or mock)
  ├── fetch_rss_headlines()   → Reuters/CNBC RSS macro news feed
  ├── fetch_market_regime()   → Alpaca paper account USO/UNG 5-day closes
  ├── compute_black76_greeks()→ Black-76 Δ/Γ/Θ/V (strategy_agent or inline scipy)
  ├── submit_paper_order()    → Alpaca paper-api.alpaca.markets (evaluate_trade() gated)
  ├── get_crew_metrics()      → dict (daily_pnl, trade_count, win_rate, avg_latency_ms)
  ├── TelemetryNotifier       → Discord + Slack webhook dispatcher
  ├── _init_db()              → SQLite schema (options_portfolio, agent_decisions, telemetry)
  └── CrewCycleResult         → dataclass returned to autonomous_agent.crew_coordinator()

dashboard.py  (Streamlit real-time telemetry UI)
  ├── main()                  → Streamlit app (run: streamlit run dashboard.py)
  ├── Tab 1: Options Portfolio → positions, Greeks (Δ/Vega bar chart), PnL
  ├── Tab 2: Agent Decisions  → cognitive audit trail, market regime
  ├── Tab 3: Telemetry Logs   → latency line chart, CRITICAL anomaly alerts
  └── Tab 4: Alert Simulator  → inject test Discord/Slack webhooks

global_ecosystem.py  (7-agent international system)
  ├── start_global_ecosystem()          → async entry — IB connect + 7-agent launch loop
  ├── get_ecosystem_metrics()           → dict (bias, vol, P&L, cushion, lockout)
  ├── OPECIEAScraperAgent               → BeautifulSoup, OPEC.org + IEA.org → macro headlines
  ├── ClaudeLeadershipAgent             → Anthropic API (claude-opus-4-7), tool_use → MarketRegime
  ├── HighFrequencyMLQuantAgent         → XGBoost vol + LR slope → TradeSignal → evaluate_trade()
  ├── IBBracketOrderAgent               → ib_insync placeOrder (port 4002 paper) → bracket orders
  ├── SPANMarginAuditAgent              → 15% cushion gate; emergency flatten if breached
  ├── FXCurrencySweepAgent              → CNY/JPY/EUR cash → USD via IB Forex orders
  ├── FastAPIWebhookServer              → uvicorn async, POST /override, GET /status
  ├── SharedEcosystemMemoryBus          → inter-agent state: bias, vol, P&L, FX, lockout flag
  ├── GLOBAL_EXCHANGE_REGISTRY          → NYMEX MCL/CL, ICE Brent, INE SC, TOCOM JC, TTF
  └── maintenance_sentinel()            → exchange lockout 17:00–18:00 ET (CME Globex reset)
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

## Hardcoded Risk Limits (risk_engine.py Section 1)

| Parameter | Value | Source |
|-----------|-------|--------|
| Starting capital | $500.00 | User-defined |
| Daily profit target | $5,000 | Heitkoetter / user |
| Max risk / trade | 2% ($10.00) | Hull Ch.16 |
| Max portfolio heat | 10% ($50.00) | IMCA |
| Max WTI contracts | 1 lot (MCL micro) | NYMEX Ch.200 |
| WTI contract size | 100 bbl (MCL) | CME Micro spec |
| Max 1d 99% VaR | $50.00 | Basel III scaled |
| Max daily loss | $100.00 (20%) | Heitkoetter |
| Max drawdown gate | 15% | PPM constraint |
| Min DTE (options) | 21 days | Bittman |
| Max net delta | ±20.0 | Bittman Ch.12 |
| Max net vega | ±$500/1%IV | Sherbin |

---

## Micro & E-mini Energy Futures (`micro_futures.py`)

### Starting Capital: $500.00 | Daily Target: $5,000

With $500 starting capital, the system operates exclusively in **paper-trading / simulation mode** using MCL (Micro WTI, 100 bbl/contract). The $5,000/day target drives position sizing logic; all limits scale from the $500 equity base.

### Instruments — CME Spec

| Symbol | Name | Size | Tick | Tick Value | Margin | Status |
|--------|------|------|------|-----------|--------|--------|
| `MCL` | Micro WTI Crude Oil | 100 bbl/contract | $0.01/bbl | $1.00 | ~$1,000 | **Primary** |
| `MNG` | Micro Natural Gas | 2,500 MMBtu/contract | $0.001/MMBtu | $2.50 | ~$500 | Available |
| `QM` | E-mini Crude Oil | 500 bbl/contract | $0.025/bbl | $12.50 | ~$3,000 | Requires larger capital |

### Strategy — Dual-SMA Crossover (10/30) with Daily $5K Target

1. **SMA Crossover** — 10-period fast MA crosses above 30-period slow MA → LONG; below → SHORT
2. **VSA Volume Confirmation** — bars failing the noise filter (wash-trading) are skipped
3. **Risk Gate** — every signal passes through `evaluate_trade()` before execution
4. **Daily Target** — agent goes flat once cumulative P&L ≥ $5,000
5. **Daily Loss Breaker** — agent halts once cumulative loss ≥ `MAX_DAILY_LOSS_USD` (from `risk_engine`)

### Position Sizing Formula

```python
# $500 account — risk per trade = 2% = $10.00
risk_usd = ACCOUNT_EQUITY_USD * MAX_RISK_PER_TRADE_PCT   # $10.00

# Size by daily target (aspirational — paper-trade mode)
target_per_contract = expected_move_usd * contract_size_bbl   # e.g. $0.10 × 100 = $10
contracts = ceil(DAILY_TARGET_USD / target_per_contract)       # ceil($5,000 / $10) = 500

# Always capped by hard risk limits (result: 1 contract max at $500 equity)
contracts = min(contracts, MAX_WTI_CONTRACTS,
                int(ACCOUNT_EQUITY_USD * MAX_RISK_PER_TRADE_PCT / risk_per_contract))
```

### Running the Micro Futures Agent

```bash
# Async live paper-trading loop (yfinance MCL=F feed)
python micro_futures.py

# Backtrader cerebro backtest
python micro_futures.py --backtest

# E-mini crude (QM) — larger contract
python micro_futures.py --instrument QM

# Custom daily profit target
python micro_futures.py --target 5000

# Fixed contract override (ignores dynamic sizing)
python micro_futures.py --contracts 10

# Brent micro / natural gas micro
python micro_futures.py --instrument MNG
```

### Micro Futures Constraints for Claude Code

#### NEVER DO:
- Add live order execution (`execute_trade()`, broker API calls) inside `micro_futures.py`
- Modify `INSTRUMENTS` dict multipliers or tick values — sourced from CME/NYMEX Ch.200
- Change the daily target constant without also updating the position sizing formula
- Remove the `evaluate_trade()` gate from the async agent loop
- Use `time.sleep()` — use `asyncio.sleep()` only

#### ALWAYS DO:
- Keep `_BT` guard around all `backtrader` imports — it is an optional dependency
- Call `evaluate_trade()` before enqueuing any `TradeSignal`
- Source account equity from `risk_engine.ACCOUNT_EQUITY_USD` (never hardcode)
- Respect `DailySession.is_done` before generating new signals
- Apply the loss circuit-breaker check (`session.realized_pnl ≤ -MAX_DAILY_LOSS_USD`) each bar

---

## VSA Agent System (`vsa_agents.py`)

### Design Principles
- **Event-driven, zero-LLM-overhead** execution layer using Python `asyncio`
- Four agents communicate exclusively through `asyncio.Queue` objects — no shared mutable state in the hot path
- `SharedMarketState` is written only by Agents 1 and 2; Agent 3 reads it; Agent 4 never touches it

### 4-Agent Topology

| Agent | Name | Interval | Responsibility |
|-------|------|----------|---------------|
| 1 | Macro-Trend Filter | 15 min | Sets `LONG_ONLY \| SHORT_ONLY \| FLAT` from 4H/Daily structure |
| 2 | VSA Sharpshooter Scanner | Per tick | Volume noise filter → SOS / SOW signal detection |
| 3 | Context & Execution | 1 s poll | Trend-alignment gate → `OrderRequest` onto `order_queue` |
| 4 | Quant Risk Manager | Per order | 1% position sizing → execution stub (API placeholder) |

### VSA Signal Definitions (Master the Markets / Tom Williams)

| Signal | Conditions |
|--------|-----------|
| **SOS** (Sign of Strength) | Wide spread ≥ `WIDE_SPREAD_BBL` + volume ≥ `HIGH_VOLUME_LOTS` + close ≥ 80% of range |
| **SOW** (Sign of Weakness) | Wide spread ≥ `WIDE_SPREAD_BBL` + volume ≥ `HIGH_VOLUME_LOTS` + close ≤ 20% of range |

### Volume Noise Filter (Agent 2)
Strips wash-trading and spoofing **before** the Sharpshooter algorithm runs:
```
avg_trade_size = bar.volume / bar.trades_count
if bar.volume ≥ NOISE_MIN_VOL and avg_trade_size < NOISE_MAX_AVG_SZ → DISCARD
```

### Position Sizing Formula (Agent 4)
```python
risk_usd      = account_balance × MAX_RISK_PER_TRADE_PCT   # 2% ceiling = $10 at $500
risk_per_lot  = |entry − stop| × 100 bbl/contract          # MCL micro (NYMEX)
position_lots = risk_usd / risk_per_lot                     # typically 1 lot at $500
```
Account equity is sourced from `risk_engine.ACCOUNT_EQUITY_USD` (starting capital: $500.00).

### Running the VSA Agents

```bash
# Offline 3-bar demo (noise filter + SOS execution)
python vsa_agents.py --demo

# Custom bar count demo
python vsa_agents.py --demo --bars 5

# Live mode skeleton (wire up your data feed inside _run_live)
python vsa_agents.py
```

### VSA Constraints for Claude Code

#### NEVER DO (VSA-specific):
- Call `execute_trade()` or any live order-routing function inside this module
- Remove the Volume Noise Filter from Agent 2 — it runs before every Sharpshooter evaluation
- Modify the `SharedMarketState` from Agent 4 (read-only for the risk agent)
- Add `time.sleep()` anywhere — use `asyncio.sleep()` only
- Hardcode account equity — always read `ACCOUNT_EQUITY_USD` from `risk_engine`

#### ALWAYS DO (VSA-specific):
- Keep Agent 3 trend-alignment check before any order is enqueued
- Use `VSA_THRESHOLDS` dict for all Sharpshooter and noise-filter parameters
- Call `data_queue.task_done()` in Agent 2's `finally` block
- Call `order_queue.task_done()` in Agent 4's `finally` block
- Cancel all tasks gracefully in the orchestrator shutdown path

### WTI Threshold Calibration
Default `VSA_THRESHOLDS` values are calibrated for **WTI crude oil 1-minute bars**:

| Key | Default | Description |
|-----|---------|-------------|
| `WIDE_SPREAD_BBL` | 1.00 | Min $/bbl spread to qualify as wide |
| `HIGH_VOLUME_LOTS` | 300 | Min contract volume for high-volume bar |
| `NOISE_MIN_VOL` | 500 | Volume floor for noise filter activation |
| `NOISE_MAX_AVG_SZ` | 0.01 | Max avg trade size before bar is discarded |
| `CLOSE_TOP_RATIO` | 0.80 | Close position ratio for SOS (bullish) |
| `CLOSE_BOTTOM_RATIO` | 0.20 | Close position ratio for SOW (bearish) |
| `TREND_REFRESH_SEC` | 900 | Agent 1 refresh interval (15 min) |
| `CONTEXT_POLL_SEC` | 1 | Agent 3 polling interval |

---

## Autonomous AI Agent System (`autonomous_agent.py`)

### Overview
Single entry point for 24/7 fully autonomous operation. Owns the asyncio event loop and coordinates every other module through 6 concurrent tasks. No human intervention required after `python autonomous_agent.py`.

### 4-Phase Market Schedule (Eastern Time)

| Phase | Hours ET | Tasks Active | What Happens |
|-------|----------|-------------|--------------|
| `PRE_MARKET` | 08:00–09:00 | PreMarketRunner | Full main.py pipeline: data → ML → signals → risk approval → NAV → Monte Carlo → stress tests → PPM |
| `MARKET_OPEN` | 09:00–16:30 | VSACoordinator + MicroCoordinator + RiskMonitor | VSA 4-agent team + MCL micro futures agent run concurrently |
| `POST_MARKET` | 16:30–18:00 | PostMarketReporter | Daily P&L summary + session JSON archived to `./logs/` |
| `OVERNIGHT` | 18:00–08:00 | OvernightMonitor | 30-min heartbeat, system health check |

### 6 Autonomous Tasks (all run as asyncio coroutines)

| Task | Always On | Description |
|------|-----------|-------------|
| `RiskMonitor` | Yes | 10s checks — halts ALL agents when P&L ≤ −$100 or ≥ $5,000 |
| `PreMarketRunner` | Phase-gated | Runs main pipeline once per day at 08:00 ET |
| `VSACoordinator` | Phase-gated | Launches/cancels VSA 4-agent team at market open/close |
| `MicroCoordinator` | Phase-gated | Launches/cancels MCL micro futures agent at market open/close |
| `PostMarketReporter` | Phase-gated | Generates daily report + archives `./logs/session_DATE.json` |
| `OvernightMonitor` | Phase-gated | Heartbeat every 30 min with equity + phase status |

### Session State (`AutoSession`)
Shared across all tasks. Written only by RiskMonitor. Read by coordinators before each action.
- `flat_for_day`: set `True` when loss limit or profit target is breached — all agents stop
- `total_realized_pnl`: sum of micro + VSA P&L for the day
- `agents_running`: live list of currently active task names

### Running the Autonomous System

```bash
# 24/7 fully autonomous mode (all phases, all agents)
python autonomous_agent.py

# Offline demo — one full cycle, no API calls required
python autonomous_agent.py --demo

# Compressed 1-day simulation (validates all agent integrations)
python autonomous_agent.py --simulate

# Current system status — phase, account, agent readiness
python autonomous_agent.py --status
```

### Autonomous Agent Constraints for Claude Code

#### NEVER DO:
- Add `execute_trade()` or any live order-routing inside `autonomous_agent.py`
- Modify `ACCOUNT_EQUITY_USD`, `DAILY_TARGET_USD`, or `MAX_DAILY_LOSS_USD` at runtime
- Remove the `session.flat_for_day` check from any coordinator task
- Bypass the `risk_monitor()` task — it is always running and cannot be disabled
- Introduce `time.sleep()` — use `asyncio.sleep()` only

#### ALWAYS DO:
- Source all dollar limits from `risk_engine.py` (never hardcode in this file)
- Cancel all asyncio tasks in the `finally` block on shutdown
- Archive session state to `./logs/` at post-market and on graceful shutdown
- Keep `RiskMonitor` as the ONLY writer to `session.flat_for_day`
- Check `session.flat_for_day` at the top of every coordinator loop iteration

---

## Global Ecosystem Agents (`global_ecosystem.py`)

### Overview
7-agent international trading system wired into `autonomous_agent.py` via `ecosystem_coordinator()`. Connects to IB TWS Paper Trading (port 4002), scrapes OPEC/IEA for macro context, runs XGBoost ML signals, routes to Claude Opus for regime detection, and audits SPAN margin — all within the same asyncio event loop. Every trade signal passes `evaluate_trade()` before any IB order.

### Exchange Registry — `GLOBAL_EXCHANGE_REGISTRY`

| Key | Exchange | Symbol | Currency | Margin | Multiplier | $500 Eligible |
|-----|----------|--------|----------|--------|-----------|---------------|
| `NYMEX_MCL` | NYMEX | MCL | USD | $1,000 | 100 bbl | **Yes** |
| `NYMEX_CL` | NYMEX | CL | USD | $10,499 | 1,000 bbl | No |
| `ICE_BRENT` | ICE | BRN | USD | $11,200 | 1,000 bbl | No |
| `INE_SC` | INE | SC | CNY | ¥45,000 | 1,000 bbl | No |
| `TOCOM_JC` | TOCOM | JC | JPY | ¥420,000 | 50 kl | No |
| `ICE_TTF` | ICEENDEX | TTF | EUR | €4,250 | 744 MWh | No |

MCL is the only `account_eligible=True` instrument at $500 equity. All others produce 0 contracts via `floor(equity × 0.8 / margin) = 0` and return early without placing orders.

### 7-Agent Topology

| # | Agent | Data In | Output | Interval |
|---|-------|---------|--------|---------|
| 1 | `OPECIEAScraperAgent` | OPEC.org, IEA.org HTML | `sentiment_score` on `SharedEcosystemMemoryBus` | 60 min |
| 2 | `ClaudeLeadershipAgent` | Scraper sentiment + price slope | `macro_bias` (BULLISH/BEARISH/NEUTRAL) | 60 min |
| 3 | `HighFrequencyMLQuantAgent` | Price series → XGBoost + LR | `TradeSignal` → `evaluate_trade()` → IB queue | Per cycle |
| 4 | `IBBracketOrderAgent` | Approved `TradeSignal` queue | IB bracket order (entry + TP + SL) via port 4002 | Per signal |
| 5 | `SPANMarginAuditAgent` | IB portfolio positions | Emergency flatten if cushion < 15% | Per cycle |
| 6 | `FXCurrencySweepAgent` | IB Forex balances (CNY/JPY/EUR) | USD repatriation Forex orders | Per cycle |
| 7 | `FastAPIWebhookServer` | HTTP POST `/override`, GET `/status` | Runtime override commands + metrics JSON | Always-on |

### `SharedEcosystemMemoryBus` — Inter-Agent State

Written by Agents 1–3; read by Agents 4–6. Never written by Agent 4 (risk agent is read-only).

| Field | Type | Written By | Purpose |
|-------|------|-----------|---------|
| `macro_bias` | str | Agent 2 (Claude) | BULLISH / BEARISH / NEUTRAL |
| `sentiment_score` | float | Agent 1 (Scraper) | −1.0 (bearish) to +1.0 (bullish) |
| `target_volatility_prediction` | float | Agent 3 (XGBoost) | Forecast annualized vol |
| `current_risk_cushion_pct` | float | Agent 5 (SPAN) | Margin headroom % |
| `exchange_lockout` | bool | `maintenance_sentinel` | True during 17:00–18:00 ET CME reset |
| `daily_pnl_usd` | float | `record_trade()` | Running session P&L |
| `daily_flat` | bool | `record_trade()` | True when target or loss limit hit |
| `usd_to_cny` / `usd_to_jpy` / `usd_to_eur` | float | Agent 6 (FX) | Live FX rates for margin conversion |

### FastAPI Webhook Endpoints

```
POST /override    body: {"command": "FLATTEN_ALL" | "HALT" | "RESUME"}
GET  /status      returns: ecosystem metrics JSON (bias, vol, P&L, cushion, lockout)
```
Server runs on `0.0.0.0:8000` via `uvicorn.Server` inside the asyncio event loop (does not block agents).

### Maintenance Sentinel (`maintenance_sentinel()`)
Sets `SharedEcosystemMemoryBus.exchange_lockout = True` between 17:00–18:00 ET daily (CME Globex daily reset). All agents check `memory.exchange_lockout` and skip order generation during this window. Lockout clears automatically at 18:00 ET.

### Claude Leadership Agent — Regime Detection Tool
Uses `claude-opus-4-7` with `tool_use` (forced via `tool_choice={"type": "any"}`):

```python
_REGIME_TOOL = {
    "name": "set_market_regime",
    "input_schema": {
        "properties": {
            "bias":        {"type": "string", "enum": ["BULLISH","BEARISH","NEUTRAL"]},
            "confidence":  {"type": "number"},
            "rationale":   {"type": "string"},
        }
    }
}
```
Output `macro_bias` is written to `SharedEcosystemMemoryBus` and gating Agent 3 signal generation. BULLISH allows LONG-only; BEARISH allows SHORT-only; NEUTRAL skips new entries.

### Position Sizing (Agent 3 → Agent 4)

```python
allocation = ACCOUNT_EQUITY_USD * 0.80   # $400 at $500 account
contracts  = floor(allocation / local_margin_usd)   # MCL: floor(400/1000) = 0 → paper simulation
# evaluate_trade() gate is mandatory before placeOrder()
approved_qty = min(raw_qty, assessment.approved_qty, MAX_WTI_CONTRACTS)  # max = 1
```

At $500 equity even MCL produces 0 via this formula. The system runs as pure paper simulation — IB bracket orders are submitted in paper mode (port 4002) with 0-contract sizing until equity scales.

### Running the Global Ecosystem

```bash
# Full 7-agent ecosystem demo (no IB connection required)
python global_ecosystem.py --demo

# Live paper-trading mode (requires IB TWS on port 4002)
python global_ecosystem.py

# Via autonomous agent (recommended — coordinates with all other agents)
python autonomous_agent.py
```

### Global Ecosystem Constraints for Claude Code

#### NEVER DO:
- Change IB connection port from 4002 — this is TWS Paper Trading. Live port (7496) is forbidden
- Remove `evaluate_trade()` gate from `HighFrequencyMLQuantAgent` before any `placeOrder()`
- Modify `GLOBAL_EXCHANGE_REGISTRY` multipliers or margin values — sourced from CME/ICE/INE specs
- Hardcode allocation amounts — always use `ACCOUNT_EQUITY_USD * 0.80`
- Write to `SharedEcosystemMemoryBus` from Agent 4, 5, or 6 (read-only for these agents)
- Add `time.sleep()` — use `asyncio.sleep()` only; FastAPI uses `uvicorn.Server` directly
- Set `account_eligible=True` for any instrument with margin > `ACCOUNT_EQUITY_USD`

#### ALWAYS DO:
- Call `evaluate_trade()` and check `ApprovalStatus.REJECTED` before every IB `placeOrder()`
- Check `memory.exchange_lockout` and `memory.daily_flat` before generating signals
- Source all risk constants from `risk_engine.py` — never hardcode dollar limits here
- Keep `_IB`, `_XGB`, `_BS4`, `_FASTAPI`, `_ANTHROPIC` guards around all optional imports
- Repatriate non-USD cash to USD via FX sweep before end of session (Agent 6)
- Keep the `maintenance_sentinel()` coroutine running alongside agent tasks

---

## CrewAI Enterprise Trading Team (`crew_agent.py` + `dashboard.py`)

### Overview
Decentralized 4-agent architecture running every 30 minutes during `MARKET_OPEN`. Agents share a ChromaDB vector memory seeded from CLAUDE.md and 16 institutional sources. Every trade signal passes `evaluate_trade()` before any Alpaca paper order. Runs as the 8th asyncio task inside `autonomous_agent.py`.

### 4-Agent Topology

| # | Agent | Role | Output |
|---|-------|------|--------|
| 1 | `IngestionOfficer` | EIA API + RSS → structured matrix | Curve structure, inventory signal, geopolitical risk |
| 2 | `FundamentalAnalyst` | Black-76 options thesis | Strategy type, strikes, expiry, Greeks confirmed |
| 3 | `RiskOfficer` | evaluate_trade() gate + Greek limits | APPROVED or REJECTED manifest |
| 4 | `ExecutionBroker` | Alpaca paper order + SQLite log | Order confirmation, ledger record |

### Knowledge Base (ChromaDB)
16 institutional sources embedded as vectors at startup:

| Source | Domain |
|--------|--------|
| `CLAUDE.md` (always available) | All 21 books, all agent constraints, pricing conventions |
| EIA Volatility Framework | Inventory, storage, weekly petroleum status |
| FERC Market Primer | Regulatory, market structure |
| CME Customer Center | Contract specs, NYMEX Ch.200 |
| PwC Commodity Risk | Risk management protocols |
| EDHEC Risk Management | Greek limits, portfolio risk |
| JPM/Barclays Whitepapers | Macro outlook, hedge fund strategy |
| QuantStart AlgoTrading | Signal pipeline, ML features |
| Lacima Energy Derivatives | Energy options pricing theory |
| Houston Futures & Options | Black-76 vs BSM, options mechanics |
| + 6 more | WorldBank, RMI, DOE, Meketa, RevenueAI, Alvarez |

Agents query: `"Black-76 energy futures options, contango/backwardation, SPAN margin, crack spreads, NYMEX MCL specs"`

### SQLite Ledger (`logs/crew_trading_ledger.db`)

| Table | Purpose |
|-------|---------|
| `options_portfolio` | Active positions: Greeks, premium, PnL, risk status |
| `agent_decisions` | Full cognitive audit trail per agent |
| `system_telemetry` | Latency, component health, error log |
| `trade_logs` | Every approved/rejected trade with capital allocation |

### Running the CrewAI System

```bash
# Offline demo — no API keys required
python crew_agent.py --demo

# Readiness status check
python crew_agent.py --status

# Live 30-min cycle loop (requires ANTHROPIC_API_KEY in .env)
python crew_agent.py

# Real-time Streamlit dashboard (separate terminal)
streamlit run dashboard.py

# Via autonomous agent (recommended — 8th task, auto-coordinated)
python autonomous_agent.py
```

### Required `.env` Keys for Full CrewAI Operation

```bash
ANTHROPIC_API_KEY=sk-ant-...          # Claude Sonnet 4.6 LLM (primary)
OPENAI_API_KEY=sk-...                 # OpenAI GPT-4o (fallback LLM + embeddings)
EIA_API_KEY=your-eia-key              # eia.gov/developer — free registration
ALPACA_API_KEY=your-alpaca-key        # paper.alpaca.markets account
ALPACA_SECRET_KEY=your-secret         # paper account secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets   # NEVER change to live URL
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...   # optional
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...     # optional
```

### CrewAI Constraints for Claude Code

#### NEVER DO:
- Change `ALPACA_BASE_URL` to `https://alpaca.markets` — paper only, always
- Remove `evaluate_trade()` pre-screen from `run_crew_cycle()` before `submit_paper_order()`
- Use BSM (`black_scholes()`) for energy options in agent tasks — must use Black-76
- Hardcode dollar limits — always read `ACCOUNT_EQUITY_USD`, `MAX_RISK_PER_TRADE_PCT` from `risk_engine.py`
- Add `time.sleep()` — use `asyncio.sleep()` only; `crew.kickoff()` runs in `run_in_executor`
- Commit `OPENAI_API_KEY`, `ALPACA_API_KEY`, or `EIA_API_KEY` — all stay in `.env` (gitignored)

#### ALWAYS DO:
- Call `compute_black76_greeks()` (not BSM) for all energy options pricing in agent tasks
- Run `evaluate_trade()` pre-screen in `run_crew_cycle()` before building the crew
- Keep all 4 graceful-degradation guards (`_CREW`, `_LC_ANTHROPIC`, `_CHROMA`, `_ALPACA`)
- Log every cycle to SQLite — both APPROVED and REJECTED — for audit trail
- Check `session.flat_for_day` in `crew_coordinator()` before each 30-min cycle
- Use `run_in_executor` for `crew.kickoff()` — it is synchronous and must not block the event loop

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
