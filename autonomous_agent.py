"""
autonomous_agent.py — Master Autonomous AI Trading System Controller
=====================================================================
System Role: Expert Quantitative Commodity Strategist (Institutional Grade)

Knowledge sources:
  Complete Guide to Day Trading (Heitkoetter)  — session lifecycle, daily limits
  Successful Algorithmic Trading (QuantStart)   — pipeline / live parity
  Risk Management & Financial Institutions (Hull 4th Ed.) — autonomous risk gates
  Master the Markets (VSA / Tom Williams)       — agent coordination
  Art & Science of Technical Analysis (Grimes)  — MA crossover regime filter
  NYMEX Chapter 200 / CME Customer Center Manual — contract specs

Architecture:
  This module owns a single asyncio event loop and launches ALL other agents
  as coordinated asyncio tasks. It is the single entry point for 24/7
  autonomous operation of the full oil-contract trading pipeline.

  Phase 1 — PRE-MARKET  (08:00–09:00 ET):
    → Run full main.py pipeline: data ingestion, ML signal, signal generation,
      risk approval, NAV model, Monte Carlo, stress tests, PPM generation.
    → Cache approved signals for the market session.

  Phase 2 — MARKET OPEN (09:00–16:30 ET):
    → VSA 4-Agent Team      : macro trend + sharpshooter + execution + risk
    → Micro Futures Agent   : SMA crossover on MCL, $5K daily target
    → Risk Monitor          : continuous drawdown + daily-loss checks
    → Signal Broadcaster    : feeds pre-market approved signals to VSA team

  Phase 3 — POST-MARKET (16:30–18:00 ET):
    → Performance Reporter  : daily P&L summary, drawdown, approved signals
    → Session Archiver      : write JSON session log to ./logs/

  Phase 4 — OVERNIGHT   (18:00–08:00 ET):
    → Health Monitor        : heartbeat every 30 min, system state check

Hard constraints (CLAUDE.md / risk_engine.py):
  NEVER : add execute_trade() or live order routing
  NEVER : bypass evaluate_trade() gate on any signal
  NEVER : modify ACCOUNT_EQUITY_USD or DAILY_TARGET_USD at runtime
  ALWAYS: go flat when daily loss ≥ $100 OR daily P&L ≥ $5,000
  ALWAYS: source account equity from risk_engine.ACCOUNT_EQUITY_USD

Usage:
    python autonomous_agent.py              # 24/7 autonomous loop
    python autonomous_agent.py --demo       # one offline demo cycle (no API)
    python autonomous_agent.py --simulate   # 1-day compressed simulation
    python autonomous_agent.py --status     # print current system state and exit
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Load .env file if present (never committed — see .gitignore) ──────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

# ── Risk engine (single source of truth for all limits) ──────────────────────
try:
    from risk_engine import (
        ACCOUNT_EQUITY_USD,
        DAILY_TARGET_USD,
        MAX_DAILY_LOSS_USD,
        MAX_RISK_PER_TRADE_PCT,
        get_risk_summary,
        performance_report,
        record_pnl,
        ApprovalStatus,
    )
    _RISK = True
except ImportError:
    ACCOUNT_EQUITY_USD   = 500.0
    DAILY_TARGET_USD     = 5_000.0
    MAX_DAILY_LOSS_USD   = 100.0
    MAX_RISK_PER_TRADE_PCT = 0.02
    _RISK = False

# ── Global Ecosystem (7-agent IB/XGBoost/Claude/FastAPI system) ──────────────
try:
    from global_ecosystem import (
        start_global_ecosystem, get_ecosystem_metrics, global_memory as ecosystem_memory,
    )
    _ECOSYSTEM = True
except ImportError:
    _ECOSYSTEM = False

# ── CrewAI Trading Team (4-agent: Ingestion/Analyst/Risk/Execution) ───────────
try:
    from crew_agent import (
        run_crew_cycle, get_crew_metrics, build_knowledge_base,
        _init_db as _init_crew_db, CREW_CYCLE_SECS,
    )
    _init_crew_db()
    _CREW = True
except ImportError:
    _CREW = False

# ── VSA 4-Agent Team ─────────────────────────────────────────────────────────
try:
    from vsa_agents import (
        SharedMarketState, VSABar,
        macro_trend_agent, vsa_sharpshooter_agent,
        context_execution_agent, quant_risk_agent,
        VSA_THRESHOLDS,
    )
    _VSA = True
except ImportError:
    _VSA = False

# ── Micro Futures Agent ───────────────────────────────────────────────────────
try:
    from micro_futures import micro_futures_agent, DailySession, INSTRUMENTS
    _MICRO = True
except ImportError:
    _MICRO = False

# ── Main Pipeline ─────────────────────────────────────────────────────────────
try:
    from main import run_full_pipeline
    _PIPELINE = True
except ImportError:
    _PIPELINE = False

# ── Baum-Welch HMM regime (displayed in status + used by risk_monitor) ────────
get_hmm_regime:         Any = None
regime_size_multiplier: Any = None
OilRegime:              Any = None
try:
    from hmm_regime import get_hmm_regime, regime_size_multiplier, OilRegime
    _HMM = True
except ImportError:
    _HMM = False

try:
    from market_architecture import get_market_arch as _get_mam
    _MAM = True
except ImportError:
    _MAM = False

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)-8s] %(name)-22s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            LOG_DIR / f"autonomous_{datetime.date.today().isoformat()}.log",
            mode="a", encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("AutonomousController")


# ============================================================================
# SECTION 1 — MARKET PHASE CONTROLLER
# ============================================================================

class MarketPhase(str, Enum):
    PRE_MARKET  = "PRE_MARKET"   # 08:00–09:00 ET
    MARKET_OPEN = "MARKET_OPEN"  # 09:00–16:30 ET
    POST_MARKET = "POST_MARKET"  # 16:30–18:00 ET
    OVERNIGHT   = "OVERNIGHT"    # 18:00–08:00 ET


# Eastern Time offsets (no external lib needed)
# CME Energy Globex hours: Sun 5 PM – Fri 4 PM CT (= Sun 6 PM – Fri 5 PM ET)
# Most liquid: 09:00–14:00 ET. We use 08:00–16:30 ET as operational window.
_PHASE_SCHEDULE = {
    # (start_hour_et, end_hour_et): phase
    (8,  9 ): MarketPhase.PRE_MARKET,
    (9,  16): MarketPhase.MARKET_OPEN,
    (16, 18): MarketPhase.POST_MARKET,
}


def _et_offset() -> int:
    """UTC offset for US/Eastern in hours: -5 (EST) or -4 (EDT)."""
    now_utc = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    # Simplified DST: EDT from 2nd Sunday March to 1st Sunday November
    year = now_utc.year
    # 2nd Sunday in March
    mar = datetime.datetime(year, 3, 8)
    mar += datetime.timedelta(days=(6 - mar.weekday()) % 7)
    # 1st Sunday in November
    nov = datetime.datetime(year, 11, 1)
    nov += datetime.timedelta(days=(6 - nov.weekday()) % 7)
    return -4 if mar <= now_utc.replace(tzinfo=None) < nov else -5


def current_phase() -> MarketPhase:
    """Return the current market phase based on Eastern Time hour."""
    et_hour = (datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).hour + _et_offset()) % 24
    for (start, end), phase in _PHASE_SCHEDULE.items():
        if start <= et_hour < end:
            return phase
    return MarketPhase.OVERNIGHT


def et_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) + datetime.timedelta(hours=_et_offset())


# ============================================================================
# SECTION 2 — SESSION STATE (shared across all agents)
# ============================================================================

@dataclass
class AutoSession:
    """
    Master session state shared across all autonomous agents.
    Written only by the RiskMonitor; read by all other tasks.
    """
    date:               datetime.date  = field(default_factory=datetime.date.today)
    phase:              MarketPhase    = MarketPhase.OVERNIGHT
    flat_for_day:       bool           = False
    halt_reason:        str            = ""

    # P&L tracking (all agents feed into this)
    total_realized_pnl: float          = 0.0
    micro_pnl:          float          = 0.0
    vsa_pnl:            float          = 0.0
    pipeline_signals:   int            = 0
    approved_signals:   int            = 0
    rejected_signals:   int            = 0

    # Cycle counters
    pre_market_runs:    int            = 0
    post_market_runs:   int            = 0
    risk_checks:        int            = 0

    # System health
    last_heartbeat:     str            = ""
    agents_running:     List[str]      = field(default_factory=list)
    errors_today:       int            = 0

    # HMM regime + MAP prediction (updated by risk_monitor every cycle)
    hmm_regime:         str            = "UNKNOWN"
    hmm_size_mult:      float          = 1.0
    hmm_map_direction:   str            = "FLAT"   # UP | DOWN | FLAT (Gupta & Dhingra 2012)
    hmm_fallon_signal:   str            = "SKIP"   # BUY | SKIP (Fallon, UMass Lowell 2012)

    def record_pnl(self, pnl: float, source: str = "unknown") -> None:
        self.total_realized_pnl += pnl
        if source == "micro":
            self.micro_pnl += pnl
        elif source == "vsa":
            self.vsa_pnl += pnl
        if _RISK:
            record_pnl(pnl)

    def check_limits(self) -> Optional[str]:
        """Return halt reason if any daily limit is breached, else None."""
        if self.total_realized_pnl >= DAILY_TARGET_USD:
            return f"DAILY TARGET HIT — P&L ${self.total_realized_pnl:,.2f} >= ${DAILY_TARGET_USD:,.0f}"
        if self.total_realized_pnl <= -MAX_DAILY_LOSS_USD:
            return f"DAILY LOSS LIMIT — P&L ${self.total_realized_pnl:,.2f} <= -${MAX_DAILY_LOSS_USD:,.0f}"
        return None

    def summary_line(self) -> str:
        return (
            f"[Session {self.date}] Phase={self.phase.value} | "
            f"HMM={self.hmm_regime}(×{self.hmm_size_mult:.2f}|MAP={self.hmm_map_direction}"
            f"|Fallon={self.hmm_fallon_signal}) | "
            f"P&L=${self.total_realized_pnl:+,.2f} | "
            f"Target=${DAILY_TARGET_USD:,.0f} | LossLimit=${MAX_DAILY_LOSS_USD:,.0f} | "
            f"Approved={self.approved_signals} | Flat={self.flat_for_day}"
        )

    def to_json(self) -> str:
        d = asdict(self)
        d["phase"] = self.phase.value
        d["date"]  = self.date.isoformat()
        return json.dumps(d, indent=2)


# ============================================================================
# SECTION 3 — RISK MONITOR (always running)
# ============================================================================

async def risk_monitor(session: AutoSession, check_interval: float = 10.0) -> None:
    """
    Runs every 10 seconds. Enforces the daily loss limit and daily target.
    Sets session.flat_for_day = True when either limit is breached.
    All other agents poll session.flat_for_day before acting.

    Hull Ch.16: the risk monitor is the last line of defence.
    It must be independent of the signal generators.
    """
    logger.info("[RiskMonitor] Started — checking every %.0fs", check_interval)
    _hmm_refresh_counter = 0
    while True:
        try:
            session.risk_checks += 1
            session.last_heartbeat = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat()

            # ── Baum-Welch HMM regime refresh (every ~5 min = 30 × 10s) ─────
            _hmm_refresh_counter += 1
            if _HMM and _hmm_refresh_counter % 30 == 1:
                try:
                    import importlib
                    pd_mod = importlib.import_module("pandas")
                    yf_mod = importlib.import_module("yfinance")
                    raw = yf_mod.download("CL=F", period="1y", interval="1d",
                                          progress=False, auto_adjust=True)
                    if isinstance(raw.columns, pd_mod.MultiIndex):
                        raw.columns = [c[0].lower() for c in raw.columns]
                    else:
                        raw.columns = [c.lower() for c in raw.columns]
                    close_s = raw["close"].dropna()
                    if len(close_s) >= 63:
                        result = get_hmm_regime(ticker="CL=F", close=close_s)
                        session.hmm_regime        = result.regime.value
                        session.hmm_size_mult     = regime_size_multiplier(result)
                        session.hmm_map_direction = result.map_direction
                        session.hmm_fallon_signal = result.fallon_direction
                        logger.info(
                            "[RiskMonitor|HMM] %s size_mult=%.2f "
                            "MAP=%s(fc=%+.4f) Fallon=%s(ret=%+.4f) | %s",
                            session.hmm_regime, session.hmm_size_mult,
                            result.map_direction, result.map_frac_change,
                            result.fallon_direction, result.fallon_predicted_return,
                            result.explanation,
                        )
                        # VOLATILE crisis regime: warn loudly (size already reduced by HMM)
                        if session.hmm_regime == "VOLATILE" and not session.flat_for_day:
                            logger.warning(
                                "[RiskMonitor|HMM] VOLATILE REGIME — crisis conditions. "
                                "Kelly size_mult=%.2f applied by all agents.", session.hmm_size_mult
                            )
                except Exception as hmm_exc:
                    logger.debug("[RiskMonitor|HMM] Refresh skipped: %s", hmm_exc)

            halt_reason = session.check_limits()
            if halt_reason and not session.flat_for_day:
                session.flat_for_day = True
                session.halt_reason  = halt_reason
                logger.warning("[RiskMonitor] HALT — %s", halt_reason)

            if session.risk_checks % 18 == 0:   # log every ~3 minutes
                logger.info("[RiskMonitor] %s", session.summary_line())

            # Daily reset at midnight ET
            today = datetime.date.today()
            if session.date != today:
                logger.info("[RiskMonitor] New trading day — resetting session.")
                session.date               = today
                session.flat_for_day       = False
                session.halt_reason        = ""
                session.total_realized_pnl = 0.0
                session.micro_pnl          = 0.0
                session.vsa_pnl            = 0.0
                session.pipeline_signals   = 0
                session.approved_signals   = 0
                session.rejected_signals   = 0
                session.pre_market_runs    = 0
                session.post_market_runs   = 0
                session.errors_today       = 0

        except Exception as exc:
            logger.error("[RiskMonitor] Error: %s", exc)
            session.errors_today += 1

        await asyncio.sleep(check_interval)


# ============================================================================
# SECTION 4 — PRE-MARKET PHASE: Full Pipeline Runner
# ============================================================================

async def pre_market_runner(session: AutoSession) -> None:
    """
    Runs once per trading day during the pre-market window (08:00–09:00 ET).
    Executes the full main.py pipeline:
      data → ML → signals → risk gate → NAV → Monte Carlo → stress → PPM

    QuantStart: the research and live pipelines must be structurally identical.
    All signal generation and risk checking run the same code path regardless
    of whether the system is in paper or live mode.
    """
    while True:
        phase = current_phase()
        session.phase = phase

        if phase == MarketPhase.PRE_MARKET and session.pre_market_runs == 0:
            logger.info("[PreMarket] Initiating full pipeline run...")
            try:
                if _PIPELINE:
                    # Run the full institutional pipeline in a thread to avoid
                    # blocking the event loop during network I/O calls
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, lambda: run_full_pipeline(verbose=True)
                    )
                    approved = result.get("approved", [])
                    session.pipeline_signals  += len(approved)
                    session.approved_signals  += len(approved)
                    session.pre_market_runs   += 1
                    logger.info(
                        "[PreMarket] Pipeline complete — %d approved signals.",
                        len(approved),
                    )
                else:
                    logger.warning("[PreMarket] main.py pipeline unavailable — running demo signals.")
                    session.pre_market_runs += 1
            except Exception as exc:
                logger.error("[PreMarket] Pipeline error: %s", exc)
                session.errors_today += 1
                session.pre_market_runs += 1   # mark done to avoid retry loop

        await asyncio.sleep(60)   # check phase every minute


# ============================================================================
# SECTION 5 — MARKET HOURS: VSA 4-Agent Team Coordinator
# ============================================================================

async def vsa_coordinator(session: AutoSession) -> None:
    """
    Launches the 4 VSA agents during market hours.
    Shuts them down when session is flat or phase leaves MARKET_OPEN.
    Re-launches next bar batch when market reopens.

    Agents communicate through asyncio.Queue — zero shared-state mutations.
    SharedMarketState is written only by Agents 1 and 2.
    """
    if not _VSA:
        logger.warning("[VSACoordinator] vsa_agents.py unavailable — skipping VSA team.")
        return

    logger.info("[VSACoordinator] Waiting for MARKET_OPEN phase...")

    while True:
        phase = current_phase()

        if phase == MarketPhase.MARKET_OPEN and not session.flat_for_day:
            logger.info("[VSACoordinator] Launching VSA 4-Agent Team...")

            shared_state      = SharedMarketState()
            market_data_queue = asyncio.Queue()
            execution_queue   = asyncio.Queue()

            tasks = [
                asyncio.create_task(
                    macro_trend_agent(shared_state), name="VSA_Trend"
                ),
                asyncio.create_task(
                    vsa_sharpshooter_agent(shared_state, market_data_queue),
                    name="VSA_Sharpshooter",
                ),
                asyncio.create_task(
                    context_execution_agent(shared_state, execution_queue),
                    name="VSA_Context",
                ),
                asyncio.create_task(
                    quant_risk_agent(execution_queue, shared_state),
                    name="VSA_Risk",
                ),
            ]
            session.agents_running += ["VSA_Trend", "VSA_Sharpshooter", "VSA_Context", "VSA_Risk"]
            logger.info("[VSACoordinator] All 4 VSA agents running.")

            # Run until session is flat or market closes
            while True:
                if session.flat_for_day or current_phase() != MarketPhase.MARKET_OPEN:
                    logger.info("[VSACoordinator] Shutting down VSA agents — %s",
                                "flat for day" if session.flat_for_day else "market closed")
                    for t in tasks:
                        t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    for name in ["VSA_Trend", "VSA_Sharpshooter", "VSA_Context", "VSA_Risk"]:
                        if name in session.agents_running:
                            session.agents_running.remove(name)
                    break
                await asyncio.sleep(15)

        await asyncio.sleep(30)   # re-check phase every 30s


# ============================================================================
# SECTION 6 — MARKET HOURS: Micro Futures Agent Coordinator
# ============================================================================

async def micro_coordinator(session: AutoSession) -> None:
    """
    Launches micro_futures_agent during market hours.
    Instrument: MCL (Micro WTI, 100 bbl/contract) — matched to $500 account.
    Goes flat automatically when session.flat_for_day is True.
    """
    if not _MICRO:
        logger.warning("[MicroCoordinator] micro_futures.py unavailable — skipping.")
        return

    logger.info("[MicroCoordinator] Waiting for MARKET_OPEN phase...")

    while True:
        phase = current_phase()

        if phase == MarketPhase.MARKET_OPEN and not session.flat_for_day:
            logger.info("[MicroCoordinator] Launching MCL Micro Futures Agent...")
            session.agents_running.append("MicroFutures_MCL")

            micro_task = asyncio.create_task(
                micro_futures_agent(
                    instrument   = "MCL",
                    fast_period  = 10,
                    slow_period  = 30,
                    daily_target = DAILY_TARGET_USD,
                    poll_secs    = 60.0,
                ),
                name="MicroFutures_MCL",
            )

            # Monitor until flat or market closed
            while True:
                if session.flat_for_day or current_phase() != MarketPhase.MARKET_OPEN:
                    logger.info("[MicroCoordinator] Cancelling MCL agent — %s",
                                "flat for day" if session.flat_for_day else "market closed")
                    micro_task.cancel()
                    await asyncio.gather(micro_task, return_exceptions=True)
                    if "MicroFutures_MCL" in session.agents_running:
                        session.agents_running.remove("MicroFutures_MCL")
                    break
                if micro_task.done():
                    exc = micro_task.exception()
                    if exc:
                        logger.error("[MicroCoordinator] MCL agent exited with error: %s", exc)
                        session.errors_today += 1
                    break
                await asyncio.sleep(30)

        await asyncio.sleep(30)


# ============================================================================
# SECTION 7 — POST-MARKET PHASE: Performance Reporter
# ============================================================================

async def post_market_reporter(session: AutoSession) -> None:
    """
    Runs once per trading day during the post-market window (16:30–18:00 ET).
    Generates daily performance summary and archives session JSON to ./logs/.

    Heitkoetter: review every session P&L before the next open.
    This agent enforces that discipline automatically.
    """
    while True:
        phase = current_phase()

        if phase == MarketPhase.POST_MARKET and session.post_market_runs == 0:
            logger.info("[PostMarket] Generating daily performance report...")
            try:
                print("\n" + "═" * 70)
                print("  AUTONOMOUS SESSION REPORT")
                print(f"  Date       : {session.date}")
                print(f"  Phase      : {phase.value}")
                print(f"  Agents Run : {', '.join(session.agents_running) or 'none'}")
                print("─" * 70)
                print(f"  Total P&L  : ${session.total_realized_pnl:+,.2f}")
                print(f"  Micro P&L  : ${session.micro_pnl:+,.2f}")
                print(f"  VSA P&L    : ${session.vsa_pnl:+,.2f}")
                print("─" * 70)
                print(f"  Target     : ${DAILY_TARGET_USD:,.0f}/day  |  Limit: ${MAX_DAILY_LOSS_USD:,.0f}/day")
                print(f"  Approved   : {session.approved_signals}  |  Rejected: {session.rejected_signals}")
                print(f"  Errors     : {session.errors_today}")
                print(f"  Halt       : {session.halt_reason or 'none'}")
                print("─" * 70)
                if _RISK:
                    print(performance_report())
                if _ECOSYSTEM:
                    eco = get_ecosystem_metrics()
                    print("─" * 70)
                    print(f"  Ecosystem Bias : {eco.get('macro_bias', 'N/A')}")
                    print(f"  Ecosystem P&L  : ${eco.get('daily_pnl_usd', 0):+,.2f}")
                    print(f"  IB Trades      : {eco.get('total_trades', 0)}")
                    print(f"  Win Rate       : {eco.get('win_rate_pct', 0):.1f}%")
                    print(f"  Risk Cushion   : {eco.get('risk_cushion_pct', 100):.1f}%")
                print("═" * 70 + "\n")

                # Archive session log
                log_path = LOG_DIR / f"session_{session.date.isoformat()}.json"
                log_path.write_text(session.to_json(), encoding="utf-8")
                logger.info("[PostMarket] Session archived → %s", log_path)

                session.post_market_runs += 1

            except Exception as exc:
                logger.error("[PostMarket] Reporter error: %s", exc)
                session.errors_today  += 1
                session.post_market_runs += 1

        await asyncio.sleep(60)


# ============================================================================
# SECTION 8 — OVERNIGHT HEALTH MONITOR
# ============================================================================

async def overnight_monitor(session: AutoSession) -> None:
    """
    Runs continuously. During overnight hours emits a heartbeat every 30 min
    and checks system health. All agents are dormant; only this task is active.
    """
    heartbeat_interval = 1_800   # 30 minutes

    while True:
        phase = current_phase()
        if phase == MarketPhase.OVERNIGHT:
            et  = et_now()
            logger.info(
                "[Overnight] Heartbeat — %s ET | Account=$%.2f | Next session: %s",
                et.strftime("%H:%M"),
                ACCOUNT_EQUITY_USD,
                "pre-market at 08:00 ET",
            )
            await asyncio.sleep(heartbeat_interval)
        else:
            await asyncio.sleep(60)


# ============================================================================
# SECTION 8b — GLOBAL ECOSYSTEM COORDINATOR
# ============================================================================

async def ecosystem_coordinator(session: AutoSession) -> None:
    """
    Launches the full global_ecosystem.py 7-agent system as a background task.
    Bridges ecosystem state (global_memory.daily_flat) with AutoSession.
    Runs throughout all market phases — ecosystem handles its own phase logic
    via the maintenance_sentinel and exchange_lockout flags.
    """
    if not _ECOSYSTEM:
        logger.warning("[EcosystemCoordinator] global_ecosystem.py unavailable — skipping.")
        return

    logger.info("[EcosystemCoordinator] Launching 7-agent Global Ecosystem...")
    session.agents_running.append("GlobalEcosystem")

    eco_task = asyncio.create_task(
        start_global_ecosystem(cycle_secs=60.0),
        name="GlobalEcosystem",
    )

    try:
        while True:
            # Mirror ecosystem flat flag to AutoSession
            if _ECOSYSTEM and ecosystem_memory.daily_flat and not session.flat_for_day:
                session.flat_for_day = True
                session.halt_reason  = "Ecosystem daily limit reached"

            if eco_task.done():
                exc = eco_task.exception() if not eco_task.cancelled() else None
                if exc:
                    logger.error("[EcosystemCoordinator] Ecosystem exited: %s", exc)
                    session.errors_today += 1
                break

            await asyncio.sleep(30)
    except asyncio.CancelledError:
        eco_task.cancel()
        await asyncio.gather(eco_task, return_exceptions=True)
        if "GlobalEcosystem" in session.agents_running:
            session.agents_running.remove("GlobalEcosystem")


# ============================================================================
# SECTION 9 — DEMO MODE (offline, no API calls)
# ============================================================================

async def _run_demo() -> None:
    """
    Offline single-cycle demo. Runs all phases in compressed time:
      ① Pre-market pipeline (if available, else synthetic signals)
      ② VSA 3-bar simulation (noise filter + SOS execution)
      ③ Micro futures 2-bar simulation (BUY crossover + exit)
      ④ Post-market report
    """
    logger.info("=" * 70)
    logger.info("  AUTONOMOUS AGENT — DEMO MODE")
    logger.info("  Account: $%.2f | Daily Target: $%.0f | Loss Limit: $%.0f",
                ACCOUNT_EQUITY_USD, DAILY_TARGET_USD, MAX_DAILY_LOSS_USD)
    logger.info("=" * 70)

    session = AutoSession()

    # ── Phase 1: Pre-market pipeline ─────────────────────────────────────────
    logger.info("\n[Demo] Phase 1: Pre-market pipeline...")
    if _PIPELINE:
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_full_pipeline)
            approved = result.get("approved", [])
            session.approved_signals = len(approved)
            logger.info("[Demo] Pipeline OK — %d approved signals.", len(approved))
        except Exception as exc:
            logger.warning("[Demo] Pipeline error (non-fatal in demo): %s", exc)
    else:
        logger.info("[Demo] Pipeline module not available — skipping.")
    session.pre_market_runs = 1

    # ── Phase 2: VSA 4-agent demo ────────────────────────────────────────────
    logger.info("\n[Demo] Phase 2: VSA 4-agent team (3-bar simulation)...")
    if _VSA:
        shared_state      = SharedMarketState()
        market_data_queue = asyncio.Queue()
        execution_queue   = asyncio.Queue()

        vsa_tasks = [
            asyncio.create_task(macro_trend_agent(shared_state),            name="VSA_Trend"),
            asyncio.create_task(vsa_sharpshooter_agent(shared_state, market_data_queue), name="VSA_Sharp"),
            asyncio.create_task(context_execution_agent(shared_state, execution_queue),  name="VSA_Ctx"),
            asyncio.create_task(quant_risk_agent(execution_queue, shared_state),         name="VSA_Risk"),
        ]
        await asyncio.sleep(0.1)

        demo_bars = [
            VSABar(timestamp=1_710_000_000_000, open=70.50, high=70.65,
                   low=70.40, close=70.58, volume=120.0, trades_count=105),
            VSABar(timestamp=1_710_000_060_000, open=70.58, high=71.80,
                   low=70.50, close=71.75, volume=850.0, trades_count=2),
            VSABar(timestamp=1_710_000_120_000, open=70.58, high=71.80,
                   low=70.50, close=71.75, volume=650.0, trades_count=950),
        ]
        for i, bar in enumerate(demo_bars, 1):
            logger.info("[Demo] VSA bar %d — close=$%.2f vol=%.0f trades=%d",
                        i, bar.close, bar.volume, bar.trades_count)
            await market_data_queue.put(bar)
            await asyncio.sleep(2)

        await market_data_queue.join()
        await execution_queue.join()
        for t in vsa_tasks:
            t.cancel()
        await asyncio.gather(*vsa_tasks, return_exceptions=True)
        logger.info("[Demo] VSA team complete.")
    else:
        logger.info("[Demo] vsa_agents.py not available — skipping VSA phase.")

    # ── Phase 3: Micro futures demo ──────────────────────────────────────────
    logger.info("\n[Demo] Phase 3: Micro Futures SMA crossover (simulated bar)...")
    if _MICRO:
        from micro_futures import SMACrossEngine, OHLCBar, size_for_daily_target
        engine = SMACrossEngine(fast=10, slow=30)
        # Feed 31 synthetic bars to warm up the engine, then cross
        base = 71.00
        for i in range(30):
            engine.update(OHLCBar(
                timestamp=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
                open=base, high=base+0.20, low=base-0.20, close=base, volume=300.0,
            ))
        # Golden cross bar
        engine.update(OHLCBar(
            timestamp=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
            open=base, high=base+1.00, low=base-0.10, close=base+0.90, volume=600.0,
        ))
        sig = engine.update(OHLCBar(
            timestamp=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
            open=base+0.90, high=base+1.50, low=base+0.80, close=base+1.40, volume=700.0,
        ))
        logger.info("[Demo] SMA signal: %s", sig or "warming up")
        lots = size_for_daily_target("MCL", entry=72.40, stop=72.30)
        logger.info("[Demo] Position size for $%.0f target: %d MCL contracts", DAILY_TARGET_USD, lots)
    else:
        logger.info("[Demo] micro_futures.py not available — skipping micro phase.")

    # ── Phase 4: Post-market report ───────────────────────────────────────────
    logger.info("\n[Demo] Phase 4: Performance summary...")
    print("\n" + "═" * 70)
    print("  DEMO SESSION COMPLETE")
    print(f"  Account     : ${ACCOUNT_EQUITY_USD:,.2f}")
    print(f"  Daily Target: ${DAILY_TARGET_USD:,.0f}  |  Loss Limit: ${MAX_DAILY_LOSS_USD:,.0f}")
    print(f"  Approved    : {session.approved_signals} pipeline signals")
    print(f"  VSA         : {'ACTIVE' if _VSA       else 'unavailable'}")
    print(f"  Micro Agent : {'ACTIVE' if _MICRO     else 'unavailable'}")
    print(f"  Pipeline    : {'ACTIVE' if _PIPELINE  else 'unavailable'}")
    print(f"  Ecosystem   : {'ACTIVE' if _ECOSYSTEM else 'unavailable'}")
    print(f"  CrewAI Team : {'ACTIVE' if _CREW      else 'unavailable'}")
    print("═" * 70)
    logger.info("[Demo] All agents completed successfully.")


# ============================================================================
# SECTION 10 — SIMULATION MODE (compressed 1-day cycle)
# ============================================================================

async def _run_simulation() -> None:
    """
    Simulates one full trading day in compressed time (seconds, not hours).
    Used to validate all agent integrations before live deployment.

    Timeline (real seconds → simulated phase):
      t=0   : PRE_MARKET  — pipeline runs
      t=15  : MARKET_OPEN — VSA + micro agents run for 30 seconds
      t=45  : POST_MARKET — performance report
      t=60  : Done
    """
    logger.info("[Simulate] Starting compressed 1-day simulation...")
    session = AutoSession()

    # Shared shutdown event
    shutdown = asyncio.Event()

    async def _sim_risk_monitor():
        """Simulated risk monitor — checks every 2 seconds."""
        while not shutdown.is_set():
            halt = session.check_limits()
            if halt and not session.flat_for_day:
                session.flat_for_day = True
                session.halt_reason  = halt
                logger.warning("[Simulate|Risk] HALT: %s", halt)
            await asyncio.sleep(2)

    async def _sim_pipeline():
        logger.info("[Simulate|Pre-Market] Running pipeline...")
        if _PIPELINE:
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, run_full_pipeline)
                session.approved_signals = len(result.get("approved", []))
            except Exception as exc:
                logger.warning("[Simulate|Pre-Market] %s", exc)
        else:
            session.approved_signals = 3   # synthetic
            logger.info("[Simulate|Pre-Market] Synthetic: 3 approved signals.")
        session.pre_market_runs = 1

    async def _sim_market():
        """Compressed market session — 30 seconds."""
        logger.info("[Simulate|Market] OPEN — VSA + Micro agents active for 30s...")
        if _VSA:
            shared  = SharedMarketState()
            dq      = asyncio.Queue()
            eq      = asyncio.Queue()
            tasks   = [
                asyncio.create_task(macro_trend_agent(shared),                  name="VSA_T"),
                asyncio.create_task(vsa_sharpshooter_agent(shared, dq),         name="VSA_S"),
                asyncio.create_task(context_execution_agent(shared, eq),         name="VSA_C"),
                asyncio.create_task(quant_risk_agent(eq, shared),                name="VSA_R"),
            ]
            await asyncio.sleep(0.1)
            bar = VSABar(1_710_000_120_000, 70.58, 71.80, 70.50, 71.75, 650.0, 950)
            await dq.put(bar)
            await asyncio.sleep(5)
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.sleep(10)
        logger.info("[Simulate|Market] Session closed.")

    async def _sim_post():
        logger.info("[Simulate|Post-Market] Generating report...")
        print("\n" + "═" * 70)
        print("  SIMULATION SESSION REPORT")
        print(f"  P&L Total   : ${session.total_realized_pnl:+,.2f}")
        print(f"  Approved    : {session.approved_signals}")
        print(f"  Halt Reason : {session.halt_reason or 'none'}")
        print("═" * 70)

    risk_task = asyncio.create_task(_sim_risk_monitor())
    await _sim_pipeline()
    await _sim_market()
    await _sim_post()
    shutdown.set()
    risk_task.cancel()
    await asyncio.gather(risk_task, return_exceptions=True)
    logger.info("[Simulate] 1-day simulation complete.")


# ============================================================================
# SECTION 11 — MASTER AUTONOMOUS LOOP (24/7)
# ── Crew Coordinator ─────────────────────────────────────────────────────────

async def crew_coordinator(session: AutoSession) -> None:
    """
    8th autonomous task — runs CrewAI 4-agent trading team every 30 min
    during MARKET_OPEN. Feeds crew P&L into session.total_realized_pnl.
    """
    if not _CREW:
        logger.warning("[CrewCoordinator] crew_agent.py not available — skipping.")
        return

    logger.info("[CrewCoordinator] Building institutional knowledge base...")
    retriever = await asyncio.get_event_loop().run_in_executor(
        None, build_knowledge_base
    )
    logger.info("[CrewCoordinator] Knowledge base ready. Waiting for MARKET_OPEN...")

    crew_running = False
    while True:
        if session.flat_for_day:
            if crew_running:
                logger.info("[CrewCoordinator] Session flat — crew halted.")
                crew_running = False
            await asyncio.sleep(30)
            continue

        phase = current_phase()

        if phase == MarketPhase.MARKET_OPEN:
            if not crew_running:
                logger.info("[CrewCoordinator] MARKET_OPEN — launching crew cycle loop.")
                crew_running = True

            try:
                result = await run_crew_cycle(retriever)
                logger.info(
                    "[CrewCoordinator] Cycle complete | Risk=%s | Trade=%s | %.0fms",
                    result.risk_status, result.trade_executed, result.latency_ms,
                )
                if result.trade_executed:
                    session.total_realized_pnl += result.pnl_estimate
            except Exception as e:
                logger.error("[CrewCoordinator] Cycle error: %s", e)

            # Sleep for full cycle interval or until session end, checking flat_for_day
            elapsed = 0
            while elapsed < CREW_CYCLE_SECS and not session.flat_for_day:
                await asyncio.sleep(10)
                elapsed += 10
        else:
            if crew_running:
                logger.info("[CrewCoordinator] Market closed — crew standing by.")
                crew_running = False
            await asyncio.sleep(60)

# ============================================================================

async def run_autonomous() -> None:
    """
    Master 24/7 autonomous controller.

    Launches all agent coordinators as concurrent asyncio tasks:
      - risk_monitor        : always running, 10s checks
      - pre_market_runner   : triggers at 08:00 ET
      - vsa_coordinator     : active 09:00–16:30 ET
      - micro_coordinator   : active 09:00–16:30 ET
      - post_market_reporter: triggers at 16:30 ET
      - overnight_monitor   : active 18:00–08:00 ET

    Press Ctrl+C to initiate graceful shutdown.
    """
    session = AutoSession()

    logger.info("═" * 70)
    logger.info("  AUTONOMOUS AI TRADING SYSTEM — STARTING")
    logger.info("  Account     : $%.2f", ACCOUNT_EQUITY_USD)
    logger.info("  Daily Target: $%.0f  |  Loss Limit: $%.0f", DAILY_TARGET_USD, MAX_DAILY_LOSS_USD)
    logger.info("  Risk/Trade  : %.0f%%  ($%.2f)", MAX_RISK_PER_TRADE_PCT*100,
                ACCOUNT_EQUITY_USD * MAX_RISK_PER_TRADE_PCT)
    logger.info("  Phase now   : %s  (%s ET)", current_phase().value,
                et_now().strftime("%H:%M"))
    logger.info("  VSA agents  : %s", "READY" if _VSA        else "UNAVAILABLE")
    logger.info("  Micro agent : %s", "READY" if _MICRO      else "UNAVAILABLE")
    logger.info("  Pipeline    : %s", "READY" if _PIPELINE   else "UNAVAILABLE")
    logger.info("  Ecosystem   : %s", "READY" if _ECOSYSTEM  else "UNAVAILABLE")
    logger.info("═" * 70)

    tasks = [
        asyncio.create_task(risk_monitor(session),          name="RiskMonitor"),
        asyncio.create_task(pre_market_runner(session),     name="PreMarket"),
        asyncio.create_task(vsa_coordinator(session),       name="VSACoordinator"),
        asyncio.create_task(micro_coordinator(session),     name="MicroCoordinator"),
        asyncio.create_task(post_market_reporter(session),  name="PostMarket"),
        asyncio.create_task(overnight_monitor(session),     name="Overnight"),
        asyncio.create_task(ecosystem_coordinator(session), name="EcosystemCoordinator"),
        asyncio.create_task(crew_coordinator(session),      name="CrewCoordinator"),
    ]

    logger.info("All 8 autonomous tasks launched. Running 24/7. Press Ctrl+C to stop.")

    try:
        await asyncio.gather(*tasks)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        logger.info("[Autonomous] Shutdown signal received. Stopping all agents...")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # Final session archive
        log_path = LOG_DIR / f"session_{session.date.isoformat()}_shutdown.json"
        log_path.write_text(session.to_json(), encoding="utf-8")
        logger.info("[Autonomous] Final session state archived → %s", log_path)
        logger.info("[Autonomous] All agents stopped. Goodbye.")


# ============================================================================
# SECTION 12 — STATUS COMMAND
# ============================================================================

def print_status() -> None:
    """Print current system state and exit."""
    phase = current_phase()
    et    = et_now()
    print("═" * 70)
    print("  AUTONOMOUS AGENT — SYSTEM STATUS")
    print("─" * 70)
    print(f"  Time (ET)    : {et.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Market Phase : {phase.value}")
    print(f"  Account      : ${ACCOUNT_EQUITY_USD:,.2f}")
    print(f"  Daily Target : ${DAILY_TARGET_USD:,.0f}/day")
    print(f"  Loss Limit   : ${MAX_DAILY_LOSS_USD:,.0f}/day")
    print(f"  Risk/Trade   : {MAX_RISK_PER_TRADE_PCT:.0%} = ${ACCOUNT_EQUITY_USD*MAX_RISK_PER_TRADE_PCT:.2f}")
    print("─" * 70)
    print(f"  VSA Agents   : {'READY' if _VSA        else 'UNAVAILABLE (install vsa_agents.py)'}")
    print(f"  Micro Agent  : {'READY' if _MICRO      else 'UNAVAILABLE (install micro_futures.py)'}")
    print(f"  Pipeline     : {'READY' if _PIPELINE   else 'UNAVAILABLE (install main.py)'}")
    print(f"  Risk Engine  : {'READY' if _RISK       else 'UNAVAILABLE (install risk_engine.py)'}")
    print(f"  Ecosystem    : {'READY' if _ECOSYSTEM  else 'UNAVAILABLE (install global_ecosystem.py)'}")
    print(f"  CrewAI Team  : {'READY' if _CREW       else 'UNAVAILABLE (pip install crewai)'}")
    print(f"  HMM Regime   : {'READY' if _HMM        else 'UNAVAILABLE (install hmm_regime.py)'}")
    print(f"  Market Arch  : {'READY' if _MAM        else 'UNAVAILABLE (install market_architecture.py)'}")
    print("─" * 70)
    # Live HMM regime snapshot
    if _HMM:
        try:
            import importlib
            pd_mod = importlib.import_module("pandas")
            yf_mod = importlib.import_module("yfinance")
            raw = yf_mod.download("CL=F", period="1y", interval="1d",
                                  progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd_mod.MultiIndex):
                raw.columns = [c[0].lower() for c in raw.columns]
            else:
                raw.columns = [c.lower() for c in raw.columns]
            close_s = raw["close"].dropna()
            if len(close_s) >= 63:
                result = get_hmm_regime(ticker="CL=F", close=close_s)
                mult   = regime_size_multiplier(result)
                probs  = result.probabilities
                print(f"  HMM State    : {result.regime.value} (size_mult={mult:.2f})")
                print(f"  P(BULL)={probs.get('BULL',0):.2f}  "
                      f"P(BEAR)={probs.get('BEAR',0):.2f}  "
                      f"P(VOLATILE)={probs.get('VOLATILE',0):.2f}  "
                      f"P(SIDEWAYS)={probs.get('SIDEWAYS',0):.2f}")
                print(f"  MAP (Gupta & Dhingra 2012): {result.map_direction} "
                      f"(fracChange={result.map_frac_change:+.4f})")
                print(f"  Fallon (UMass Lowell 2012): {result.fallon_direction} "
                      f"(pred_return={result.fallon_predicted_return:+.4f})")
                print(f"  Rationale    : {result.explanation[:80]}")
        except Exception as hmm_e:
            print(f"  HMM State    : (unavailable — {hmm_e})")
    print("─" * 70)
    # Phase-1 exchange latency table (Market Architecture Math)
    if _MAM:
        try:
            mam      = _get_mam()
            latencies = mam.all_exchange_latencies()
            print("  EXCHANGE LATENCY ADVANTAGE (microwave vs fiber)")
            for route, data in latencies.items():
                print(f"    {route:<22} microwave={data['microwave_microseconds']:.0f}µs  "
                      f"fiber={data['fiber_microseconds']:.0f}µs  "
                      f"advantage={data['advantage_microseconds']:.0f}µs")
        except Exception as mam_e:
            print(f"  Market Arch  : (unavailable — {mam_e})")
        print("─" * 70)
    log_files = sorted(LOG_DIR.glob("session_*.json"))
    print(f"  Session logs : {len(log_files)} archived  →  {LOG_DIR}/")
    if log_files:
        latest = log_files[-1]
        try:
            data = json.loads(latest.read_text(encoding="utf-8"))
            print(f"  Last session : {latest.name}")
            print(f"    P&L        : ${data.get('total_realized_pnl', 0):+,.2f}")
            print(f"    Approved   : {data.get('approved_signals', 0)}")
            print(f"    Halt       : {data.get('halt_reason') or 'none'}")
        except Exception:
            pass
    print("═" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous AI Oil Contract Trading System Controller"
    )
    parser.add_argument("--demo",     action="store_true",
                        help="Run one offline demo cycle (no API calls)")
    parser.add_argument("--simulate", action="store_true",
                        help="Run compressed 1-day simulation")
    parser.add_argument("--status",   action="store_true",
                        help="Print system status and exit")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.demo:
        asyncio.run(_run_demo())
        return

    if args.simulate:
        asyncio.run(_run_simulation())
        return

    # Default: full 24/7 autonomous mode
    asyncio.run(run_autonomous())


if __name__ == "__main__":
    main()
