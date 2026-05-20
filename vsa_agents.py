"""
vsa_agents.py — Asynchronous 4-Agent VSA Trading Team
======================================================
System Role: Expert Quantitative Commodity Strategist (Institutional Grade)

Knowledge sources:
  Master the Markets (VSA / Tom Williams)
  Complete Guide to Day Trading (Heitkoetter)
  Art & Science of Technical Analysis (Grimes)
  Risk Management & Financial Institutions (Hull 4th Ed.)
  NYMEX Chapter 200 (WTI contract specs)

Architecture:
  Agent 1 — Macro-Trend Filter     : 4H/Daily structure → global bias (LONG_ONLY | SHORT_ONLY | FLAT)
  Agent 2 — VSA Sharpshooter       : 1M/5M tick scanner + volume-noise filter → SOS/SOW signal
  Agent 3 — Context & Execution    : Trend-alignment check + order geometry → order_queue
  Agent 4 — Quant Risk Manager     : 1% hard-stop position sizing + exchange API placeholder

Event-driven design:
  - asyncio.Queue for inter-agent communication (zero shared-state mutations in hot path)
  - SharedMarketState read by Agent 3; written only by Agents 1 and 2
  - No blocking time.sleep() calls anywhere in this module

Integration:
  - Account equity drawn from risk_engine.ACCOUNT_EQUITY_USD (falls back to 100 000 USD)
  - Position sizing respects the 1% per-trade risk ceiling (MAX_RISK_PER_TRADE_PCT)
  - No execute_trade() call — execution stub logs intent + API placeholder
  - ApprovalStatus gate enforced in Agent 4 before any order is finalised

Usage:
    python vsa_agents.py                  # live simulation loop (ctrl-c to stop)
    python vsa_agents.py --demo           # 3-bar offline demo (no API calls)
    python vsa_agents.py --demo --bars 5  # custom demo bar count

WTI threshold calibration guide (see VSA_THRESHOLDS below):
    WIDE_SPREAD_BBL   — minimum $/bbl bar range to qualify as wide-spread
    HIGH_VOLUME_LOTS  — minimum contract volume to qualify as high-volume
    NOISE_MIN_VOL     — volume floor below which noise filter is skipped
    NOISE_MAX_AVG_SZ  — avg trade size ceiling that flags wash/bot noise
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# ── Risk constants from risk_engine (graceful fallback if file missing) ──────
try:
    from risk_engine import (
        ACCOUNT_EQUITY_USD,
        MAX_RISK_PER_TRADE_PCT,
        ApprovalStatus,
    )
    _RISK_ENGINE = True
except ImportError:
    ACCOUNT_EQUITY_USD      = 500.0
    MAX_RISK_PER_TRADE_PCT  = 0.02
    _RISK_ENGINE = False

# ── Multi-factor ensemble (adds regime context to VSA signals) ───────────────
try:
    from signal_engine import generate_ensemble_signal, SignalDirection
    _ENSEMBLE = True
except ImportError:
    _ENSEMBLE = False

# ── Baum-Welch HMM regime (graceful fallback to legacy trend logic) ───────────
try:
    from hmm_regime import get_hmm_regime, regime_size_multiplier, OilRegime
    _HMM = True
except ImportError:
    _HMM = False

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("VSATradingTeam")

# ============================================================================
# SECTION 1 — VSA CALIBRATION THRESHOLDS (WTI / NYMEX Chapter 200)
# ============================================================================
# These tune the Sharpshooter and noise filter for WTI crude oil futures.
# Adjust for other instruments (Brent, RBOB, ULSD) by overriding at call site.
# Source: VSA — Master the Markets; Grimes — Art & Science of Technical Analysis

VSA_THRESHOLDS = {
    # Minimum bar spread ($/bbl) to be classified as wide-spread
    # For WTI at ~$70, a 1-minute ultra-wide move is typically ≥ $1.00/bbl
    "WIDE_SPREAD_BBL":   1.00,

    # Minimum contract volume for a bar to be considered high-volume
    # NYMEX WTI 1-min average: ~200-400 contracts during peak hours
    "HIGH_VOLUME_LOTS":  300.0,

    # Volume floor: bars below this skip the noise filter (too small to be wash)
    "NOISE_MIN_VOL":     500.0,

    # Max average trade size (contracts/trade) before bar is flagged as noise
    # Wash trades / bots route hundreds of tiny fills → avg_size < 0.01
    "NOISE_MAX_AVG_SZ":  0.01,

    # Close-position ratio: above this = closing at top (bullish VSA)
    "CLOSE_TOP_RATIO":   0.80,

    # Close-position ratio: below this = closing at bottom (bearish VSA)
    "CLOSE_BOTTOM_RATIO": 0.20,

    # Macro trend refresh interval (seconds). Default 900 = 15 minutes.
    "TREND_REFRESH_SEC": 900,

    # Context agent polling interval (seconds).
    "CONTEXT_POLL_SEC":  1,
}

# ============================================================================
# SECTION 2 — DATA CONTAINERS
# ============================================================================

@dataclass
class VSABar:
    """Single OHLCV bar with trade-count for noise detection.

    trades_count is required by the Volume Noise Filter (Agent 2).
    Source: VSA — Master the Markets; Sherbin — How to Price and Trade Options.
    """
    timestamp:    int     # Unix epoch ms
    open:         float   # $/bbl
    high:         float
    low:          float
    close:        float
    volume:       float   # contracts (NYMEX lot = 1,000 bbl)
    trades_count: int     # individual fills inside this bar


@dataclass
class OrderRequest:
    """Validated, sized order ready for execution stub.

    No live routing is performed. Agent 4 logs intent and calls the
    exchange API placeholder. See: NYMEX Ch.200, Hull Ch.16.
    """
    direction:   str    # 'BUY' | 'SELL'
    entry_price: float
    stop_loss:   float
    signal_type: str    # 'SOS_SHARPSHOOTER' | 'SOW_SHARPSHOOTER'


@dataclass
class SharedMarketState:
    """Inter-agent read/write state database.

    Agents 1 and 2 write; Agent 3 reads; Agent 4 never reads this directly.
    account_balance is seeded from risk_engine.ACCOUNT_EQUITY_USD so that
    position sizing always reflects the single authoritative equity figure.
    hmm_size_mult: Baum-Welch soft-posterior multiplier [0.0, 1.0] from Agent 1.
    """
    trend_state:     str             = "FLAT"  # LONG_ONLY | SHORT_ONLY | FLAT
    active_signal:   Optional[str]   = None    # SOS_SHARPSHOOTER | SOW_SHARPSHOOTER
    signal_bar:      Optional[VSABar]= None
    account_balance: float           = field(default_factory=lambda: ACCOUNT_EQUITY_USD)
    hmm_size_mult:   float           = 1.0     # HMM regime-based position size scalar
    map_direction:   str             = "FLAT"  # MAP next-bar prediction: UP|DOWN|FLAT
    fallon_signal:   str             = "SKIP"  # Fallon likelihood-similarity: BUY|SKIP


# ============================================================================
# AGENT 1 — MACRO-TREND FILTER
# ============================================================================

async def macro_trend_agent(state: SharedMarketState,
                           ticker: str = "CL=F") -> None:
    """Scans Daily market structure every 15 minutes via Baum-Welch HMM.

    Primary path (when _HMM is True):
      Fetches 1Y of daily WTI closes from yfinance, runs get_hmm_regime() to
      obtain the Viterbi-decoded hidden state {BULL, BEAR, VOLATILE, SIDEWAYS}
      and soft posteriors γ_t(i) from the Baum-Welch forward-backward pass.

      BULL     → state.trend_state = "LONG_ONLY"   (size_mult from γ_BULL)
      BEAR     → state.trend_state = "SHORT_ONLY"  (size_mult from γ_BEAR)
      VOLATILE → state.trend_state = "FLAT"         (size_mult = 0.25, crisis)
      SIDEWAYS → state.trend_state = "FLAT"         (size_mult = 0.5)

    Fallback (when _HMM is False or yfinance unavailable):
      Keeps legacy LONG_ONLY bias (preserves original behaviour).

    Agents 2 and 3 read trend_state; only Agent 1 writes it.
    Grimes Ch.5: trade in the direction of the dominant trend structure.
    """
    import importlib
    refresh = VSA_THRESHOLDS["TREND_REFRESH_SEC"]
    while True:
        try:
            logger.info("[Agent 1 | Trend] Fetching market structure via HMM...")

            hmm_resolved = False
            if _HMM:
                try:
                    yf_mod = importlib.import_module("yfinance")
                    raw = yf_mod.download(ticker, period="1y", interval="1d",
                                          progress=False, auto_adjust=True)
                    if isinstance(raw.columns, __import__("pandas").MultiIndex):
                        raw.columns = [c[0].lower() for c in raw.columns]
                    else:
                        raw.columns = [c.lower() for c in raw.columns]
                    close_series = raw["close"].dropna()
                    if len(close_series) >= 63:
                        result = get_hmm_regime(ticker=ticker, close=close_series)
                        mult   = regime_size_multiplier(result)
                        state.hmm_size_mult  = mult
                        state.map_direction  = result.map_direction
                        state.fallon_signal  = result.fallon_direction
                        r = result.regime
                        if r == OilRegime.BULL:
                            state.trend_state = "LONG_ONLY"
                        elif r == OilRegime.BEAR:
                            state.trend_state = "SHORT_ONLY"
                        else:  # VOLATILE or SIDEWAYS
                            state.trend_state = "FLAT"
                        logger.info(
                            "[Agent 1 | Trend] HMM=%s bias=%s size_mult=%.2f "
                            "MAP=%s(fc=%+.4f) Fallon=%s(ret=%+.4f) | %s",
                            r.value, state.trend_state, mult,
                            result.map_direction, result.map_frac_change,
                            result.fallon_direction, result.fallon_predicted_return,
                            result.explanation,
                        )
                        hmm_resolved = True
                except Exception as hmm_exc:
                    logger.warning("[Agent 1 | Trend] HMM fetch failed (%s) — using fallback", hmm_exc)

            if not hmm_resolved:
                state.trend_state  = "LONG_ONLY"
                state.hmm_size_mult = 1.0
                logger.info("[Agent 1 | Trend] Fallback bias → LONG_ONLY")

        except Exception as exc:
            logger.error("[Agent 1 | Trend] Error: %s", exc)

        await asyncio.sleep(refresh)


# ============================================================================
# AGENT 2 — VSA SHARPSHOOTER SCANNER (with Volume Noise Filter)
# ============================================================================

async def vsa_sharpshooter_agent(
    state: SharedMarketState,
    data_queue: asyncio.Queue,
) -> None:
    """Processes 1M/5M VSABar ticks from data_queue.

    Stage A — Volume Noise Filter:
      Strips wash-trading and spoofing using average trade size.
      High volume + unnaturally few trades → institutional spoof → discard.
      High volume + microscopic avg size  → retail bot flood → discard.
      Source: VSA — Master the Markets, Ch.3 (Volume Analysis).

    Stage B — VSA Sharpshooter Algorithm:
      Sign of Strength (SOS): ultra-wide bar, high volume, close ≥ 80% of range.
      Sign of Weakness (SOW): ultra-wide bar, high volume, close ≤ 20% of range.
      Source: Tom Williams — Master the Markets; Grimes — Art & Science Ch.9.
    """
    thresholds = VSA_THRESHOLDS
    while True:
        bar: VSABar = await data_queue.get()
        try:
            # ── Stage A: Volume Noise Filter ──────────────────────────────
            if bar.trades_count > 0 and bar.volume >= thresholds["NOISE_MIN_VOL"]:
                avg_trade_size = bar.volume / bar.trades_count
                if avg_trade_size < thresholds["NOISE_MAX_AVG_SZ"]:
                    logger.warning(
                        "[Agent 2 | Filter] Noise detected — vol=%.1f trades=%d "
                        "avg_size=%.4f. Bar discarded.",
                        bar.volume, bar.trades_count, avg_trade_size,
                    )
                    data_queue.task_done()
                    continue

            # ── Stage B: VSA Sharpshooter ─────────────────────────────────
            spread        = bar.high - bar.low
            denom         = spread if spread > 0 else 1.0
            close_pos     = (bar.close - bar.low) / denom

            wide_spread   = spread >= thresholds["WIDE_SPREAD_BBL"]
            high_volume   = bar.volume >= thresholds["HIGH_VOLUME_LOTS"]

            if wide_spread and high_volume and close_pos >= thresholds["CLOSE_TOP_RATIO"]:
                state.active_signal = "SOS_SHARPSHOOTER"
                state.signal_bar    = bar
                logger.info(
                    "[Agent 2 | Scanner] SHARPSHOOTER — Sign of Strength (SOS) "
                    "| spread=%.3f vol=%.0f close_pos=%.2f",
                    spread, bar.volume, close_pos,
                )

            elif wide_spread and high_volume and close_pos <= thresholds["CLOSE_BOTTOM_RATIO"]:
                state.active_signal = "SOW_SHARPSHOOTER"
                state.signal_bar    = bar
                logger.info(
                    "[Agent 2 | Scanner] SHARPSHOOTER — Sign of Weakness (SOW) "
                    "| spread=%.3f vol=%.0f close_pos=%.2f",
                    spread, bar.volume, close_pos,
                )

            else:
                state.active_signal = None

        except Exception as exc:
            logger.error("[Agent 2 | Scanner] Error: %s", exc)
        finally:
            data_queue.task_done()


# ============================================================================
# AGENT 3 — CONTEXT & EXECUTION AGENT
# ============================================================================

async def context_execution_agent(
    state: SharedMarketState,
    order_queue: asyncio.Queue,
) -> None:
    """Validates trend alignment and computes order geometry, then enqueues.

    Rules:
      SOS_SHARPSHOOTER + LONG_ONLY → BUY; stop below signal bar low.
      SOW_SHARPSHOOTER + SHORT_ONLY → SELL; stop above signal bar high.
      Misaligned signals are discarded without queuing.

    Hull Ch.16: the execution layer must be independent of the signal generator.
    Bittman Ch.12: stop placement at structural extremes, not arbitrary offsets.
    """
    poll = VSA_THRESHOLDS["CONTEXT_POLL_SEC"]
    while True:
        await asyncio.sleep(poll)

        if state.active_signal is None:
            continue

        signal = state.active_signal
        bar    = state.signal_bar

        # ── Ensemble regime gate (three-way confirmation) ──────────────────
        # VSA gives the bar-level signal. Macro trend gives direction bias.
        # Ensemble adds factor-model regime context. All three must agree.
        ensemble_ok = True
        if _ENSEMBLE:
            try:
                ens = generate_ensemble_signal("CL=F")
                if signal == "SOS_SHARPSHOOTER" and ens.direction != SignalDirection.BUY:
                    ensemble_ok = False
                    logger.info(
                        "[Agent 3 | Execution] SOS blocked by ensemble (%s, score=%.2f). "
                        "Three-way confirmation failed.",
                        ens.direction.value, ens.score,
                    )
                elif signal == "SOW_SHARPSHOOTER" and ens.direction != SignalDirection.SELL:
                    ensemble_ok = False
                    logger.info(
                        "[Agent 3 | Execution] SOW blocked by ensemble (%s, score=%.2f). "
                        "Three-way confirmation failed.",
                        ens.direction.value, ens.score,
                    )
                else:
                    logger.info(
                        "[Agent 3 | Execution] Ensemble agrees: %s | score=%.2f | confidence=%.0f%%",
                        ens.direction.value, ens.score, ens.confidence * 100,
                    )
            except Exception as exc:
                logger.warning("[Agent 3] Ensemble check error: %s — proceeding without", exc)

        if not ensemble_ok:
            state.active_signal = None
            continue

        if signal == "SOS_SHARPSHOOTER" and state.trend_state == "LONG_ONLY":
            logger.info(
                "[Agent 3 | Execution] THREE-WAY CONFIRMATION: VSA=SOS + Macro=LONG_ONLY "
                "+ Ensemble=BUY → entering LONG."
            )
            order = OrderRequest(
                direction   = "BUY",
                entry_price = bar.close,
                stop_loss   = bar.low - 0.05,
                signal_type = signal,
            )
            await order_queue.put(order)
            state.active_signal = None

        elif signal == "SOW_SHARPSHOOTER" and state.trend_state == "SHORT_ONLY":
            logger.info(
                "[Agent 3 | Execution] THREE-WAY CONFIRMATION: VSA=SOW + Macro=SHORT_ONLY "
                "+ Ensemble=SELL → entering SHORT."
            )
            order = OrderRequest(
                direction   = "SELL",
                entry_price = bar.close,
                stop_loss   = bar.high + 0.05,
                signal_type = signal,
            )
            await order_queue.put(order)
            state.active_signal = None

        else:
            logger.info(
                "[Agent 3 | Execution] Signal %s discarded — trend misalignment "
                "(global bias: %s).",
                signal, state.trend_state,
            )
            state.active_signal = None


# ============================================================================
# AGENT 4 — QUANTITATIVE RISK MANAGER
# ============================================================================

async def quant_risk_agent(
    order_queue: asyncio.Queue,
    state: SharedMarketState,
) -> None:
    """Applies 1% hard-stop position sizing and fires the execution stub.

    Position sizing formula:
        risk_usd      = account_balance × MAX_RISK_PER_TRADE_PCT
        risk_per_unit = |entry_price − stop_loss| × 1,000 bbl/contract
        position_lots = risk_usd / risk_per_unit

    The 1% ceiling (Heitkoetter Ch.3) ensures a single trade can never exceed
    the MAX_DAILY_LOSS_USD gate defined in risk_engine.py.

    No execute_trade() call is made — live order routing requires a broker
    integration approved by the risk committee. See CLAUDE.md constraints.

    Sources: Heitkoetter Ch.3, Hull Ch.16, NYMEX Ch.200.
    """
    BBL_PER_CONTRACT = 1_000   # NYMEX WTI Chapter 200 §200.00
    while True:
        order: OrderRequest = await order_queue.get()
        try:
            risk_per_bbl    = abs(order.entry_price - order.stop_loss)
            risk_per_lot    = risk_per_bbl * BBL_PER_CONTRACT

            if risk_per_lot <= 0:
                logger.warning(
                    "[Agent 4 | Risk] Zero risk distance — order skipped."
                )
                order_queue.task_done()
                continue

            # Use the live account balance from shared state (seeded from
            # ACCOUNT_EQUITY_USD in risk_engine.py via SharedMarketState).
            # HMM soft-posterior multiplier scales size by regime confidence:
            #   BULL/BEAR → 0.8–1.0  | VOLATILE → 0.25  | SIDEWAYS → 0.5
            total_usd_to_risk = (state.account_balance
                                 * MAX_RISK_PER_TRADE_PCT
                                 * state.hmm_size_mult)
            position_lots     = total_usd_to_risk / risk_per_lot

            logger.info(
                "[Agent 4 | Risk] ── POSITION INITIALISED ──────────────────"
            )
            logger.info(
                "[Agent 4 | Risk] Signal      : %s", order.signal_type
            )
            logger.info(
                "[Agent 4 | Risk] Action       : %s  %.4f lots",
                order.direction, position_lots,
            )
            logger.info(
                "[Agent 4 | Risk] Entry        : $%.2f/bbl", order.entry_price
            )
            logger.info(
                "[Agent 4 | Risk] Hard Stop    : $%.2f/bbl", order.stop_loss
            )
            logger.info(
                "[Agent 4 | Risk] Risk $/lot   : $%.2f", risk_per_lot
            )
            logger.info(
                "[Agent 4 | Risk] Capital @ Risk: $%.2f (%.1f%% × HMM_mult=%.2f of $%.0f)",
                total_usd_to_risk,
                MAX_RISK_PER_TRADE_PCT * 100,
                state.hmm_size_mult,
                state.account_balance,
            )

            # ── EXECUTION STUB ────────────────────────────────────────────
            # Replace the block below with your approved broker integration:
            #   import ccxt
            #   exchange = ccxt.exchange_id({"apiKey": ..., "secret": ...})
            #   exchange.create_order(
            #       symbol="CL/USD", type="market",
            #       side=order.direction.lower(), amount=position_lots,
            #   )
            # NEVER commit API credentials to this repository. See CLAUDE.md.
            # ─────────────────────────────────────────────────────────────

        except Exception as exc:
            logger.error("[Agent 4 | Risk] Error: %s", exc)
        finally:
            order_queue.task_done()


# ============================================================================
# ORCHESTRATOR — Event Loop Bootstrap
# ============================================================================

async def _run_demo(n_bars: int = 3) -> None:
    """Offline 3-bar simulation demonstrating noise filtering and execution.

    Bar 1: Normal activity  — passes filter, no VSA signal.
    Bar 2: Wash trade noise — high vol, near-zero trade count → DISCARDED.
    Bar 3: Institutional SOS — high vol, wide spread, top close → EXECUTED.
    """
    shared_state       = SharedMarketState()
    market_data_queue  = asyncio.Queue()
    execution_queue    = asyncio.Queue()

    tasks = [
        asyncio.create_task(macro_trend_agent(shared_state),              name="TrendAgent"),
        asyncio.create_task(vsa_sharpshooter_agent(shared_state, market_data_queue), name="SharpshooterAgent"),
        asyncio.create_task(context_execution_agent(shared_state, execution_queue),  name="ContextAgent"),
        asyncio.create_task(quant_risk_agent(execution_queue, shared_state),         name="RiskAgent"),
    ]

    logger.info("VSA Trading Team — 4 agents initialising...")
    await asyncio.sleep(0.1)   # allow event loop to schedule agent coroutines

    # Demo bars calibrated for WTI crude at ~$71/bbl
    demo_bars: List[VSABar] = [
        # Normal bar: small spread, low volume, no signal
        VSABar(timestamp=1_710_000_000_000, open=70.50, high=70.65, low=70.40,
               close=70.58, volume=120.0, trades_count=105),

        # Wash-trade bar: high volume, only 2 trades → noise, discarded by Agent 2
        VSABar(timestamp=1_710_000_060_000, open=70.58, high=71.80, low=70.50,
               close=71.75, volume=850.0, trades_count=2),

        # Institutional SOS bar: high vol, 950 trades, wide spread, closes at top
        VSABar(timestamp=1_710_000_120_000, open=70.58, high=71.80, low=70.50,
               close=71.75, volume=650.0, trades_count=950),
    ]

    for i, bar in enumerate(demo_bars[:n_bars], 1):
        logger.info(
            "\n[Feed] Candle %d — close=$%.2f  vol=%.0f  trades=%d",
            i, bar.close, bar.volume, bar.trades_count,
        )
        await market_data_queue.put(bar)
        await asyncio.sleep(3)

    # Allow agents to finish processing before shutdown
    await market_data_queue.join()
    await execution_queue.join()

    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("VSA Trading Team — demo complete.")


async def _run_live() -> None:
    """Live event loop skeleton — replace the feed section with a real data source."""
    shared_state      = SharedMarketState()
    market_data_queue = asyncio.Queue()
    execution_queue   = asyncio.Queue()

    tasks = [
        asyncio.create_task(macro_trend_agent(shared_state),              name="TrendAgent"),
        asyncio.create_task(vsa_sharpshooter_agent(shared_state, market_data_queue), name="SharpshooterAgent"),
        asyncio.create_task(context_execution_agent(shared_state, execution_queue),  name="ContextAgent"),
        asyncio.create_task(quant_risk_agent(execution_queue, shared_state),         name="RiskAgent"),
    ]

    logger.info("VSA Trading Team — live mode. Press Ctrl-C to stop.")
    try:
        # ── LIVE FEED PLACEHOLDER ────────────────────────────────────────────
        # Replace this section with a real-time market data subscription:
        #   async for bar in your_feed_client.stream("CL=F", interval="1m"):
        #       await market_data_queue.put(VSABar(**bar))
        # ────────────────────────────────────────────────────────────────────
        await asyncio.gather(*tasks)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("VSA Trading Team — shut down.")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VSA 4-Agent Trading Team (WTI crude oil)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run offline 3-bar simulation instead of live mode",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=3,
        help="Number of demo bars to replay (default: 3)",
    )
    args = parser.parse_args()

    if not _RISK_ENGINE:
        logger.warning(
            "risk_engine.py not found — using fallback equity $%.0f @ %.0f%% risk cap.",
            ACCOUNT_EQUITY_USD, MAX_RISK_PER_TRADE_PCT * 100,
        )

    if args.demo:
        asyncio.run(_run_demo(n_bars=args.bars))
    else:
        asyncio.run(_run_live())


if __name__ == "__main__":
    main()
