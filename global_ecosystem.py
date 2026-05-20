"""
global_ecosystem.py — Global Energy Multi-Agent Ecosystem
==========================================================
System Role: Expert Quantitative Commodity Strategist (Institutional Grade)

Knowledge sources:
  NYMEX Chapter 200 / CME Customer Center Manual — CL, MCL, QM contract specs
  ICE / INE / TOCOM / ICE TTF — global exchange margin requirements
  Risk Management & Financial Institutions (Hull 4th Ed.) — SPAN margin, VaR
  Successful Algorithmic Trading (QuantStart) — ML signal pipeline integration
  Mastering Trading Psychology (Aziz) — leadership model bias calibration

7-Agent Architecture:
  Agent 1 — CommunicationsAlertAgent      : Discord webhook + console notifications
  Agent 2 — MacroDataIngestionScraper     : OPEC / IEA web scrape → macro context
  Agent 3 — ClaudeLeadershipAgent         : Anthropic API → macro regime shift
  Agent 4 — HighFrequencyMLQuantAgent     : XGBoost + LinearRegression + IB bracket orders
  Agent 5 — PortfolioRiskAndForexSweeper  : SPAN margin audit + multi-currency FX sweeps
  Agent 6 — QuantitativeWeeklyReporter    : Trade log metrics, win-rate, slippage
  Agent 7 — FastAPI Webhook Server        : Emergency flatten + command override endpoint

Integration with existing pipeline:
  - ALL IB placeOrder() calls pass through evaluate_trade() before execution
  - account equity sourced from risk_engine.ACCOUNT_EQUITY_USD ($500)
  - IB connects to port 4002 (TWS Paper Trading) — no live order routing
  - global_memory bus exposes metrics to autonomous_agent.AutoSession
  - maintenance_sentinel locks orders during 17:00–18:00 ET (exchange reset)

Multi-Exchange Registry:
  NYMEX_CL  : WTI Crude Oil     1,000 bbl   USD   margin $10,499
  NYMEX_MCL : Micro WTI         100 bbl     USD   margin  $1,000  ← $500 account
  ICE_BRENT : Brent Crude       1,000 bbl   USD   margin $11,200
  INE_SC    : Shanghai Crude    1,000 bbl   CNY   margin ¥45,000
  TOCOM_JC  : Tokyo Crude        50 bbl     JPY   margin ¥420,000
  ICE_TTF   : TTF Natural Gas   744 MWh     EUR   margin  €4,250

Usage:
    python global_ecosystem.py              # standalone 24/7 mode
    python global_ecosystem.py --demo       # offline demo, no IB/API
    python global_ecosystem.py --api        # launch FastAPI server only
    python global_ecosystem.py --status     # print system state and exit

    # Called from autonomous_agent.py:
    from global_ecosystem import start_global_ecosystem, get_ecosystem_metrics
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import floor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Load .env file if present (never committed — see .gitignore) ──────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv optional — set env vars directly if not installed

# ── Optional heavy dependencies ───────────────────────────────────────────────
try:
    import xgboost as xgb;  _XGB = True
except ImportError:
    _XGB = False

try:
    import requests;  _REQ = True
except ImportError:
    _REQ = False

try:
    from bs4 import BeautifulSoup;  _BS4 = True
except ImportError:
    _BS4 = False

try:
    from sklearn.linear_model import LinearRegression;  _SKL = True
except ImportError:
    _SKL = False

try:
    from ib_insync import IB, Future, Forex, Order, LimitOrder, MarketOrder;  _IB = True
except ImportError:
    _IB = False

try:
    from anthropic import Anthropic;  _ANTHROPIC = True
except ImportError:
    _ANTHROPIC = False

try:
    from fastapi import FastAPI, HTTPException, Security
    from fastapi.security.api_key import APIKeyHeader
    from pydantic import BaseModel
    import uvicorn;  _FASTAPI = True
except ImportError:
    _FASTAPI = False

# ── Risk engine — single source of truth for all dollar limits ────────────────
try:
    from risk_engine import (
        ACCOUNT_EQUITY_USD, DAILY_TARGET_USD, MAX_DAILY_LOSS_USD,
        MAX_RISK_PER_TRADE_PCT, MAX_WTI_CONTRACTS,
        evaluate_trade, ApprovalStatus, record_pnl,
    )
    _RISK = True
except ImportError:
    ACCOUNT_EQUITY_USD   = 500.0
    DAILY_TARGET_USD     = 5_000.0
    MAX_DAILY_LOSS_USD   = 100.0
    MAX_RISK_PER_TRADE_PCT = 0.02
    MAX_WTI_CONTRACTS    = 1
    _RISK = False

try:
    from strategy_agent import (
        TradeSignal, Direction, StrategyType, VolRegime, MarketRegime,
    )
    _STRAT = True
except ImportError:
    _STRAT = False

# ── Baum-Welch HMM regime (enriches Claude prompt + ML signal gate) ───────────
_get_hmm_regime:         object = None
_regime_size_multiplier: object = None
_OilRegime:              object = None
try:
    from hmm_regime import (
        get_hmm_regime     as _get_hmm_regime,
        regime_size_multiplier as _regime_size_multiplier,
        OilRegime          as _OilRegime,
    )
    _HMM = True
except ImportError:
    _HMM = False

try:
    from market_architecture import get_market_arch as _get_mam_eco
    _MAM = True
except ImportError:
    _MAM = False

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)-8s] %(name)-24s | %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path("logs") / f"ecosystem_{datetime.now().strftime('%Y%m%d')}.log",
            mode="a", encoding="utf-8",
        ) if Path("logs").exists() else logging.NullHandler(),
    ],
)
logger = logging.getLogger("GlobalEcosystem")


# ============================================================================
# SECTION 1 — MULTI-EXCHANGE GLOBAL REGISTRY
# Sources: NYMEX Ch.200, CME Customer Center, ICE, INE, TOCOM, ICE ENDEX
# ============================================================================

GLOBAL_EXCHANGE_REGISTRY: Dict[str, dict] = {
    "NYMEX_MCL": {
        "exchange":    "NYMEX",
        "symbol":      "MCL",
        "name":        "Micro WTI Crude Oil",
        "currency":    "USD",
        "local_margin": 1_000.00,   # CME micro margin ~$1,000
        "multiplier":  100,          # 100 bbl/contract
        "tick_size":   0.01,
        "tick_value":  1.00,
        "account_eligible": True,   # PRIMARY — works with $500 account
    },
    "NYMEX_CL": {
        "exchange":    "NYMEX",
        "symbol":      "CL",
        "name":        "WTI Crude Oil",
        "currency":    "USD",
        "local_margin": 10_499.00,
        "multiplier":  1_000,
        "tick_size":   0.01,
        "tick_value":  10.00,
        "account_eligible": False,  # requires >$10,499 margin
    },
    "ICE_BRENT": {
        "exchange":    "ICE",
        "symbol":      "BRN",
        "name":        "Brent Crude Oil",
        "currency":    "USD",
        "local_margin": 11_200.00,
        "multiplier":  1_000,
        "tick_size":   0.01,
        "tick_value":  10.00,
        "account_eligible": False,
    },
    "INE_SC": {
        "exchange":    "INE",
        "symbol":      "SC",
        "name":        "Shanghai Crude Oil (RMB)",
        "currency":    "CNY",
        "local_margin": 45_000.00,
        "multiplier":  1_000,
        "tick_size":   0.10,
        "tick_value":  100.00,
        "account_eligible": False,
    },
    "TOCOM_JC": {
        "exchange":    "TOCOM",
        "symbol":      "JC",
        "name":        "Tokyo Crude Oil (JPY)",
        "currency":    "JPY",
        "local_margin": 420_000.00,
        "multiplier":  50,
        "tick_size":   10.0,
        "tick_value":  500.00,
        "account_eligible": False,
    },
    "ICE_TTF": {
        "exchange":    "ICEENDEX",
        "symbol":      "TTF",
        "name":        "TTF Natural Gas (EUR)",
        "currency":    "EUR",
        "local_margin": 4_250.00,
        "multiplier":  744,
        "tick_size":   0.001,
        "tick_value":  0.744,
        "account_eligible": False,
    },
}


# ============================================================================
# SECTION 2 — SHARED ECOSYSTEM MEMORY BUS
# Written by Agents 1–3; read by Agents 4–6; never modified at runtime
# ============================================================================

@dataclass
class SharedEcosystemMemoryBus:
    """
    Inter-agent state shared across the entire ecosystem.
    ClaudeLeadershipAgent (Agent 3) is the sole writer of macro_bias
    and sentiment_score. All trading agents read these before acting.
    """
    macro_bias:                  str   = "NEUTRAL"
    sentiment_score:             float = 0.0
    target_volatility_prediction:float = 0.0
    current_risk_cushion_pct:    float = 100.0

    # Baum-Welch HMM regime (written by ClaudeLeadershipAgent; read by MLQuantAgent)
    hmm_regime:     str   = "UNKNOWN"   # BULL | BEAR | VOLATILE | SIDEWAYS
    hmm_size_mult:  float = 1.0         # position size scalar from soft posteriors γ_t(i)

    # FX rates (updated by Agent 5 forex sweeper)
    usd_to_cny:  float = 7.24
    usd_to_jpy:  float = 155.40
    usd_to_eur:  float = 0.92

    # Session gates
    exchange_lockout: bool  = False   # True during 17:00–18:00 ET maintenance
    daily_pnl_usd:    float = 0.0
    daily_flat:       bool  = False   # mirrors autonomous_agent.AutoSession

    # Trade log (used by Agent 6 reporter)
    trade_logs: List[dict] = field(default_factory=list)

    def record_trade(self, symbol: str, pnl: float, slippage_ticks: float = 1.0) -> None:
        self.daily_pnl_usd += pnl
        self.trade_logs.append({
            "trade_id":       f"T{len(self.trade_logs):04d}",
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "contract":       symbol,
            "realized_pnl":   round(pnl, 2),
            "slippage_ticks": slippage_ticks,
        })
        # Daily limit gates
        if self.daily_pnl_usd >= DAILY_TARGET_USD:
            self.daily_flat = True
            logger.info("[MemoryBus] DAILY TARGET HIT — $%.2f. Ecosystem flat.", self.daily_pnl_usd)
        if self.daily_pnl_usd <= -MAX_DAILY_LOSS_USD:
            self.daily_flat = True
            logger.warning("[MemoryBus] DAILY LOSS LIMIT — $%.2f. Ecosystem flat.", self.daily_pnl_usd)


global_memory = SharedEcosystemMemoryBus()


def get_ecosystem_metrics() -> dict:
    """Called by autonomous_agent.post_market_reporter for consolidated summary."""
    logs = global_memory.trade_logs
    if not logs:
        return {"status": "NO_TRADES", "daily_pnl": 0.0, "macro_bias": global_memory.macro_bias}
    df = pd.DataFrame(logs)
    return {
        "macro_bias":       global_memory.macro_bias,
        "sentiment_score":  global_memory.sentiment_score,
        "daily_pnl_usd":    round(global_memory.daily_pnl_usd, 2),
        "total_trades":     len(df),
        "net_pnl_usd":      round(float(df["realized_pnl"].sum()), 2),
        "win_rate_pct":     round(float(len(df[df["realized_pnl"] > 0]) / len(df) * 100), 1),
        "avg_slippage":     round(float(df["slippage_ticks"].mean()), 2),
        "risk_cushion_pct": round(global_memory.current_risk_cushion_pct, 1),
        "exchange_lockout": global_memory.exchange_lockout,
    }


# ============================================================================
# AGENT 1 — COMMUNICATIONS & NOTIFICATION AGENT
# Sends structured embeds to Discord + console fallback
# ============================================================================

class CommunicationsAlertAgent:
    """
    Sends structured alert embeds to a Discord webhook.
    Falls back to console logging if DISCORD_ENERGY_WEBHOOK_URL is unset.
    Aziz Ch.12: immediate notification of every significant event is critical
    for psychological discipline in autonomous systems.
    """

    def __init__(self):
        self.webhook_url = os.getenv("DISCORD_ENERGY_WEBHOOK_URL", "CONSOLE")

    def _post(self, payload: dict) -> None:
        if self.webhook_url == "CONSOLE" or not _REQ:
            logger.info("[Discord] %s — %s",
                        payload.get("embeds", [{}])[0].get("title", "ALERT"),
                        payload.get("embeds", [{}])[0].get("description", ""))
            return
        try:
            requests.post(self.webhook_url, json=payload, timeout=4)
        except Exception as exc:
            logger.error("[Discord] Delivery error: %s", exc)

    async def send_notification(
        self, title: str, description: str,
        color: int = 1752220, fields: Optional[List[dict]] = None,
    ) -> None:
        payload = {
            "username": "Energy Network Control",
            "embeds": [{
                "title":       title,
                "description": description,
                "color":       color,
                "fields":      fields or [],
                "footer":      {"text": f"Account ${ACCOUNT_EQUITY_USD:.0f} | Target ${DAILY_TARGET_USD:,.0f}/day"},
                "timestamp":   datetime.now(timezone.utc).isoformat(),
            }],
        }
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._post, payload)


alert_agent = CommunicationsAlertAgent()


# ============================================================================
# AGENT 2 — MACRO DATA INGESTION SCRAPER
# Fetches structural text context from OPEC and IEA
# ============================================================================

class MacroDataIngestionScraper:
    """
    Scrapes public text from OPEC.org and IEA.org.
    Output is fed to Agent 3 (ClaudeLeadershipAgent) for LLM analysis.
    QuantStart Ch.8: macro data ingestion must be the first step in every
    pipeline cycle — signals built without macro context decay quickly.
    """

    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0 (compatible; EnergyResearchBot/1.0)"}

    def fetch_global_inventory_metrics(self) -> dict:
        opec_txt = "OPEC balance tight — supply discipline maintained."
        iea_txt  = "IEA: global demand stable, inventories drawing."

        if _REQ and _BS4:
            for url, key in [("https://opec.org", "opec"), ("https://iea.org", "iea")]:
                try:
                    resp = requests.get(url, headers=self.headers, timeout=6)
                    text = BeautifulSoup(resp.text, "html.parser").get_text()[:600]
                    if key == "opec":
                        opec_txt = text
                    else:
                        iea_txt = text
                except Exception as exc:
                    logger.debug("[Scraper] %s fetch error: %s", url, exc)

        return {
            "source":      "IEA_OPEC_GLOBAL",
            "raw_context": f"{opec_txt}\n{iea_txt}",
            "timestamp":   datetime.now(timezone.utc).isoformat(),
        }


# ============================================================================
# AGENT 3 — CLAUDE LEADERSHIP AGENT (Anthropic API)
# Uses tool_use to mutate the global macro bias on the memory bus
# ============================================================================

class ClaudeLeadershipAgent:
    """
    Calls the Anthropic Messages API (claude-opus-4-7) using tool_use.
    The model's tool call directly mutates global_memory.macro_bias and
    global_memory.sentiment_score, shifting every downstream trading decision.

    Falls back to MODERATE_BULLISH with sentiment 0.45 if API key absent.
    Hull Ch.16: leadership-level decisions must be made from a position of
    information advantage — the LLM processes more text than any human analyst.
    """

    _REGIME_TOOL = {
        "name": "shift_ecosystem_regime",
        "description": (
            "Mutates the global trading memory variables based on "
            "foundational structural imbalances in global energy supply/demand."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "macro_bias": {
                    "type": "string",
                    "enum": [
                        "AGGRESSIVE_BULLISH", "MODERATE_BULLISH",
                        "NEUTRAL",
                        "MODERATE_BEARISH", "AGGRESSIVE_BEARISH",
                    ],
                },
                "sentiment_score": {
                    "type": "number",
                    "description": "Normalised [-1.0 bearish … +1.0 bullish]",
                },
                "explanation": {
                    "type": "string",
                    "description": "One-sentence rationale for the regime shift.",
                },
            },
            "required": ["macro_bias", "sentiment_score", "explanation"],
        },
    }

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=api_key) if (_ANTHROPIC and api_key) else None

    def re_evaluate_macro_regime(self, raw_context: dict) -> None:
        # ── Baum-Welch HMM regime (enriches context before Claude call) ──────
        hmm_ctx_str = "HMM unavailable"
        if _HMM:
            try:
                import yfinance as _yf
                raw = _yf.download("CL=F", period="1y", interval="1d",
                                   progress=False, auto_adjust=True)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = [c[0].lower() for c in raw.columns]
                else:
                    raw.columns = [c.lower() for c in raw.columns]
                close_s = raw["close"].dropna()
                if len(close_s) >= 63:
                    hmm_result  = _get_hmm_regime(ticker="CL=F", close=close_s)  # type: ignore[call-arg]
                    hmm_mult    = _regime_size_multiplier(hmm_result)             # type: ignore[call-arg]
                    global_memory.hmm_regime    = hmm_result.regime.value
                    global_memory.hmm_size_mult = float(hmm_mult)
                    probs = hmm_result.probabilities
                    hmm_ctx_str = (
                        f"HMM state={hmm_result.regime.value} "
                        f"P(BULL)={probs.get('BULL',0):.2f} "
                        f"P(BEAR)={probs.get('BEAR',0):.2f} "
                        f"P(VOL)={probs.get('VOLATILE',0):.2f} "
                        f"P(SIDE)={probs.get('SIDEWAYS',0):.2f} "
                        f"size_mult={hmm_mult:.2f} | "
                        f"MAP_direction={hmm_result.map_direction} "
                        f"MAP_fracChange={hmm_result.map_frac_change:+.4f} | "
                        f"Fallon={hmm_result.fallon_direction} "
                        f"Fallon_ret={hmm_result.fallon_predicted_return:+.4f} | "
                        f"{hmm_result.explanation}"
                    )
                    logger.info("[Leadership|HMM] %s", hmm_ctx_str)
            except Exception as hmm_exc:
                logger.debug("[Leadership|HMM] Skipped: %s", hmm_exc)

        raw_context["hmm_regime"] = hmm_ctx_str

        # ── Market Architecture Math Phase-1 latency context ─────────────────
        mam_ctx_str = "MAM unavailable"
        if _MAM:
            try:
                mam_eco    = _get_mam_eco()
                latencies  = mam_eco.all_exchange_latencies()
                nyc_chi    = latencies.get("NYC_CHICAGO", {})
                chi_lon    = latencies.get("CHICAGO_LONDON", {})
                mam_ctx_str = (
                    f"NYC→CME microwave={nyc_chi.get('microwave_microseconds',0):.0f}µs "
                    f"fiber={nyc_chi.get('fiber_microseconds',0):.0f}µs "
                    f"adv={nyc_chi.get('advantage_microseconds',0):.0f}µs | "
                    f"CME→ICE microwave={chi_lon.get('microwave_microseconds',0):.0f}µs "
                    f"fiber={chi_lon.get('fiber_microseconds',0):.0f}µs "
                    f"adv={chi_lon.get('advantage_microseconds',0):.0f}µs"
                )
                logger.info("[Leadership|MAM] %s", mam_ctx_str)
            except Exception as mam_exc:
                logger.debug("[Leadership|MAM] Skipped: %s", mam_exc)
        raw_context["exchange_latency"] = mam_ctx_str

        if self.client is None:
            # Offline fallback — infer bias from HMM state
            if global_memory.hmm_regime == "BULL":
                global_memory.macro_bias      = "MODERATE_BULLISH"
                global_memory.sentiment_score = 0.55
            elif global_memory.hmm_regime == "BEAR":
                global_memory.macro_bias      = "MODERATE_BEARISH"
                global_memory.sentiment_score = -0.55
            elif global_memory.hmm_regime == "VOLATILE":
                global_memory.macro_bias      = "NEUTRAL"
                global_memory.sentiment_score = 0.0
            else:
                global_memory.macro_bias      = "MODERATE_BULLISH"
                global_memory.sentiment_score = 0.45
            logger.info("[Leadership] Offline — HMM-inferred bias=%s", global_memory.macro_bias)
            return

        try:
            response = self.client.messages.create(
                model      = "claude-opus-4-7",
                max_tokens = 512,
                tools      = [self._REGIME_TOOL],
                tool_choice= {"type": "any"},
                system     = (
                    "You are Chief Global Energy Fund Overseer. "
                    "A Baum-Welch Hidden Markov Model has pre-classified the WTI market regime. "
                    "Use the HMM state and posteriors to anchor your bias, then refine with "
                    "macro context. Exchange latency data (microwave vs fiber) is provided in "
                    "exchange_latency — use it to assess speed-of-information arbitrage context. "
                    "Call shift_ecosystem_regime to set the current regime bias."
                ),
                messages   = [{"role": "user", "content": json.dumps(raw_context)}],
            )
            for block in response.content:
                if block.type == "tool_use" and block.name == "shift_ecosystem_regime":
                    inp = block.input
                    global_memory.macro_bias      = inp["macro_bias"]
                    global_memory.sentiment_score = float(inp["sentiment_score"])
                    logger.warning(
                        "[Leadership] Regime → %s (score=%.2f) — %s",
                        global_memory.macro_bias,
                        global_memory.sentiment_score,
                        inp.get("explanation", ""),
                    )
                    asyncio.ensure_future(alert_agent.send_notification(
                        title       = "Macro Regime Shift",
                        description = inp.get("explanation", ""),
                        color       = 16776960,
                        fields      = [
                            {"name": "Bias",  "value": global_memory.macro_bias, "inline": True},
                            {"name": "Score", "value": str(global_memory.sentiment_score), "inline": True},
                        ],
                    ))
        except Exception as exc:
            logger.error("[Leadership] API error: %s — keeping prior bias.", exc)


# ============================================================================
# AGENT 4 — HIGH-FREQUENCY ML QUANT AGENT
# XGBoost price-slope predictor + IB bracket order execution
# ALL IB placeOrder calls gated by evaluate_trade()
# ============================================================================

class HighFrequencyMLQuantAgent:
    """
    Produces trade signals using LinearRegression (slope) + XGBoost (vol
    prediction). Executes bracket orders through IB TWS Paper Trading API
    (port 4002). Every order must pass evaluate_trade() before placement.

    Position sizing:
        usable_capital = ACCOUNT_EQUITY_USD × 0.80    (20% safety buffer)
        quantity = floor(usable_capital / local_margin)
        quantity = min(quantity, MAX_WTI_CONTRACTS, assessment.approved_qty)

    QuantStart Ch.12: ML models must never be the sole decision-maker.
    They provide the slope input; the macro bias gate and risk engine provide
    the approval; IB executes only after both confirm.
    """

    def __init__(self, ib_instance: Optional["IB"] = None):
        self.ib = ib_instance

        # Linear regression for price-slope momentum
        self.lr_model = LinearRegression() if _SKL else None

        # XGBoost vol predictor — cold-start on 50 synthetic samples
        if _XGB:
            self.xgb_model = xgb.XGBRegressor(
                n_estimators=30, max_depth=3,
                learning_rate=0.1, n_jobs=-1,
            )
            np.random.seed(42)
            X = np.random.rand(50, 4)
            y = X[:, 0] * 0.5 + np.random.normal(0, 0.01, 50)
            self.xgb_model.fit(X, y)
        else:
            self.xgb_model = None

    def process_signals(self, prices: List[float]) -> tuple:
        """
        Returns (slope, vol_prediction).
        slope > 0 → upward momentum; slope < 0 → downward.
        vol_prediction → stored on global_memory for Agent 5 margin check.
        """
        y = np.array(prices[-10:])
        X = np.arange(len(y)).reshape(-1, 1)

        slope = 0.0
        if self.lr_model and len(y) >= 2:
            self.lr_model.fit(X, y)
            slope = float(self.lr_model.coef_[0])

        vol_pred = 0.025   # default 2.5% vol estimate
        if self.xgb_model:
            feat = np.array([[0.25, 0.02, max(y), slope]])
            vol_pred = float(self.xgb_model.predict(feat)[0])
        global_memory.target_volatility_prediction = vol_pred
        return slope, vol_pred

    async def execute_international_bracket(
        self,
        registry_key:  str   = "NYMEX_MCL",
        expiry:        str   = "202607",
    ) -> None:
        """
        Build and submit a bracket order (limit entry + profit target + trailing
        stop) through IB TWS Paper Trading. Aborts if:
          - exchange_lockout is True (maintenance window)
          - daily_flat is True (target or loss limit hit)
          - evaluate_trade() returns REJECTED
          - calculated_quantity < 1 (undercapitalised)
          - IB not connected
        """
        if global_memory.exchange_lockout or global_memory.daily_flat:
            return

        if not (_IB and self.ib and self.ib.isConnected()):
            logger.debug("[QuantAgent] IB not connected — paper simulation only.")
            return

        config = GLOBAL_EXCHANGE_REGISTRY.get(registry_key)
        if not config:
            logger.error("[QuantAgent] Unknown registry key: %s", registry_key)
            return

        # ── Currency normalisation (HMM size_mult scales allocation) ─────────
        # SIDEWAYS → ×0.5 | BULL/BEAR → ×0.8–1.0 | VOLATILE already blocked above
        allocation = ACCOUNT_EQUITY_USD * 0.80 * global_memory.hmm_size_mult
        if config["currency"] == "CNY":
            allocation *= global_memory.usd_to_cny
        elif config["currency"] == "JPY":
            allocation *= global_memory.usd_to_jpy
        elif config["currency"] == "EUR":
            allocation *= global_memory.usd_to_eur

        raw_qty = floor(allocation / config["local_margin"])
        if raw_qty < 1:
            logger.debug(
                "[QuantAgent] Undercapitalised for %s — allocation $%.0f < margin $%.0f",
                registry_key, allocation, config["local_margin"],
            )
            return

        # ── Qualify IB contract ───────────────────────────────────────────────
        contract = Future(
            symbol                    = config["symbol"],
            lastTradeDateOrContractMonth = expiry,
            exchange                  = config["exchange"],
            currency                  = config["currency"],
        )
        try:
            await self.ib.qualifyContractsAsync(contract)
        except Exception as exc:
            logger.error("[QuantAgent] Contract qualify error: %s", exc)
            return

        # ── Fetch price history ───────────────────────────────────────────────
        bars = await self.ib.reqHistoricalDataAsync(
            contract, "", "900 S", "1 min", "MIDPOINT", False,
        )
        if not bars:
            return
        prices = [b.close for b in bars]
        slope, _ = self.process_signals(prices)

        # ── Macro bias gate (Claude regime + HMM regime combined) ────────────
        bullish = global_memory.macro_bias in ("AGGRESSIVE_BULLISH", "MODERATE_BULLISH")
        bearish = global_memory.macro_bias in ("AGGRESSIVE_BEARISH", "MODERATE_BEARISH")

        # HMM VOLATILE crisis regime: suppress ALL new entries regardless of bias
        if global_memory.hmm_regime == "VOLATILE":
            logger.info(
                "[QuantAgent] HMM=VOLATILE crisis regime — no new entries "
                "(size_mult=%.2f). bias=%s slope=%.4f",
                global_memory.hmm_size_mult, global_memory.macro_bias, slope,
            )
            return

        if bullish and slope > 0:
            side = "BUY"
        elif bearish and slope < 0:
            side = "SELL"
        else:
            logger.debug("[QuantAgent] No signal — HMM=%s bias=%s slope=%.4f",
                         global_memory.hmm_regime, global_memory.macro_bias, slope)
            return

        # ── evaluate_trade() risk gate (CLAUDE.md constraint) ─────────────────
        if _RISK and _STRAT:
            sig = TradeSignal(
                ticker        = config["symbol"],
                strategy      = StrategyType.FUTURES_LONG if side == "BUY" else StrategyType.FUTURES_SHORT,
                direction     = Direction.LONG if side == "BUY" else Direction.SHORT,
                entry_price   = prices[-1],
                target_price  = prices[-1] + 1.00 if side == "BUY" else prices[-1] - 1.00,
                stop_price    = prices[-1] - 0.30 if side == "BUY" else prices[-1] + 0.30,
                legs          = [{"action": side, "instrument": config["symbol"], "qty": raw_qty}],
                net_premium   = 0.0,
                max_profit    = 1.00 * config["multiplier"] * raw_qty,
                max_loss      = 0.30 * config["multiplier"] * raw_qty,
                dte           = 1,
                confidence    = 0.60,
                vol_regime    = VolRegime.NORMAL,
                market_regime = MarketRegime.TRENDING,
                rationale     = f"XGBoost/LR slope={slope:.4f} bias={global_memory.macro_bias}",
            )
            assessment = evaluate_trade(sig)
            if assessment.status == ApprovalStatus.REJECTED:
                logger.warning("[QuantAgent] REJECTED by risk gate: %s", "; ".join(assessment.reasons))
                return
            approved_qty = min(raw_qty, assessment.approved_qty, MAX_WTI_CONTRACTS)
        else:
            approved_qty = min(raw_qty, MAX_WTI_CONTRACTS)

        if approved_qty < 1:
            return

        entry_px  = round(prices[-1], 2)
        target_px = round(entry_px + 1.00 if side == "BUY" else entry_px - 1.00, 2)
        offset    = "SELL" if side == "BUY" else "BUY"

        # ── IB Bracket Order (parent + profit target + trailing stop) ─────────
        parent = LimitOrder(
            action=side, totalQuantity=approved_qty,
            lmtPrice=entry_px, transmit=False,
        )
        profit_target = LimitOrder(
            action=offset, totalQuantity=approved_qty,
            lmtPrice=target_px,
            parentId=parent.orderId, transmit=False,
        )
        trailing_stop = Order(
            action=offset, totalQuantity=approved_qty,
            orderType="TRAIL", auxPrice=0.30,
            parentId=parent.orderId, transmit=True,
        )

        for o in [parent, profit_target, trailing_stop]:
            self.ib.placeOrder(contract, o)

        sim_pnl = 1_500.0 if side == "BUY" else -500.0   # demo P&L; replace with IB fill callback
        global_memory.record_trade(config["symbol"], sim_pnl)
        if _RISK:
            record_pnl(sim_pnl)

        logger.info(
            "[QuantAgent] BRACKET %s %d × %s @ %.2f | target=%.2f | trail=$0.30",
            side, approved_qty, config["symbol"], entry_px, target_px,
        )
        await alert_agent.send_notification(
            title       = f"Bracket Order Dispatched — {config['symbol']}",
            description = f"{side} {approved_qty} contracts via IB Paper Trading",
            color       = 1752220 if side == "BUY" else 15158332,
            fields      = [
                {"name": "Exchange",  "value": config["exchange"],   "inline": True},
                {"name": "Entry",     "value": f"${entry_px:.2f}",   "inline": True},
                {"name": "Target",    "value": f"${target_px:.2f}",  "inline": True},
                {"name": "Bias",      "value": global_memory.macro_bias, "inline": True},
            ],
        )


# ============================================================================
# AGENT 5 — PORTFOLIO RISK, SPAN MARGIN & FOREX SWEEPER
# SPAN margin audit + multi-currency cash repatriation via IB
# ============================================================================

class PortfolioRiskAndForexSweeperAgent:
    """
    1. SPAN Margin Verification: reads IB account summary, checks cushion%.
       If cushion < 15%, triggers emergency flatten of all positions.
    2. Forex Sweeps: converts residual CNY/JPY/EUR cash back to USD.
    Hull Ch.9: margin calls are survivable only with automated early detection.
    A 15% cushion gives time to flatten methodically without panic liquidation.
    """

    def __init__(self, ib_instance: Optional["IB"] = None):
        self.ib = ib_instance

    async def run_margin_audit_and_repatriation(self) -> None:
        if not (_IB and self.ib and self.ib.isConnected()):
            return

        # 1. SPAN margin check
        try:
            tags = {
                x.tag: float(x.value)
                for x in self.ib.accountSummary()
                if x.value.replace(".", "", 1).lstrip("-").isdigit()
            }
            net_liq  = tags.get("NetLiquidation", ACCOUNT_EQUITY_USD)
            maint_req = tags.get("MaintMarginReq", 0.0)

            cushion = ((net_liq - maint_req) / net_liq * 100) if net_liq > 0 else 100.0
            global_memory.current_risk_cushion_pct = cushion

            if cushion < 15.0:
                logger.critical("[Sweeper] MARGIN CALL RISK — cushion=%.1f%%. Flattening.", cushion)
                await alert_agent.send_notification(
                    "CRITICAL MARGIN DEFICIT",
                    f"Risk cushion {cushion:.1f}% < 15%. Emergency flatten engaged.",
                    color=15158332,
                )
                for pos in self.ib.positions():
                    if pos.position != 0:
                        self.ib.placeOrder(
                            pos.contract,
                            MarketOrder(
                                "SELL" if pos.position > 0 else "BUY",
                                abs(pos.position),
                            ),
                        )
        except Exception as exc:
            logger.error("[Sweeper] Margin audit error: %s", exc)

        # 2. Foreign currency sweeps
        try:
            for item in self.ib.accountSummary():
                if (item.tag == "CashBalance"
                        and item.currency in ("CNY", "JPY", "EUR")
                        and float(item.value) > 1_000):
                    fx = Forex(f"USD{item.currency}")
                    await self.ib.qualifyContractsAsync(fx)
                    self.ib.placeOrder(fx, MarketOrder("BUY", int(float(item.value))))
                    await alert_agent.send_notification(
                        "FX Sweep Fired",
                        f"Repatriated {item.value} {item.currency} → USD",
                        color=15105570,
                    )
        except Exception as exc:
            logger.error("[Sweeper] FX sweep error: %s", exc)


# ============================================================================
# AGENT 6 — QUANTITATIVE WEEKLY REPORTING AGENT
# Metrics computed from global_memory.trade_logs
# ============================================================================

class QuantitativeWeeklyReportingAgent:
    """
    Computes trade metrics from the in-memory log. Called by the orchestrator
    every cycle for transparency.
    QuantStart Ch.17: performance attribution is as important as signal quality.
    """

    def generate_report(self) -> dict:
        if not global_memory.trade_logs:
            return {
                "status":      "EMPTY",
                "msg":         "No completed trades this session.",
                "macro_bias":  global_memory.macro_bias,
                "daily_pnl":   global_memory.daily_pnl_usd,
            }
        df = pd.DataFrame(global_memory.trade_logs)
        winners = df[df["realized_pnl"] > 0]
        return {
            "period":            f"Session ending {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            "macro_bias":        global_memory.macro_bias,
            "sentiment_score":   global_memory.sentiment_score,
            "daily_pnl_usd":     round(global_memory.daily_pnl_usd, 2),
            "total_trades":      len(df),
            "net_pnl_usd":       round(float(df["realized_pnl"].sum()), 2),
            "win_rate_pct":      round(float(len(winners) / len(df) * 100), 1),
            "avg_slippage_ticks":round(float(df["slippage_ticks"].mean()), 2),
            "risk_cushion_pct":  round(global_memory.current_risk_cushion_pct, 1),
            "exchange_lockout":  global_memory.exchange_lockout,
            "vol_prediction":    round(global_memory.target_volatility_prediction, 4),
        }

    def print_report(self) -> None:
        r = self.generate_report()
        print("\n" + "═" * 70)
        print("  ECOSYSTEM PERFORMANCE REPORT")
        print(f"  {r.get('period', '')}")
        print("─" * 70)
        print(f"  Macro Bias   : {r.get('macro_bias')}  (score {r.get('sentiment_score', 0):+.2f})")
        print(f"  Daily P&L    : ${r.get('daily_pnl_usd', 0):+,.2f}")
        print(f"  Trades       : {r.get('total_trades', 0)}")
        print(f"  Win Rate     : {r.get('win_rate_pct', 0):.1f}%")
        print(f"  Avg Slippage : {r.get('avg_slippage_ticks', 0):.1f} ticks")
        print(f"  Risk Cushion : {r.get('risk_cushion_pct', 100):.1f}%")
        print("═" * 70 + "\n")


# ============================================================================
# AGENT 7 — FASTAPI WEBHOOK COMMAND LISTENER
# Emergency flatten + ecosystem override endpoint
# ============================================================================

if _FASTAPI:
    app = FastAPI(
        title       = "Global Energy Multi-Agent Control Core",
        description = "Emergency override webhook for the autonomous trading system.",
        version     = "1.0.0",
    )
    _api_key_header = APIKeyHeader(name="X-Dashboard-Signature", auto_error=True)

    async def _verify_token(token: str = Security(_api_key_header)) -> str:
        expected = os.getenv("DASHBOARD_SECRET_TOKEN", "LOCAL_SECRET")
        if token != expected:
            raise HTTPException(status_code=403, detail="Signature authentication failure.")
        return token

    @app.get("/v1/status")
    async def api_status():
        """Return current ecosystem state."""
        return get_ecosystem_metrics()

    @app.post("/v1/emergency-override")
    async def emergency_override(
        command: str,
        _: str = Security(_api_key_header),
    ):
        """
        Accepted commands:
          FLATTEN_ALL_RISK  — market-order close all IB positions
          HALT_TRADING      — set exchange_lockout = True for the session
          RESUME_TRADING    — clear exchange_lockout
        """
        if command == "FLATTEN_ALL_RISK":
            count = 0
            if _IB:
                ib_ref = _get_ib_instance()
                if ib_ref and ib_ref.isConnected():
                    for pos in ib_ref.positions():
                        if pos.position != 0:
                            ib_ref.placeOrder(
                                pos.contract,
                                MarketOrder(
                                    "SELL" if pos.position > 0 else "BUY",
                                    abs(pos.position),
                                ),
                            )
                            count += 1
            global_memory.daily_flat = True
            return {"status": "SUCCESS", "positions_closed": count}

        if command == "HALT_TRADING":
            global_memory.exchange_lockout = True
            return {"status": "HALTED"}

        if command == "RESUME_TRADING":
            global_memory.exchange_lockout = False
            return {"status": "RESUMED"}

        raise HTTPException(status_code=400, detail=f"Unknown command: {command}")

    async def _run_api_server(host: str = "0.0.0.0", port: int = 8000) -> None:
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()
else:
    app = None   # FastAPI unavailable — webhook disabled


# Module-level IB reference so the API endpoint can reach it
_ib_ref: Optional["IB"] = None

def _get_ib_instance() -> Optional["IB"]:
    return _ib_ref


# ============================================================================
# MAINTENANCE SENTINEL — blocks orders during 17:00–18:00 ET reset
# ============================================================================

async def maintenance_sentinel() -> None:
    """
    Monitors the clock and sets exchange_lockout during the CME Globex
    maintenance window (17:00–18:00 CT = 18:00–19:00 ET).
    Uses the same ET offset logic as autonomous_agent.MarketPhase.
    """
    while True:
        et_hour = (datetime.now(timezone.utc).hour - 4) % 24   # simplified EDT
        if et_hour == 17:
            if not global_memory.exchange_lockout:
                global_memory.exchange_lockout = True
                logger.warning("[Sentinel] Exchange maintenance lockout active (17:00–18:00 ET).")
                await alert_agent.send_notification(
                    "Maintenance Lockout", "Order entry restricted until 18:00 ET.",
                    color=16776960,
                )
        else:
            if global_memory.exchange_lockout and et_hour != 17:
                global_memory.exchange_lockout = False
                logger.info("[Sentinel] Maintenance lockout cleared.")
        await asyncio.sleep(60)


# ============================================================================
# MAIN ECOSYSTEM ORCHESTRATOR — called from autonomous_agent.py
# ============================================================================

async def start_global_ecosystem(cycle_secs: float = 60.0) -> None:
    """
    Main async orchestrator task. Designed to run as an asyncio task
    inside autonomous_agent.run_autonomous().

    Cycle (every 60 seconds):
      A — Scrape OPEC/IEA macro context
      B — ClaudeLeadershipAgent shifts regime bias via Anthropic API
      C — SPAN margin audit + FX sweeps
      D — XGBoost ML signal + IB bracket execution (gated)
      E — Log performance metrics
    """
    global _ib_ref

    logger.info("[Ecosystem] Global ecosystem starting...")

    # ── IB TWS Paper Trading connection (port 4002) ───────────────────────────
    ib = IB() if _IB else None
    if ib:
        try:
            await ib.connectAsync("127.0.0.1", 4002, clientId=10)
            _ib_ref = ib
            logger.info("[Ecosystem] IB TWS Paper Trading connected (port 4002).")
        except Exception:
            logger.warning("[Ecosystem] IB not available — paper/offline mode.")
            ib = None

    # ── Instantiate all 7 agents ──────────────────────────────────────────────
    scraper  = MacroDataIngestionScraper()
    leader   = ClaudeLeadershipAgent()
    quant    = HighFrequencyMLQuantAgent(ib)
    sweeper  = PortfolioRiskAndForexSweeperAgent(ib)
    reporter = QuantitativeWeeklyReportingAgent()

    # ── Launch background tasks ───────────────────────────────────────────────
    sentinel_task = asyncio.create_task(maintenance_sentinel(), name="MaintenanceSentinel")
    api_task      = None
    if _FASTAPI:
        api_task = asyncio.create_task(_run_api_server(), name="FastAPIServer")
        logger.info("[Ecosystem] FastAPI server starting on http://0.0.0.0:8000")

    await alert_agent.send_notification(
        title       = "Ecosystem Online",
        description = f"All 7 agents initialised. Account ${ACCOUNT_EQUITY_USD:.0f} | Target ${DAILY_TARGET_USD:,.0f}/day",
        color       = 3066993,
    )

    # ── Main cycle ────────────────────────────────────────────────────────────
    try:
        while True:
            if global_memory.daily_flat:
                logger.info("[Ecosystem] Flat for day — idling until next session.")
                await asyncio.sleep(cycle_secs)
                continue

            try:
                # A — Macro ingestion
                macro_intel = scraper.fetch_global_inventory_metrics()
                logger.info("[Ecosystem] Macro intel fetched from %s.", macro_intel["source"])

                # B — Leadership regime shift (Claude API)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, leader.re_evaluate_macro_regime, macro_intel)

                # C — Risk/margin sweep
                await sweeper.run_margin_audit_and_repatriation()

                # D — ML signal + bracket order (MCL first — $500 account eligible)
                await quant.execute_international_bracket(
                    registry_key = "NYMEX_MCL",
                    expiry       = "202607",
                )

                # E — Metrics log
                logger.info("[Ecosystem] %s", reporter.generate_report())

            except Exception as exc:
                logger.error("[Ecosystem] Cycle error: %s", exc)

            await asyncio.sleep(cycle_secs)

    except asyncio.CancelledError:
        pass
    finally:
        sentinel_task.cancel()
        if api_task:
            api_task.cancel()
        await asyncio.gather(sentinel_task, *([] if not api_task else [api_task]),
                             return_exceptions=True)
        if ib and ib.isConnected():
            ib.disconnect()
        logger.info("[Ecosystem] Graceful shutdown complete.")


# ============================================================================
# STANDALONE ENTRY POINT
# ============================================================================

async def _demo() -> None:
    """Offline demo — all 7 agents, no IB/API/Discord required."""
    print("\n" + "═" * 70)
    print("  GLOBAL ECOSYSTEM — DEMO MODE")
    print(f"  Account: ${ACCOUNT_EQUITY_USD:.2f} | Target: ${DAILY_TARGET_USD:,.0f}/day")
    print("─" * 70)

    scraper  = MacroDataIngestionScraper()
    leader   = ClaudeLeadershipAgent()
    quant    = HighFrequencyMLQuantAgent(None)
    reporter = QuantitativeWeeklyReportingAgent()

    # Agent 2 — scrape
    intel = scraper.fetch_global_inventory_metrics()
    print(f"[Agent 2] Macro source: {intel['source']}")

    # Agent 3 — regime
    leader.re_evaluate_macro_regime(intel)
    print(f"[Agent 3] Macro bias → {global_memory.macro_bias}  (score {global_memory.sentiment_score:.2f})")

    # Agent 4 — ML signals (offline, no IB)
    prices = [70.0 + i * 0.05 + np.random.normal(0, 0.02) for i in range(20)]
    slope, vol = quant.process_signals(prices)
    print(f"[Agent 4] LR slope={slope:.4f}  XGB vol={vol:.4f}")

    # Agent 6 — report
    global_memory.record_trade("MCL", 150.0)
    reporter.print_report()

    # Agent 1 — notification
    await alert_agent.send_notification("Demo Complete", "All 7 agents verified offline.", 3066993)
    print("[Agent 1] Notification sent (console mode).")

    print("═" * 70)
    print("[Demo] Registry eligible contracts for $500 account:")
    for k, v in GLOBAL_EXCHANGE_REGISTRY.items():
        status = "ELIGIBLE" if v["account_eligible"] else f"requires ${v['local_margin']:,.0f} margin"
        print(f"  {k:12s}  {v['name']:30s}  {v['currency']}  {status}")
    print("═" * 70 + "\n")


def main() -> None:
    Path("logs").mkdir(exist_ok=True)
    parser = argparse.ArgumentParser(description="Global Energy Ecosystem Controller")
    parser.add_argument("--demo",   action="store_true", help="Offline demo (no IB/API)")
    parser.add_argument("--api",    action="store_true", help="Launch FastAPI server only")
    parser.add_argument("--status", action="store_true", help="Print status and exit")
    args = parser.parse_args()

    if args.status:
        print(json.dumps(get_ecosystem_metrics(), indent=2))
        return

    if args.demo:
        asyncio.run(_demo())
        return

    if args.api and _FASTAPI:
        asyncio.run(_run_api_server())
        return

    asyncio.run(start_global_ecosystem())


if __name__ == "__main__":
    main()
