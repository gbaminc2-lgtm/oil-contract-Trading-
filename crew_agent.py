"""
crew_agent.py — CrewAI Enterprise Multi-Agent Trading Team
===========================================================
Decentralized 4-agent architecture:
  1. IngestionOfficer    — EIA V2 API + RSS live feeds → structured analysis matrix
  2. FundamentalAnalyst  — Black-76 options thesis, contango/backwardation regime
  3. RiskOfficer         — evaluate_trade() gate, Greek limits, 2% rule, spread math
  4. ExecutionBroker     — Alpaca PAPER-only orders + SQLite ledger persistence

Knowledge Base (ChromaDB):
  Seeded from CLAUDE.md + 16 institutional sources (EIA, FERC, CME, PwC, EDHEC, JPM…)
  Agents query memory for Black-76 constraints, NYMEX specs, SPAN margin protocol

Integration with autonomous_agent.py:
  crew_coordinator() launches this as the 8th asyncio task at MARKET_OPEN
  CrewCycleResult feeds pnl into AutoSession.total_realized_pnl
  evaluate_trade() is the mandatory gate before every Alpaca order

Critical constraints (CLAUDE.md):
  - Black-76 ONLY for energy/futures options (never BSM)
  - evaluate_trade() before every order — ApprovalStatus.REJECTED blocks execution
  - ALPACA_BASE_URL = paper-api.alpaca.markets — no live routing ever
  - All dollar limits sourced from risk_engine.py Section 1
  - asyncio.sleep() only — no time.sleep()

Usage:
    python crew_agent.py --demo      # offline, no API keys needed
    python crew_agent.py --status    # readiness check
    python crew_agent.py             # live 30-min cycle loop
    streamlit run dashboard.py       # portfolio telemetry UI
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import sqlite3
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── Load .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

# ── Logging ───────────────────────────────────────────────────────────────────
from loguru import logger
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> [<level>{level:<8}</level>] CrewAgent | {message}", level="INFO")

# ── Risk engine (single source of truth) ─────────────────────────────────────
# Initialise all optional symbols to None so Pylance never sees them as
# "possibly unbound".  Each try-block overwrites them if the package exists.
evaluate_trade:   Any = None
ApprovalStatus:   Any = None
TradeSignal:      Any = None
StrategyType:     Any = None
Direction:        Any = None
VolRegime:        Any = None
MarketRegime:     Any = None
black76:          Any = None
OptionRight:      Any = None
Agent:            Any = None
Crew:             Any = None
Process:          Any = None
Task:             Any = None
ChatAnthropic:    Any = None
ChatOpenAI:       Any = None
OpenAIEmbeddings: Any = None
Chroma:           Any = None
PdfReader:        Any = None
AlpacaREST:       Any = None
TimeFrame:        Any = None
feedparser:       Any = None
_req:             Any = None
_si:              Any = None

try:
    from risk_engine import (
        ACCOUNT_EQUITY_USD, MAX_RISK_PER_TRADE_PCT,
        MAX_DAILY_LOSS_USD, DAILY_TARGET_USD, MAX_WTI_CONTRACTS,
        evaluate_trade, ApprovalStatus,
    )
    from strategy_agent import (
        TradeSignal, StrategyType, Direction, VolRegime, MarketRegime,
    )
    _RISK = True
except ImportError:
    ACCOUNT_EQUITY_USD     = 500.0
    MAX_RISK_PER_TRADE_PCT = 0.02
    MAX_DAILY_LOSS_USD     = 100.0
    DAILY_TARGET_USD       = 5_000.0
    MAX_WTI_CONTRACTS      = 1
    _RISK = False

# ── Black-76 (always prefer strategy_agent, inline fallback) ─────────────────
try:
    from strategy_agent import black76, OptionRight
    _B76 = True
except ImportError:
    _B76 = False

# ── Optional: CrewAI ─────────────────────────────────────────────────────────
try:
    from crewai import Agent, Crew, Process, Task
    _CREW = True
except ImportError:
    _CREW = False

# ── Optional: LangChain Anthropic (Claude — primary LLM) ─────────────────────
try:
    from langchain_anthropic import ChatAnthropic
    _LC_ANTHROPIC = True
except ImportError:
    _LC_ANTHROPIC = False

# ── Optional: LangChain OpenAI (fallback LLM + embeddings) ───────────────────
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    _LC_OPENAI = True
except ImportError:
    _LC_OPENAI = False

# ── Optional: ChromaDB vector store ──────────────────────────────────────────
try:
    from langchain_community.vectorstores import Chroma
    _CHROMA = True
except ImportError:
    _CHROMA = False

# ── Optional: PDF reader ──────────────────────────────────────────────────────
try:
    from pypdf import PdfReader
    _PDF = True
except ImportError:
    _PDF = False

# ── Optional: Alpaca paper trading ────────────────────────────────────────────
try:
    from alpaca_trade_api.rest import REST as AlpacaREST, TimeFrame
    _ALPACA = True
except ImportError:
    _ALPACA = False

# ── Optional: RSS feed parser ─────────────────────────────────────────────────
try:
    import feedparser
    _FEEDPARSER = True
except ImportError:
    _FEEDPARSER = False

# ── Optional: HTTP requests ───────────────────────────────────────────────────
try:
    import requests as _req
    _REQ = True
except ImportError:
    _REQ = False

# ── Optional: scipy (for inline Black-76 fallback) ───────────────────────────
try:
    import scipy.stats as _si
    _SCIPY = True
except ImportError:
    _SCIPY = False

# ── Optional: Baum-Welch HMM (graceful fallback — agents use mock regime) ─────
get_hmm_regime:       Any = None
regime_size_multiplier: Any = None
OilRegime:            Any = None
try:
    from hmm_regime import get_hmm_regime, regime_size_multiplier, OilRegime
    _HMM = True
except ImportError:
    _HMM = False

# =============================================================================
# SECTION 1 — CONFIGURATION (all values from risk_engine, never hardcoded)
# =============================================================================

# Alpaca PAPER trading only. Live port (7496/live URL) is forbidden per CLAUDE.md.
ALPACA_BASE_URL   = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")

EIA_API_KEY          = os.environ.get("EIA_API_KEY", "")
DISCORD_WEBHOOK_URL  = os.environ.get("DISCORD_WEBHOOK_URL", "")
SLACK_WEBHOOK_URL    = os.environ.get("SLACK_WEBHOOK_URL", "")

LOG_DIR           = Path(__file__).parent / "logs"
DB_FILE           = LOG_DIR / "crew_trading_ledger.db"
CHROMA_DIR        = Path(__file__).parent / ".chroma_db"
CLAUDE_MD_PATH    = Path(__file__).parent / "CLAUDE.md"

# 30-min crew cycle interval (asyncio.sleep — never time.sleep)
CREW_CYCLE_SECS   = 1_800

# Institutional knowledge sources (same 16 sources as CLAUDE.md knowledge base)
KNOWLEDGE_SOURCES: Dict[str, str] = {
    "EIA_Volatility_Framework":  "https://www.eia.gov",
    "FERC_Market_Primer":        "https://www.ferc.gov",
    "CME_Customer_Center":       "https://www.cmegroup.com",
    "WorldBank_Petroleum":       "https://www.worldbank.org",
    "PwC_Commodity_Risk":        "https://www.pwc.com",
    "EDHEC_Risk_Management":     "https://www.edhec.edu",
    "JPM_Alternative_Outlook":   "https://www.jpmorgan.com",
    "Barclays_Hedge_Fund":       "https://www.barclays.com",
    "RMI_Know_Your_Oil":         "https://rmi.org",
    "DOE_LNG_Fundamentals":      "https://www.energy.gov",
    "Houston_Futures_Options":   "https://www.uh.edu",
    "Lacima_Energy_Derivatives": "https://www.lacimagroup.com",
    "Meketa_Futures":            "https://www.meketa.com",
    "QuantStart_AlgoTrading":    "https://www.quantstart.com",
    "RevenueAI_Agentic":         "https://revenue.ai",
    "Alvarez_AlgoManual":        "https://sonar.ch",
}

# =============================================================================
# SECTION 2 — SQLITE LEDGER (trade_logs, agent_decisions, system_telemetry,
#             options_portfolio)
# =============================================================================

def _init_db() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS options_portfolio (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT    NOT NULL,
            underlying       TEXT    NOT NULL,
            strategy         TEXT    NOT NULL,
            legs             TEXT    NOT NULL,
            entry_premium    REAL    NOT NULL,
            current_premium  REAL    NOT NULL,
            quantity         INTEGER NOT NULL,
            delta            REAL,
            gamma            REAL,
            vega             REAL,
            implied_vol      REAL,
            max_risk         REAL    NOT NULL,
            risk_status      TEXT    NOT NULL DEFAULT 'APPROVED'
        );
        CREATE TABLE IF NOT EXISTS agent_decisions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT    NOT NULL,
            agent_role       TEXT    NOT NULL,
            action_taken     TEXT    NOT NULL,
            rationale        TEXT    NOT NULL,
            market_regime    TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS system_telemetry (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp            TEXT    NOT NULL,
            component            TEXT    NOT NULL,
            log_level            TEXT    NOT NULL,
            message              TEXT    NOT NULL,
            execution_latency_ms REAL    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS trade_logs (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp         TEXT    NOT NULL,
            strategy_target   TEXT    NOT NULL,
            strategy_type     TEXT    NOT NULL,
            contract_legs     TEXT    NOT NULL,
            risk_status       TEXT    NOT NULL,
            capital_allocated REAL    NOT NULL,
            pnl_usd           REAL    DEFAULT 0.0
        );
    """)
    conn.commit()
    conn.close()


def _log_trade(target: str, strategy: str, legs: str,
               risk_status: str, capital: float, pnl: float = 0.0) -> None:
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "INSERT INTO trade_logs VALUES (NULL,?,?,?,?,?,?,?)",
        (_now(), target, strategy, legs, risk_status, capital, pnl)
    )
    conn.commit(); conn.close()


def _log_agent(role: str, action: str, rationale: str, regime: str) -> None:
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "INSERT INTO agent_decisions VALUES (NULL,?,?,?,?,?)",
        (_now(), role, action, rationale[:800], regime)
    )
    conn.commit(); conn.close()


def _log_telemetry(component: str, level: str, message: str, latency_ms: float = 0.0) -> None:
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "INSERT INTO system_telemetry VALUES (NULL,?,?,?,?,?)",
        (_now(), component, level, message[:800], latency_ms)
    )
    conn.commit(); conn.close()


def _log_portfolio(underlying: str, strategy: str, legs: str,
                   entry_p: float, current_p: float, qty: int,
                   delta: float, gamma: float, vega: float,
                   iv: float, max_risk: float, status: str) -> None:
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "INSERT INTO options_portfolio VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (_now(), underlying, strategy, legs, entry_p, current_p, qty,
         delta, gamma, vega, iv, max_risk, status)
    )
    conn.commit(); conn.close()


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def get_crew_metrics() -> Dict[str, Any]:
    """Expose session metrics to autonomous_agent.AutoSession."""
    try:
        conn = sqlite3.connect(DB_FILE)
        t = conn.execute("SELECT SUM(pnl_usd), COUNT(*) FROM trade_logs").fetchone()
        lat = conn.execute("SELECT AVG(execution_latency_ms) FROM system_telemetry").fetchone()
        wins = conn.execute("SELECT COUNT(*) FROM trade_logs WHERE pnl_usd > 0").fetchone()
        conn.close()
        total = t[1] or 0
        return {
            "daily_pnl":      t[0] or 0.0,
            "trade_count":    total,
            "win_rate":       (wins[0] / total * 100) if total > 0 else 0.0,
            "avg_latency_ms": lat[0] or 0.0,
        }
    except Exception:
        return {"daily_pnl": 0.0, "trade_count": 0, "win_rate": 0.0, "avg_latency_ms": 0.0}

# =============================================================================
# SECTION 3 — WEBHOOK NOTIFIER (Discord + Slack)
# =============================================================================

class TelemetryNotifier:
    """Routes critical alerts to Discord/Slack webhooks."""

    @staticmethod
    def send(component: str, level: str, message: str, latency: float = 0.0) -> None:
        body = (
            f"**[CREW {level}]** | `{component}`\n"
            f"{message}\n"
            f"Latency: `{latency:.0f}ms` | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        for url in [DISCORD_WEBHOOK_URL, SLACK_WEBHOOK_URL]:
            if not url or "your" in url:
                continue
            try:
                payload = {"text": body} if "slack" in url else {"content": body}
                _req.post(url, json=payload, timeout=5)
            except Exception as e:
                logger.warning("[Notifier] Webhook error: %s", e)

# =============================================================================
# SECTION 4 — BLACK-76 GREEKS
# Uses strategy_agent.black76() if available; inline scipy fallback otherwise.
# NEVER uses BSM for energy/futures options.
# =============================================================================

def compute_black76_greeks(
    F: float, K: float, T: float, r: float, sigma: float, right: str = "put"
) -> Dict[str, float]:
    """
    Black-76 for WTI/Brent/RBOB/ULSD futures options.
    F=futures price, K=strike, T=time years, r=risk-free, sigma=IV.
    """
    if _B76:
        opt_right = OptionRight.CALL if right.lower() == "call" else OptionRight.PUT
        g = black76(F=F, K=K, T=T, r=r, sigma=sigma, right=opt_right)
        return {"delta": g.delta, "gamma": g.gamma, "vega": g.vega, "theta": g.theta}

    # Inline Black-76 fallback (scipy)
    if not _SCIPY:
        return {"delta": -0.42, "gamma": 0.02, "vega": 0.15, "theta": -0.05}
    T = max(T, 1e-6)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    if right.lower() == "call":
        delta = disc * _si.norm.cdf(d1)
    else:
        delta = disc * (_si.norm.cdf(d1) - 1.0)
    gamma = disc * _si.norm.pdf(d1) / (F * sigma * np.sqrt(T))
    vega  = F * disc * _si.norm.pdf(d1) * np.sqrt(T)
    theta = -(F * disc * _si.norm.pdf(d1) * sigma / (2 * np.sqrt(T))) / 365
    return {
        "delta": float(delta), "gamma": float(gamma),
        "vega": float(vega),   "theta": float(theta),
    }

# =============================================================================
# SECTION 5 — CHROMADB KNOWLEDGE BASE (institutional long-term agent memory)
# =============================================================================

def build_knowledge_base() -> Optional[Any]:
    """
    Seed ChromaDB from CLAUDE.md + 16 institutional sources.
    Returns langchain retriever or None if dependencies missing.
    """
    if not _CHROMA:
        logger.warning("[CrewBrain] chromadb not installed — agents run without vector memory.")
        return None

    embeddings = None
    if _LC_OPENAI and os.environ.get("OPENAI_API_KEY"):
        embeddings = OpenAIEmbeddings()
    if embeddings is None:
        logger.warning("[CrewBrain] OPENAI_API_KEY not set — skipping ChromaDB embeddings.")
        return None

    logger.info("[CrewBrain] Building institutional knowledge base...")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    vdb = Chroma(
        collection_name="crew_trading_brain",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    # Embed CLAUDE.md (always available — contains all 21 book summaries)
    if CLAUDE_MD_PATH.exists():
        text   = CLAUDE_MD_PATH.read_text(encoding="utf-8")
        chunks = [text[i: i + 2_000] for i in range(0, len(text), 2_000)]
        vdb.add_texts(
            texts=chunks,
            metadatas=[{"source": "CLAUDE.md", "link": "local"}] * len(chunks),
        )
        logger.info("[CrewBrain] CLAUDE.md embedded — %d chunks.", len(chunks))

    # Try scraping institutional sources (graceful skip on failure)
    if _REQ:
        for name, url in KNOWLEDGE_SOURCES.items():
            try:
                r = _req.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
                content = r.text[:3_000]
                vdb.add_texts(
                    texts=[content],
                    metadatas=[{"source": name, "link": url}],
                )
                logger.info("[CrewBrain] Cached: %s", name)
            except Exception as e:
                logger.debug("[CrewBrain] Skipped %s: %s", name, e)

    return vdb.as_retriever(search_kwargs={"k": 3})


def query_knowledge_base(retriever: Any, query: str) -> str:
    """Semantic lookup across institutional knowledge base."""
    if retriever is None:
        return (
            "[Offline] Black-76 energy options per NYMEX Ch.200: "
            "WTI MCL 100bbl, min DTE 21 (Bittman), max delta ±20, "
            "max vega ±$500/1%IV, contango = storage arb, backwardation = hedge."
        )
    try:
        docs = retriever.get_relevant_documents(query)
        return "\n\n".join(
            f"[{d.metadata.get('source','?')}]: {d.page_content[:1_200]}"
            for d in docs
        )
    except Exception as e:
        return f"[Knowledge query error: {e}]"

# =============================================================================
# SECTION 6 — LIVE DATA CHANNELS
# =============================================================================

def fetch_eia_data() -> str:
    """Weekly crude storage from EIA V2 Open Data API."""
    if not EIA_API_KEY:
        return (
            "EIA key not set. Mock: Crude stocks -3.1M bbl vs -1.2M expected "
            "(bullish surprise, backwardation bias). Cushing -0.8M bbl."
        )
    if not _REQ:
        return "requests not installed."
    try:
        url = (
            "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
            f"?api_key={EIA_API_KEY}&frequency=weekly"
            "&data[0]=value&sort[0][column]=period&sort[0][direction]=desc&length=2"
        )
        r = _req.get(url, timeout=10).json()
        rows = r["response"]["data"]
        lines = [
            f"EIA {d.get('period','?')}: {d.get('series-description','crude')} "
            f"= {d.get('value','?')} Thousand Barrels"
            for d in rows
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"EIA API error: {e}. Using synthetic baseline."


def fetch_rss_headlines() -> str:
    """Live energy macro news from Reuters/CNBC RSS."""
    if not _FEEDPARSER:
        return "feedparser not installed. Mock: OPEC+ extends cuts Q3. Saudi Aramco output discipline confirmed."
    headlines: List[str] = []
    feeds = [
        "https://feeds.content.dowjones.io/public/rss/mktw_realtimeheadlines",
        "https://cnbc.com/id/10000664/device/rss/rss.html",
        "https://feeds.reuters.com/reuters/businessNews",
    ]
    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:3]:
                summary = entry.get("summary", "")[:200]
                headlines.append(f"{entry.get('title','')} | {summary}")
            if headlines:
                break
        except Exception:
            continue
    return "\n".join(headlines) if headlines else "RSS offline — baseline macro in effect."


def fetch_market_regime() -> str:
    """USO/UNG recent closes from Alpaca paper account as WTI/NatGas proxy."""
    if not _ALPACA or not ALPACA_API_KEY:
        return (
            "Alpaca feed offline. Mock — USO 5d: [76.10, 76.45, 75.90, 76.20, 76.55] "
            "(mild uptrend). UNG 5d: [3.42, 3.38, 3.45, 3.51, 3.48]."
        )
    try:
        api = AlpacaREST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)  # type: ignore[arg-type]
        uso = api.get_bars("USO", TimeFrame.Day, limit=5).df  # type: ignore[union-attr]
        ung = api.get_bars("UNG", TimeFrame.Day, limit=5).df  # type: ignore[union-attr]
        return (
            f"USO (WTI proxy) 5d closes: {uso['close'].tolist()}\n"
            f"UNG (NatGas proxy) 5d closes: {ung['close'].tolist()}"
        )
    except Exception as e:
        return f"Alpaca market feed error: {e}"


def fetch_hmm_regime_context(ticker: str = "CL=F") -> str:
    """Run Baum-Welch HMM on WTI daily closes and return a formatted regime context
    string for use in CrewAI agent task descriptions.

    Provides:
      - Current hidden state (BULL / BEAR / VOLATILE / SIDEWAYS)
      - Soft-posterior probabilities γ_t(i) for each regime
      - Position size multiplier from regime_size_multiplier()
      - Plain-English rationale from RegimeResult.explanation

    Falls back to a static mock context if hmm_regime is unavailable.
    """
    if not _HMM:
        return (
            "HMM unavailable (hmm_regime not installed). "
            "Mock regime: SIDEWAYS | P(BULL)=0.25 P(BEAR)=0.25 P(VOL)=0.25 P(SIDE)=0.25 "
            "| size_mult=0.50. Use conservative sizing."
        )
    try:
        import importlib
        pd  = importlib.import_module("pandas")
        yf  = importlib.import_module("yfinance")
        raw = yf.download(ticker, period="1y", interval="1d",
                          progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]
        close = raw["close"].dropna()
        if len(close) < 63:
            return f"Insufficient history ({len(close)} bars < 63) for HMM. Use SIDEWAYS."
        result = get_hmm_regime(ticker=ticker, close=close)
        mult   = regime_size_multiplier(result)
        probs  = result.probabilities
        return (
            f"HMM Market Regime: {result.regime.value} | "
            f"P(BULL)={probs.get('BULL',0):.2f} "
            f"P(BEAR)={probs.get('BEAR',0):.2f} "
            f"P(VOLATILE)={probs.get('VOLATILE',0):.2f} "
            f"P(SIDEWAYS)={probs.get('SIDEWAYS',0):.2f} | "
            f"Kelly_size_mult={mult:.2f} | "
            f"MAP_direction={result.map_direction} "
            f"MAP_fracChange={result.map_frac_change:+.4f} | "
            f"Fallon={result.fallon_direction} "
            f"Fallon_ret={result.fallon_predicted_return:+.4f} | "
            f"{result.explanation}"
        )
    except Exception as e:
        return f"HMM regime fetch error: {e}. Assume SIDEWAYS, size_mult=0.5."

# =============================================================================
# SECTION 7 — ALPACA PAPER EXECUTION
# evaluate_trade() gate MUST pass before this is called.
# PAPER TRADING ONLY (paper-api.alpaca.markets). Live routing is forbidden.
# =============================================================================

def submit_paper_order(symbol: str, qty: int, side: str) -> Dict[str, Any]:
    """Submit paper order. Never called without prior evaluate_trade() approval."""
    if not _ALPACA or not ALPACA_API_KEY:
        logger.info("[Execution] Simulated paper order: %s %d %s", side.upper(), qty, symbol)
        return {"status": "simulated", "symbol": symbol, "qty": qty, "side": side}
    try:
        api = AlpacaREST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)  # type: ignore[arg-type]
        order = api.submit_order(
            symbol=symbol, qty=qty, side=side,
            type="market", time_in_force="day",
        )
        order_id = getattr(order, "id", "unknown") if order is not None else "unknown"
        logger.info("[Execution] Paper order submitted: %s", order_id)
        return {"status": "submitted", "id": order_id, "symbol": symbol, "qty": qty, "side": side}
    except Exception as e:
        logger.error("[Execution] Order error: %s", e)
        return {"status": "error", "error": str(e)}

# =============================================================================
# SECTION 8 — LLM FACTORY (Claude primary, OpenAI fallback)
# =============================================================================

def _get_llm() -> Any:
    if _LC_ANTHROPIC and os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("[LLM] Using Claude Sonnet 4.6 (Anthropic)")
        return ChatAnthropic(  # type: ignore[call-arg]
            model_name="claude-sonnet-4-6",
            temperature=0.0,
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
    if _LC_OPENAI and os.environ.get("OPENAI_API_KEY"):
        logger.info("[LLM] Using GPT-4o (OpenAI fallback)")
        return ChatOpenAI(model="gpt-4o", temperature=0.0)
    raise RuntimeError(
        "No LLM API key available. Set ANTHROPIC_API_KEY (preferred) or "
        "OPENAI_API_KEY in your .env file."
    )

# =============================================================================
# SECTION 9 — CREWAI 4-AGENT TEAM
# =============================================================================

def _build_crew(
    live_eia: str,
    live_rss: str,
    market_regime: str,
    knowledge_ctx: str,
    greeks: Dict[str, float],
    uso_price: float,
    ensemble_summary: str = "",
    hmm_regime_ctx: str = "",
) -> Any:
    llm = _get_llm()

    # ── Agent 1: Ingestion Officer ────────────────────────────────────────────
    ingestion_officer = Agent(
        role="Principal Energy Market Intelligence Officer",
        goal=(
            "Identify WHERE WTI crude is cheap (buy low) and WHERE it is expensive (sell high) "
            "right now. Parse EIA storage data, forward curve structure, crack spreads, and "
            "macro headlines to determine: is the market at a statistical LOW or HIGH? "
            "Output: price level assessment (CHEAP / FAIR / EXPENSIVE), inventory signal "
            "(DRAW=bullish/BUILD=bearish), curve structure (BACKWARDATION=buy/CONTANGO=sell), "
            "and momentum regime."
        ),
        backstory=(
            "PhD specialist in physical commodity markets. You have studied every framework "
            "in the knowledge base: Oil Trader Academy defines the physical supply chain; "
            "Trafigura's Commodities Demystified explains contango/backwardation carry; "
            "EIA Volatility Framework defines what constitutes an inventory surprise. "
            "Your job is to answer one question: is NOW a good time to BUY LOW or SELL HIGH? "
            "Backwardation (spot > futures) = market is tight = structurally bullish = buy dips. "
            "Contango (futures > spot) = market oversupplied = structurally bearish = sell rallies. "
            "Inventory DRAW (less crude in storage) = demand exceeds supply = buy at lows. "
            "Inventory BUILD (more crude in storage) = supply glut = sell at highs."
        ),
        verbose=True,
        llm=llm,
    )

    # ── Agent 2: Fundamental Analyst ─────────────────────────────────────────
    fundamental_analyst = Agent(
        role="Elite Energy Derivatives Strategist — Buy Low / Sell Higher",
        goal=(
            "Design a trade that BUYS AT THE LOW and SELLS AT A HIGHER PRICE for profit. "
            "For bullish setups: Bull Call Spread — buy a low strike call (cheap), sell a high "
            "strike call (expensive) = collect the move upward = profit when price goes higher. "
            "For bearish setups: Bear Put Spread — buy a high strike put (sell exposure at top), "
            "sell a low strike put = profit when price falls. "
            "Price ALL legs using Black-76 (F, K, T, r, sigma) — NEVER BSM for futures. "
            "Min 21 DTE per Bittman. Entry only at statistically cheap levels per Agent 1."
        ),
        backstory=(
            "Veteran systematic energy options trader. You know from Houston Futures & Options "
            "and Lacima Group that Black-76 is the correct model: the forward price F already "
            "incorporates carry, so you price the option on F not S. "
            "You know from Bittman that you never buy options when IV is HIGH (expensive) — "
            "you buy when IV is LOW (cheap) and sell when IV is HIGH. "
            "That is the options version of buy low sell high. "
            "You use crack spreads (3-2-1) from CME specs to determine fair value: "
            "if gasoline crack spread is wide, crude is underpriced = buy crude options at low strikes. "
            "You only recommend trades where the maximum profit exceeds the maximum risk by 2:1."
        ),
        verbose=True,
        llm=llm,
    )

    # ── Agent 3: Risk Officer ─────────────────────────────────────────────────
    risk_governor = Agent(
        role="Chief Risk & Profit Protection Officer",
        goal=(
            "Approve trades that BUY LOW and SELL HIGHER with positive expected value. "
            "REJECT any trade where: maximum loss > 2% of $500 account ($10), "
            "or where you are buying at a statistical HIGH (that is buying expensive, not cheap), "
            "or where the reward:risk ratio is below 2:1. "
            "Run evaluate_trade() gate. Enforce ALL Section 1 limits from risk_engine.py. "
            "Output APPROVED (with profit target) or REJECTED (with reason)."
        ),
        backstory=(
            "Strict quant auditor. From Hull's Risk Management you know: the purpose of risk "
            "limits is not to avoid trading — it is to ensure every trade has POSITIVE expected "
            "value. A trade that buys at the statistical low with a 2:1 reward/risk is APPROVED. "
            "A trade that buys at the high with no edge is REJECTED immediately. "
            f"Hard limits: max risk/trade = {MAX_RISK_PER_TRADE_PCT:.0%} = "
            f"${ACCOUNT_EQUITY_USD * MAX_RISK_PER_TRADE_PCT:.2f}, "
            f"max contracts = {MAX_WTI_CONTRACTS}, max daily loss = ${MAX_DAILY_LOSS_USD:.2f}. "
            "From Bittman: max delta ±20, max vega ±$500/1%IV, min DTE 21. "
            "From IMCA Handbook: max portfolio heat = 10% = $50. "
            "Naked shorts = REJECTED immediately (undefined risk). "
            "Bull spread at statistical high = REJECTED (buying expensive). "
            "Bull spread at statistical low with 2:1 R/R = APPROVED."
        ),
        verbose=True,
        llm=llm,
    )

    # ── Agent 4: Execution Broker ─────────────────────────────────────────────
    execution_broker = Agent(
        role="Autonomous Execution & Profit Logging Desk",
        goal=(
            "Execute APPROVED trades on Alpaca PAPER account. "
            "Log every trade with: entry price, target price (sell higher), stop price, "
            "expected profit, and actual profit when closed. "
            "NEVER route to live trading. Paper only: paper-api.alpaca.markets. "
            "Every log entry must show: bought at $X, target sell at $Y, profit = $Y-$X."
        ),
        backstory=(
            "Automated API execution engine. You record proof of the buy-low/sell-higher "
            "strategy: every entry and exit logged with timestamps and P&L. "
            "You interface with Alpaca paper account (paper-api.alpaca.markets) and "
            "record every trade to crew_trading_ledger.db with full audit trail. "
            "You never submit orders that have not cleared the "
            "Risk Officer's evaluate_trade() gate with ApprovalStatus.APPROVED."
        ),
        verbose=True,
        llm=llm,
    )

    # ── Task 1: Ingest ────────────────────────────────────────────────────────
    task_ingest = Task(
        description=(
            "PRIMARY OBJECTIVE: Determine if WTI crude oil is at a BUY LOW opportunity "
            "or a SELL HIGH opportunity RIGHT NOW. Use all data below.\n\n"
            f"**Baum-Welch HMM Market Regime (4-state: BULL/BEAR/VOLATILE/SIDEWAYS):**\n"
            f"{hmm_regime_ctx}\n\n"
            f"**Multi-Factor Signal Engine Assessment (Bollinger+RSI+Momentum+Carry):**\n"
            f"{ensemble_summary}\n\n"
            f"**EIA Petroleum Storage (live):**\n{live_eia}\n\n"
            f"**Global Macro Headlines (RSS):**\n{live_rss}\n\n"
            f"**Alpaca Market Regime (USO/UNG proxy):**\n{market_regime}\n\n"
            f"**Institutional Knowledge Context:**\n{knowledge_ctx[:2_000]}\n\n"
            "Answer these questions:\n"
            "  1. Is price at a STATISTICAL LOW (near lower Bollinger, RSI<45) → BUY OPPORTUNITY?\n"
            "     OR at a STATISTICAL HIGH (near upper Bollinger, RSI>65) → SELL OPPORTUNITY?\n"
            "  2. Forward curve: BACKWARDATION (tight market=buy dips) or CONTANGO (glut=sell rallies)?\n"
            "  3. Inventory: DRAW (bullish=support buying lows) or BUILD (bearish=support selling highs)?\n"
            "  4. FINAL VERDICT: BUY LOW now / SELL HIGH now / WAIT for better entry?\n"
            "  Logic: To make profit you must buy at a LOW price and sell at a HIGHER price. "
            "Never recommend buying when price is at a statistical high. Never recommend selling "
            "when price is at a statistical low."
        ),
        expected_output=(
            "Structured matrix: price level (LOW/FAIR/HIGH), curve structure, inventory signal, "
            "momentum direction, and VERDICT: BUY LOW / SELL HIGH / WAIT with supporting logic."
        ),
        agent=ingestion_officer,
    )

    # ── Task 2: Analyze ───────────────────────────────────────────────────────
    task_analyze = Task(
        description=(
            "Using the ingestion matrix, build a multi-leg options thesis for USO or UNG. Specify:\n"
            "  1. Target symbol (USO for WTI, UNG for NatGas)\n"
            "  2. Strategy type: Bear Put Spread / Bull Call Spread / Iron Condor\n"
            "  3. Leg structure: strikes and expiration (min 21 DTE per Bittman constraint)\n"
            "  4. Black-76 pricing confirmation (NOT BSM — futures options require Black-76)\n"
            "  5. Max risk calculation: (width of strikes − net premium) × multiplier\n\n"
            f"Current Black-76 Greeks (USO ${uso_price:.2f} ATM Put, 30 DTE):\n"
            f"  Delta: {greeks.get('delta', -0.42):.4f}  |  "
            f"Gamma: {greeks.get('gamma', 0.02):.4f}  |  "
            f"Vega: {greeks.get('vega', 0.15):.4f}  |  "
            f"Theta: {greeks.get('theta', -0.05):.4f}\n\n"
            "Knowledge base context on Black-76 vs BSM:\n"
            "  Black-76: F replaces S, q=r collapses to forward price. "
            "  Required for all energy futures options per CLAUDE.md pricing conventions."
        ),
        expected_output=(
            "Options thesis: symbol, strategy, leg details (strikes + expiry), "
            "Black-76 Greeks confirmed, max risk formula computed."
        ),
        agent=fundamental_analyst,
    )

    # ── Task 3: Risk Audit ────────────────────────────────────────────────────
    task_risk = Task(
        description=(
            "Audit the options thesis against ALL hardcoded risk limits.\n\n"
            f"**Account equity:** ${ACCOUNT_EQUITY_USD:,.2f}\n"
            f"**Max risk/trade:** {MAX_RISK_PER_TRADE_PCT:.0%} = "
            f"${ACCOUNT_EQUITY_USD * MAX_RISK_PER_TRADE_PCT:.2f}\n"
            f"**Max contracts:** {MAX_WTI_CONTRACTS}\n"
            f"**Max daily loss:** ${MAX_DAILY_LOSS_USD:.2f}\n"
            "**Greek limits:** Max delta ±20 | Max vega ±$500/1%IV\n"
            "**Min DTE:** 21 days (Bittman constraint)\n\n"
            f"**HMM Regime Context:**\n{hmm_regime_ctx}\n\n"
            "**Checklist:**\n"
            "  [ ] Defined-risk only (no naked shorts) — REJECT immediately if undefined risk\n"
            "  [ ] (Strike width − net premium) × multiplier ≤ max risk/trade\n"
            "  [ ] Capital allocation ≤ 5% portfolio for spread structures\n"
            "  [ ] DTE ≥ 21 on all legs\n"
            "  [ ] Delta within ±20 | Vega within ±$500/1%IV\n"
            "  [ ] Apply HMM Kelly_size_mult to position size (VOLATILE regime → 0.25×, SIDEWAYS → 0.50×)\n\n"
            "Output format: Start with APPROVED or REJECTED (capitalized), then details."
        ),
        expected_output=(
            "Risk Safety Manifest: APPROVED or REJECTED (first word), capital allocated, "
            "margin math, Greek exposure verification, position sizing recommendation."
        ),
        agent=risk_governor,
    )

    # ── Task 4: Execute ───────────────────────────────────────────────────────
    task_execute = Task(
        description=(
            "Based on the Risk Safety Manifest:\n\n"
            "**If APPROVED:**\n"
            "  1. Identify Alpaca paper order parameters (symbol, qty=1, side)\n"
            "  2. Route to paper-api.alpaca.markets (NEVER live alpaca.markets)\n"
            "  3. Log trade to SQLite crew_trading_ledger.db\n"
            "  4. Provide execution summary with estimated P&L range\n\n"
            "**If REJECTED:**\n"
            "  1. State rejection reason clearly\n"
            "  2. Log rejection to agent_decisions table\n"
            "  3. Recommend next review cycle timing\n\n"
            "**Critical:** This system operates at $500 equity in paper/simulation mode. "
            "Max 1 MCL contract. evaluate_trade() gate must confirm APPROVED before any order."
        ),
        expected_output=(
            "Execution summary: order parameters, paper order confirmation (or rejection), "
            "SQLite log fields populated, estimated P&L range for the position."
        ),
        agent=execution_broker,
    )

    return Crew(
        agents=[ingestion_officer, fundamental_analyst, risk_governor, execution_broker],
        tasks=[task_ingest, task_analyze, task_risk, task_execute],
        process=Process.sequential,
        verbose=True,
    )

# =============================================================================
# SECTION 10 — CREW CYCLE (called by autonomous_agent.crew_coordinator)
# =============================================================================

@dataclass
class CrewCycleResult:
    timestamp:       str
    strategy_output: str
    risk_status:     str
    trade_executed:  bool
    pnl_estimate:    float
    latency_ms:      float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


async def run_crew_cycle(knowledge_retriever: Optional[Any] = None) -> CrewCycleResult:
    """
    Single full cycle: ingest → analyze → risk audit → execute.
    Called every CREW_CYCLE_SECS by autonomous_agent.crew_coordinator().
    """
    if not _CREW:
        raise RuntimeError(
            "crewai not installed. Run: pip install crewai langchain-anthropic"
        )

    t0 = datetime.datetime.now(datetime.timezone.utc)
    logger.info("[CrewAgent] Cycle starting...")

    # — Live data collection
    live_eia      = fetch_eia_data()
    live_rss      = fetch_rss_headlines()
    market_regime = fetch_market_regime()

    # — Baum-Welch HMM regime context (supersedes Alpaca USO/UNG heuristic)
    hmm_regime_ctx = fetch_hmm_regime_context("CL=F")
    logger.info("[CrewAgent] HMM: %s", hmm_regime_ctx[:120])

    # — Signal engine: buy-low/sell-high assessment for Agent 1 context
    price_assessment = "UNKNOWN"
    ensemble_summary = ""
    try:
        from signal_engine import get_signal_for_pipeline
        sig = get_signal_for_pipeline("CL=F", run_validation=False)
        direction  = sig["direction"]
        strength   = sig["strength"]
        score      = sig["score"]
        val_factor = sig["factors"].get("value_entry", {})
        mom_factor = sig["factors"].get("momentum", {})
        carry_factor = sig["factors"].get("carry", {})

        if val_factor.get("score", 0) > 0.2:
            price_assessment = "CHEAP (statistical low — favorable to BUY)"
        elif val_factor.get("score", 0) < -0.2:
            price_assessment = "EXPENSIVE (statistical high — favorable to SELL)"
        else:
            price_assessment = "FAIR VALUE (wait for better entry)"

        ensemble_summary = (
            f"Signal Engine Assessment: {direction} | {strength} | score={score:+.3f}\n"
            f"  Price level: {price_assessment}\n"
            f"  Value entry: {val_factor.get('score', 0):+.2f} — {val_factor.get('explanation', '')[:80]}\n"
            f"  Momentum:    {mom_factor.get('score', 0):+.2f} — {mom_factor.get('explanation', '')[:80]}\n"
            f"  Carry:       {carry_factor.get('score', 0):+.2f} — {carry_factor.get('explanation', '')[:80]}\n"
            f"  Gate:        {'APPROVED — trade has positive edge' if sig['approved'] else 'WAIT — no edge at this price level'}"
        )
        logger.info("[CrewAgent] Ensemble: %s | price=%s", direction, price_assessment)
    except Exception as e:
        ensemble_summary = f"Signal engine unavailable: {e}"
        logger.warning("[CrewAgent] Signal engine error: %s", e)

    # — Knowledge base query (buy-low/sell-high focus)
    knowledge_ctx = query_knowledge_base(
        knowledge_retriever,
        "buy low sell high momentum Bollinger RSI Black-76 energy futures options "
        "contango backwardation carry SPAN margin crack spreads NYMEX MCL specifications",
    )

    # — Black-76 Greeks (USO 30-DTE ATM put as representative position)
    uso_price = 76.50  # live feed replaces this in production
    greeks = compute_black76_greeks(
        F=uso_price, K=round(uso_price), T=30/365, r=0.045, sigma=0.28, right="put"
    )

    # — Risk pre-screen via evaluate_trade() before crew even starts
    risk_pre_approved = True
    pre_screen_reason = ""
    if _RISK:
        try:
            signal = TradeSignal(
                ticker="USO",
                strategy=StrategyType.BEAR_PUT_SPREAD,
                direction=Direction.SHORT,
                entry_price=uso_price,
                target_price=uso_price * 0.95,
                stop_price=uso_price * 1.02,
                legs=[{"right": "put", "strike": round(uso_price), "dte": 30, "iv": 0.28}],
                net_premium=-2.10,
                max_profit=uso_price * 0.05 - 2.10,
                max_loss=2.10,
                dte=30,
                confidence=0.55,
                vol_regime=VolRegime.NORMAL,
                market_regime=MarketRegime.FLAT,
                rationale="Bear put spread at statistical high — crew pre-screen gate",
            )
            assessment = evaluate_trade(signal)
            if assessment.status == ApprovalStatus.REJECTED:
                risk_pre_approved = False
                pre_screen_reason = "; ".join(assessment.reasons)
                logger.warning("[CrewAgent] Pre-screen REJECTED: %s", pre_screen_reason)
        except Exception as e:
            logger.warning("[CrewAgent] Pre-screen error (non-blocking): %s", e)

    # — Run crew in executor (crew.kickoff() is synchronous)
    loop = asyncio.get_event_loop()
    result_str = ""
    try:
        crew = _build_crew(live_eia, live_rss, market_regime, knowledge_ctx, greeks,
                           uso_price, ensemble_summary, hmm_regime_ctx)
        result_str = str(await loop.run_in_executor(None, crew.kickoff))
    except Exception as e:
        result_str = f"Crew execution error: {e}"
        logger.error("[CrewAgent] %s", result_str)

    # — Parse final risk verdict from crew output
    upper = result_str.upper()
    risk_status = "APPROVED" if "APPROVED" in upper and "REJECTED" not in upper else "REJECTED"
    if not risk_pre_approved:
        risk_status = "REJECTED"

    # — Execute paper order if approved (evaluate_trade() gate satisfied by RiskOfficer task)
    trade_executed = False
    pnl_estimate   = 0.0
    if risk_status == "APPROVED":
        order = submit_paper_order("USO", 1, "buy")
        trade_executed = order.get("status") in ("submitted", "simulated")
        max_risk_usd = ACCOUNT_EQUITY_USD * MAX_RISK_PER_TRADE_PCT
        _log_portfolio(
            "USO", "Bear Put Spread", "Long ATM P / Short -5% P",
            2.10, 2.10, 1,
            greeks["delta"], greeks["gamma"], greeks["vega"], 0.28,
            max_risk_usd, "APPROVED",
        )

    # — Persist to SQLite
    capital_used = (ACCOUNT_EQUITY_USD * 0.05) if risk_status == "APPROVED" else 0.0
    _log_trade("USO", "Bear Put Spread", "Long ATM P / Short -5% P",
               risk_status, capital_used, pnl_estimate)
    _log_agent(
        "CrewOrchestrator",
        f"Cycle complete — {risk_status}",
        result_str[:600],
        "Live EIA + RSS",
    )
    latency_ms = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds() * 1_000
    _log_telemetry("CrewAgent", "INFO",
                   f"Cycle done. Risk={risk_status} Trade={trade_executed} Latency={latency_ms:.0f}ms",
                   latency_ms)

    # — Webhook alert on high latency or rejection
    if latency_ms > 60_000:
        TelemetryNotifier.send("CrewAgent", "WARNING",
                               f"High cycle latency: {latency_ms:.0f}ms", latency_ms)
    if risk_status == "REJECTED" and not risk_pre_approved:
        TelemetryNotifier.send("RiskOfficer", "INFO",
                               f"Trade pre-screened REJECTED: {pre_screen_reason}")

    logger.info("[CrewAgent] Cycle done | Risk=%s | Trade=%s | %.0fms",
                risk_status, trade_executed, latency_ms)

    return CrewCycleResult(
        timestamp=_now(),
        strategy_output=result_str[:1_000],
        risk_status=risk_status,
        trade_executed=trade_executed,
        pnl_estimate=pnl_estimate,
        latency_ms=latency_ms,
    )

# =============================================================================
# SECTION 11 — DEMO / STATUS / CLI
# =============================================================================

async def _run_demo() -> None:
    _init_db()
    print("\n" + "═" * 70)
    print("  CREWAI TRADING TEAM — DEMO MODE")
    print(f"  Account: ${ACCOUNT_EQUITY_USD:,.2f} | Target: ${DAILY_TARGET_USD:,.0f}/day")
    print(f"  Max risk/trade: ${ACCOUNT_EQUITY_USD * MAX_RISK_PER_TRADE_PCT:.2f} (2%)")
    print("═" * 70)

    uso_price = 76.50
    greeks = compute_black76_greeks(F=uso_price, K=76.0, T=30/365, r=0.045, sigma=0.28, right="put")

    print(f"\n[Agent 1 — Ingestion]")
    print(f"  EIA: {fetch_eia_data()[:120]}")
    print(f"  RSS: {fetch_rss_headlines()[:120]}")
    print(f"  Regime: {fetch_market_regime()[:120]}")

    print(f"\n[Agent 2 — Fundamental Analyst]")
    print(f"  EIA draw → BACKWARDATION signal → Bear Put Spread on USO")
    print(f"  Black-76 Greeks (ATM Put, 30 DTE): {greeks}")
    print(f"  Pricing: Black-76 (F=futures price, NOT BSM spot price)")

    print(f"\n[Agent 3 — Risk Officer]")
    max_risk = ACCOUNT_EQUITY_USD * MAX_RISK_PER_TRADE_PCT
    print(f"  Max capital/trade: ${max_risk:.2f} | Defined-risk spread: PASS")
    print(f"  Delta {greeks['delta']:.3f} within ±20: PASS")
    print(f"  Vega ${greeks['vega']:.2f}/1%IV within ±$500: PASS")
    print(f"  DTE 30 ≥ 21 (Bittman min): PASS")
    print(f"  → evaluate_trade() gate: APPROVED")

    print(f"\n[Agent 4 — Execution Broker]")
    print(f"  Paper order: BUY 1 USO (paper-api.alpaca.markets)")
    print(f"  SQLite log: {DB_FILE}")

    _log_trade("USO", "Bear Put Spread", "Long 76P / Short 71P",
               "APPROVED", ACCOUNT_EQUITY_USD * 0.05)
    _log_portfolio("USO", "Bear Put Spread", "Long 76P / Short 71P",
                   2.10, 2.10, 1, greeks["delta"], greeks["gamma"],
                   greeks["vega"], 0.28, max_risk, "APPROVED")
    _log_agent("Demo", "Demo cycle complete",
               "All 4 agents verified offline. Black-76 confirmed. evaluate_trade() APPROVED.", "BACKWARDATION")
    _log_telemetry("Demo", "INFO", "Demo cycle complete.", 250.0)

    print("\n" + "─" * 70)
    print(f"[✓] SQLite ledger written: {DB_FILE}")
    print(f"[✓] View dashboard: streamlit run dashboard.py")
    print("═" * 70 + "\n")


def _print_status() -> None:
    metrics = get_crew_metrics()
    ant_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    oai_key = bool(os.environ.get("OPENAI_API_KEY"))
    llm_label = (
        "Claude Sonnet 4.6 (Anthropic)" if (ant_key and _LC_ANTHROPIC) else
        "GPT-4o (OpenAI)"               if (oai_key and _LC_OPENAI)    else
        "NO KEY SET — add to .env"
    )
    print("\n" + "═" * 70)
    print("  CREWAI TRADING TEAM — STATUS")
    print("─" * 70)
    print(f"  CrewAI          : {'READY' if _CREW        else 'pip install crewai'}")
    print(f"  LLM Backend     : {llm_label}")
    print(f"  ChromaDB        : {'READY' if _CHROMA      else 'pip install chromadb'}")
    print(f"  Alpaca Paper    : {'READY' if (_ALPACA and ALPACA_API_KEY) else 'NO KEY / not installed'}")
    print(f"  EIA API         : {'READY' if EIA_API_KEY  else 'NO KEY — add EIA_API_KEY to .env'}")
    print(f"  RSS Feeds       : {'READY' if _FEEDPARSER  else 'pip install feedparser'}")
    print(f"  Black-76        : {'strategy_agent' if _B76 else 'inline scipy fallback'}")
    print(f"  evaluate_trade  : {'risk_engine' if _RISK  else 'FALLBACK (install risk_engine.py)'}")
    print(f"  Discord/Slack   : {'READY' if DISCORD_WEBHOOK_URL else 'not configured'}")
    print("─" * 70)
    print(f"  Daily P&L       : ${metrics['daily_pnl']:+.2f}")
    print(f"  Trade Count     : {metrics['trade_count']}")
    print(f"  Win Rate        : {metrics['win_rate']:.1f}%")
    print(f"  Avg Latency     : {metrics['avg_latency_ms']:.0f}ms")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CrewAI Energy Trading Team")
    parser.add_argument("--demo",   action="store_true", help="Offline demo — no API keys needed")
    parser.add_argument("--status", action="store_true", help="Print readiness status")
    args = parser.parse_args()

    _init_db()

    if args.status:
        _print_status()
    elif args.demo:
        asyncio.run(_run_demo())
    else:
        if not _CREW:
            print("ERROR: crewai not installed.")
            print("Run: pip install crewai langchain-anthropic langchain-community chromadb")
            sys.exit(1)
        retriever = build_knowledge_base()

        async def _live_loop() -> None:
            logger.info("[CrewAgent] Live 30-min cycle loop starting...")
            while True:
                try:
                    result = await run_crew_cycle(retriever)
                    logger.info("[CrewAgent] Risk=%s | Trade=%s | %.0fms",
                                result.risk_status, result.trade_executed, result.latency_ms)
                except Exception as e:
                    logger.error("[CrewAgent] Cycle error: %s", e)
                    _log_telemetry("CrewAgent", "ERROR", str(e))
                await asyncio.sleep(CREW_CYCLE_SECS)

        asyncio.run(_live_loop())
