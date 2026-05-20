"""
risk_engine.py — Hardcoded Risk Parameters & Position Sizing
============================================================
Sources: Handbook of Risk (IMCA/Wiley), Risk Management and Financial
Institutions (John Hull 4th Ed.), How to Price and Trade Options (Sherbin),
Trading Options as a Professional (Bittman), Art & Science of Technical
Analysis (Grimes), Complete Guide to Day Trading (Heitkoetter),
Oil Trader Academy / NYMEX Chapter 200.

Responsibilities:
  - Hardcoded firm-level risk limits (never overridden at runtime)
  - Position sizing: Kelly fraction, fixed-fractional, Greeks-based
  - Portfolio Greeks aggregator & limit checker
  - Value-at-Risk (parametric & historical)
  - Expected Shortfall / CVaR
  - Drawdown tracker
  - Oil-specific lot sizing (NYMEX Chapter 200 specs)
  - Trade approval gate: returns APPROVED / REJECTED with reason

All dollar values in USD. All Greeks use per-contract (100-share) conventions.
"""

from __future__ import annotations

import datetime
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategy_agent import Greeks as BSMResult, TradeSignal, Direction, StrategyType

logger = logging.getLogger(__name__)

# ============================================================================
# SECTION 1 — HARDCODED RISK PARAMETERS
# These are intentionally hardcoded and must NOT be loaded from config files
# or environment variables without a formal risk-committee sign-off.
# Hull Ch.16: "The first line of risk management is hard limits."
# ============================================================================

# ---------------------------------------------------------------------------
# Account & Allocation Limits
# Starting capital: $500.00  |  Daily target: $5,000
# Heitkoetter Ch.4: scale all dollar limits proportionally to account size.
# ---------------------------------------------------------------------------
ACCOUNT_EQUITY_USD           = 500.00
MAX_RISK_PER_TRADE_PCT       = 0.02
MAX_RISK_PER_TRADE_USD       = ACCOUNT_EQUITY_USD * MAX_RISK_PER_TRADE_PCT   # $10.00

MAX_PORTFOLIO_RISK_PCT       = 0.10
MAX_PORTFOLIO_RISK_USD       = ACCOUNT_EQUITY_USD * MAX_PORTFOLIO_RISK_PCT   # $50.00

MAX_SECTOR_CONCENTRATION_PCT = 0.25
MAX_SINGLE_TICKER_PCT        = 0.15

# ---------------------------------------------------------------------------
# Options-Specific Limits (Bittman / Sherbin — scaled to $500 account)
# ---------------------------------------------------------------------------
MAX_OPEN_OPTIONS_POSITIONS   = 3
MAX_CONTRACTS_PER_POSITION   = 1
MIN_DTE_ENTRY                = 21
MAX_DTE_ENTRY                = 90
MIN_OPTION_LIQUIDITY_OI      = 100
MIN_BID_ASK_WIDTH_FILTER     = 0.50
MAX_IV_RANK_FOR_DEBIT        = 50
MIN_IV_RANK_FOR_CREDIT       = 50

# Portfolio Greeks hard limits (scaled to $500 micro account)
MAX_PORTFOLIO_DELTA          = 20.0
MAX_PORTFOLIO_GAMMA          = 5.0
MAX_PORTFOLIO_THETA_NEGATIVE = -50.0
MAX_PORTFOLIO_VEGA           = 500.0

# ---------------------------------------------------------------------------
# Futures-Specific Limits — Micro WTI MCL / NYMEX
# MCL: 100 bbl/contract, tick=$0.01/bbl=$1.00, margin~$1,000
# With $500 account: paper-trade MCL; 1 contract max; tight $0.10/bbl stops
# ---------------------------------------------------------------------------
WTI_CONTRACT_SIZE_BBL        = 100         # MCL micro contract (1/10 of CL)
WTI_INITIAL_MARGIN_USD       = 1_000       # MCL approximate initial margin
WTI_MAINTENANCE_MARGIN_USD   = 800         # MCL maintenance margin
MAX_WTI_CONTRACTS            = 1           # 1 micro contract max at $500
MAX_WTI_LOSS_PER_CONTRACT_USD= 10.00       # 2% of $500 = $10 hard limit
WTI_TICK_VALUE_USD           = 1.00        # MCL tick value ($0.01/bbl × 100 bbl)

# ---------------------------------------------------------------------------
# Day-Trading Limits (Heitkoetter Ch.3 — scaled to $500)
# $100 daily loss limit = 20% of equity (protective circuit-breaker)
# ---------------------------------------------------------------------------
MAX_DAILY_LOSS_USD           = 100
MAX_TRADES_PER_DAY           = 5
MAX_CONSECUTIVE_LOSSES       = 3

# ---------------------------------------------------------------------------
# VaR Parameters (Hull Ch.9 / IMCA Handbook — scaled to $500)
# ---------------------------------------------------------------------------
VAR_CONFIDENCE_LEVEL         = 0.99
VAR_LOOKBACK_DAYS            = 252
MAX_VAR_USD                  = 50          # 10% of $500 equity
EXPECTED_SHORTFALL_MULTIPLIER= 1.3

# ---------------------------------------------------------------------------
# Kelly / Position Sizing
# ---------------------------------------------------------------------------
KELLY_FRACTION_CAP           = 0.25
WIN_RATE_CONSERVATIVE        = 0.50
PAYOFF_RATIO_CONSERVATIVE    = 1.5

# ---------------------------------------------------------------------------
# Daily Profit Target
# ---------------------------------------------------------------------------
DAILY_TARGET_USD             = 5_000.0     # $5,000/day target (paper-trading)


# ============================================================================
# SECTION 2 — DATA CONTAINERS
# ============================================================================

class ApprovalStatus(str, Enum):
    APPROVED  = "APPROVED"
    REJECTED  = "REJECTED"
    FLAGGED   = "FLAGGED"


@dataclass
class PortfolioGreeks:
    """Aggregate Greeks across all open positions."""
    net_delta:  float = 0.0
    net_gamma:  float = 0.0
    net_theta:  float = 0.0
    net_vega:   float = 0.0
    net_rho:    float = 0.0

    def add_bsm(self, bsm: BSMResult, qty: float) -> None:
        """qty: positive = long, negative = short; multiply by 100 (shares per contract)."""
        multiplier = qty * 100
        self.net_delta += bsm.delta * multiplier
        self.net_gamma += bsm.gamma * multiplier
        self.net_theta += bsm.theta * multiplier
        self.net_vega  += bsm.vega  * multiplier
        self.net_rho   += bsm.rho   * multiplier


@dataclass
class RiskAssessment:
    """Full risk assessment for a proposed trade. Returned by evaluate_trade()."""
    trade:              TradeSignal
    status:             ApprovalStatus
    reasons:            List[str]
    approved_qty:       int
    position_size_usd:  float
    max_loss_usd:       float
    risk_pct_account:   float
    kelly_fraction:     float
    var_1day_99:        float
    expected_shortfall: float
    timestamp:          datetime.datetime = field(default_factory=datetime.datetime.utcnow)

    def summary(self) -> str:
        lines = [
            f"{'='*60}",
            f"TRADE: {self.trade.ticker} | {self.trade.strategy.value}",
            f"STATUS: {self.status.value}",
            f"Approved Qty: {self.approved_qty} contracts",
            f"Max Loss: ${self.max_loss_usd:,.0f} ({self.risk_pct_account:.2%} of account)",
            f"Kelly Fraction: {self.kelly_fraction:.2%}",
            f"1-Day 99% VaR: ${self.var_1day_99:,.0f}",
            f"Expected Shortfall: ${self.expected_shortfall:,.0f}",
        ]
        if self.reasons:
            lines.append("Reasons: " + "; ".join(self.reasons))
        lines.append("="*60)
        return "\n".join(lines)


@dataclass
class DrawdownTracker:
    """
    Tracks peak equity and current drawdown.

    Hull Ch.15 / IMCA: a 50% drawdown requires a 100% recovery —
    drawdown management is the single most important aspect of survival.
    """
    peak_equity:        float
    current_equity:     float
    daily_pnl:          List[float] = field(default_factory=list)
    consecutive_losses: int         = 0
    daily_loss_today:   float       = 0.0

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity

    @property
    def max_historical_drawdown(self) -> float:
        if not self.daily_pnl:
            return 0.0
        equity_curve = np.cumsum(self.daily_pnl) + self.peak_equity
        roll_max = np.maximum.accumulate(equity_curve)
        dd = (equity_curve - roll_max) / roll_max
        return float(dd.min())

    def record_trade(self, pnl: float) -> None:
        self.current_equity += pnl
        self.daily_loss_today += min(pnl, 0)
        self.daily_pnl.append(pnl)
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

    def daily_reset(self) -> None:
        self.daily_loss_today = 0.0


# ============================================================================
# SECTION 3 — POSITION SIZING
# ============================================================================

def kelly_position_size(
    win_rate: float = WIN_RATE_CONSERVATIVE,
    avg_win:  float = PAYOFF_RATIO_CONSERVATIVE,
    avg_loss: float = 1.0,
    fraction: float = KELLY_FRACTION_CAP,
) -> float:
    """
    Kelly fraction capped at KELLY_FRACTION_CAP.

    Full Kelly = (p*b - q) / b  where b = avg_win/avg_loss.
    Use quarter-Kelly in live trading — full Kelly maximises long-run
    growth but is extremely volatile (Hull / IMCA Handbook).
    """
    b = avg_win / avg_loss
    q = 1.0 - win_rate
    full_kelly = (win_rate * b - q) / b
    if full_kelly <= 0:
        logger.warning("Negative Kelly — edge is negative. Do not trade.")
        return 0.0
    capped = min(full_kelly * fraction, MAX_RISK_PER_TRADE_PCT)
    logger.info("Kelly: full=%.3f capped=%.3f (x%.1f fraction)", full_kelly, capped, fraction)
    return capped


def fixed_fractional_size(
    account_equity: float,
    max_loss_per_contract: float,
    risk_fraction: float = MAX_RISK_PER_TRADE_PCT,
) -> int:
    """
    Fixed-fractional position sizing.

    Heitkoetter Ch.4: divide dollar risk allowance by max loss per contract.
    Never round up.
    """
    dollar_risk = account_equity * risk_fraction
    if max_loss_per_contract <= 0:
        return 0
    contracts = int(math.floor(dollar_risk / max_loss_per_contract))
    logger.info(
        "Fixed-fractional: $%.0f risk / $%.0f per contract = %d contracts",
        dollar_risk, max_loss_per_contract, contracts,
    )
    return max(contracts, 0)


def wti_lot_size(
    account_equity: float,
    entry_price: float,
    stop_price: float,
) -> int:
    """
    WTI futures position size in lots.

    NYMEX Ch.200: 1 lot = 1,000 barrels. Risk per lot = |entry - stop| x 1,000.
    Never exceed MAX_WTI_CONTRACTS.
    """
    if entry_price <= 0 or stop_price <= 0:
        return 0
    risk_per_bbl = abs(entry_price - stop_price)
    risk_per_lot = risk_per_bbl * WTI_CONTRACT_SIZE_BBL
    if risk_per_lot <= 0:
        return 0
    dollar_risk = account_equity * MAX_RISK_PER_TRADE_PCT
    lots = int(math.floor(dollar_risk / risk_per_lot))
    lots = min(lots, MAX_WTI_CONTRACTS)
    logger.info(
        "WTI sizing: $%.0f risk / $%.0f per lot = %d lots (cap=%d)",
        dollar_risk, risk_per_lot, lots, MAX_WTI_CONTRACTS,
    )
    return max(lots, 0)


# ============================================================================
# SECTION 4 — VAR & EXPECTED SHORTFALL
# ============================================================================

def parametric_var(
    position_value: float,
    daily_vol: float,
    confidence: float = VAR_CONFIDENCE_LEVEL,
    holding_days: int = 1,
) -> float:
    """
    Parametric (variance-covariance) VaR.

    Hull Ch.9: VaR = position x sigma_daily x sqrt(T) x z_alpha.
    Assumes normally distributed returns.
    """
    from scipy import stats as st
    z = st.norm.ppf(confidence)
    var = position_value * daily_vol * math.sqrt(holding_days) * z
    logger.info("Parametric VaR(%.0f%%, %dd): $%.0f", confidence * 100, holding_days, var)
    return max(var, 0.0)


def historical_var(
    returns: np.ndarray,
    portfolio_value: float,
    confidence: float = VAR_CONFIDENCE_LEVEL,
) -> float:
    """
    Historical simulation VaR.

    Hull Ch.9: sort past returns and read off the percentile. Captures
    fat tails and skew that parametric VaR misses. Requires >=252 data points.
    """
    if len(returns) < 30:
        logger.warning("Insufficient return history for historical VaR.")
        return 0.0
    percentile = np.percentile(returns, (1 - confidence) * 100)
    var = -percentile * portfolio_value
    logger.info("Historical VaR(%.0f%%): $%.0f", confidence * 100, var)
    return max(var, 0.0)


def expected_shortfall(
    var: float,
    multiplier: float = EXPECTED_SHORTFALL_MULTIPLIER,
) -> float:
    """
    Expected Shortfall (CVaR / ES).

    Hull Ch.9 / Basel III: average loss given loss exceeds VaR.
    For a normal distribution, ES ~ VaR x 1.25-1.35.
    The regulatory standard (FRTB) now uses ES instead of VaR.
    """
    es = var * multiplier
    logger.info("Expected Shortfall: $%.0f (VaR x %.1f)", es, multiplier)
    return es


# ============================================================================
# SECTION 5 — PORTFOLIO GREEKS LIMIT CHECK
# ============================================================================

def check_greeks_limits(greeks: PortfolioGreeks) -> List[str]:
    """
    Returns list of limit breaches (empty = all clear).

    Bittman Ch.12: Greeks limits are the guardrails of an options book.
    """
    breaches = []
    if abs(greeks.net_delta) > MAX_PORTFOLIO_DELTA:
        breaches.append(
            f"Delta breach: {greeks.net_delta:+.1f} (limit +-{MAX_PORTFOLIO_DELTA})"
        )
    if abs(greeks.net_gamma) > MAX_PORTFOLIO_GAMMA:
        breaches.append(
            f"Gamma breach: {greeks.net_gamma:+.2f} (limit +-{MAX_PORTFOLIO_GAMMA})"
        )
    if greeks.net_theta < MAX_PORTFOLIO_THETA_NEGATIVE:
        breaches.append(
            f"Theta breach: ${greeks.net_theta:+.0f}/day "
            f"(floor ${MAX_PORTFOLIO_THETA_NEGATIVE:.0f})"
        )
    if abs(greeks.net_vega) > MAX_PORTFOLIO_VEGA:
        breaches.append(
            f"Vega breach: ${greeks.net_vega:+,.0f} per 1% IV "
            f"(limit +-${MAX_PORTFOLIO_VEGA:,.0f})"
        )
    return breaches


# ============================================================================
# SECTION 6 — TRADE APPROVAL GATE
# ============================================================================

_drawdown_tracker = DrawdownTracker(
    peak_equity    = ACCOUNT_EQUITY_USD,
    current_equity = ACCOUNT_EQUITY_USD,
)


def evaluate_trade(
    signal: TradeSignal,
    portfolio_greeks: Optional[PortfolioGreeks] = None,
    current_open_positions: int = 0,
    iv_rank: float = 50.0,
) -> RiskAssessment:
    """
    Full pre-trade risk gate.

    Hull Ch.16 / IMCA: every trade must pass through an independent pre-trade
    risk check, separate from the signal generator.
    """
    reasons: List[str] = []
    status   = ApprovalStatus.APPROVED
    approved_qty = 1

    # 1. Daily loss circuit breaker
    if abs(_drawdown_tracker.daily_loss_today) >= MAX_DAILY_LOSS_USD:
        reasons.append(
            f"Daily loss limit hit: ${abs(_drawdown_tracker.daily_loss_today):,.0f} "
            f"(limit ${MAX_DAILY_LOSS_USD:,.0f}) — flat for the day."
        )
        status = ApprovalStatus.REJECTED

    # 2. Consecutive loss cool-off
    if _drawdown_tracker.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
        reasons.append(
            f"Cool-off: {_drawdown_tracker.consecutive_losses} consecutive losses."
        )
        status = ApprovalStatus.REJECTED

    # 3. Max open positions
    if (current_open_positions >= MAX_OPEN_OPTIONS_POSITIONS
            and signal.strategy not in (StrategyType.FUTURES_LONG, StrategyType.FUTURES_SHORT)):
        reasons.append(
            f"Position count limit: {current_open_positions}/{MAX_OPEN_OPTIONS_POSITIONS} open."
        )
        status = ApprovalStatus.REJECTED

    # 4. DTE check
    if signal.dte < MIN_DTE_ENTRY:
        reasons.append(
            f"DTE too short: {signal.dte}d (min {MIN_DTE_ENTRY}d). Gamma risk too high."
        )
        status = ApprovalStatus.REJECTED

    if signal.dte > MAX_DTE_ENTRY:
        reasons.append(f"DTE too long: {signal.dte}d (max {MAX_DTE_ENTRY}d).")
        status = ApprovalStatus.FLAGGED if status == ApprovalStatus.APPROVED else status

    # 5. IV-rank filter
    credit_strategies = {
        StrategyType.IRON_CONDOR,
        StrategyType.COLLAR,
    }
    debit_strategies = {
        StrategyType.LONG_CALL, StrategyType.LONG_PUT,
        StrategyType.CALENDAR_SPREAD,
        StrategyType.BULL_CALL_SPREAD, StrategyType.BEAR_PUT_SPREAD,
    }

    if signal.strategy in credit_strategies and iv_rank < MIN_IV_RANK_FOR_CREDIT:
        reasons.append(
            f"IV rank too low for credit strategy: {iv_rank:.0f}% < {MIN_IV_RANK_FOR_CREDIT}%."
        )
        status = ApprovalStatus.FLAGGED if status == ApprovalStatus.APPROVED else status

    if signal.strategy in debit_strategies and iv_rank > MAX_IV_RANK_FOR_DEBIT:
        reasons.append(
            f"IV rank too high for debit strategy: {iv_rank:.0f}% > {MAX_IV_RANK_FOR_DEBIT}%."
        )
        status = ApprovalStatus.FLAGGED if status == ApprovalStatus.APPROVED else status

    # 6. Max loss sizing
    max_loss_per_contract = abs(signal.max_loss)
    if max_loss_per_contract <= 0:
        max_loss_per_contract = MAX_RISK_PER_TRADE_USD

    approved_qty = fixed_fractional_size(
        account_equity        = _drawdown_tracker.current_equity,
        max_loss_per_contract = max_loss_per_contract,
        risk_fraction         = MAX_RISK_PER_TRADE_PCT,
    )

    if approved_qty == 0:
        reasons.append(
            f"Max loss per contract (${max_loss_per_contract:,.0f}) "
            f"exceeds 2% account risk (${MAX_RISK_PER_TRADE_USD:,.0f})."
        )
        status = ApprovalStatus.REJECTED

    # 7. WTI lot cap
    if signal.strategy in (StrategyType.FUTURES_LONG, StrategyType.FUTURES_SHORT):
        approved_qty = min(approved_qty, MAX_WTI_CONTRACTS)

    # 8. Portfolio Greeks check
    if portfolio_greeks is not None:
        breaches = check_greeks_limits(portfolio_greeks)
        if breaches:
            reasons.extend(breaches)
            status = ApprovalStatus.REJECTED

    # 9. VaR calculation
    position_value = _drawdown_tracker.current_equity * MAX_RISK_PER_TRADE_PCT * approved_qty
    daily_vol = (
        0.025 if signal.strategy in (StrategyType.FUTURES_LONG, StrategyType.FUTURES_SHORT)
        else 0.015
    )
    var_1d = parametric_var(position_value, daily_vol)
    es     = expected_shortfall(var_1d)

    if var_1d > MAX_VAR_USD:
        reasons.append(f"VaR breach: ${var_1d:,.0f} > limit ${MAX_VAR_USD:,.0f}.")
        status = ApprovalStatus.FLAGGED if status == ApprovalStatus.APPROVED else status

    # 10. Kelly cross-check
    kf = kelly_position_size(
        win_rate = WIN_RATE_CONSERVATIVE,
        avg_win  = signal.risk_reward or PAYOFF_RATIO_CONSERVATIVE,
    )

    max_loss_usd = max_loss_per_contract * approved_qty
    risk_pct     = max_loss_usd / _drawdown_tracker.current_equity

    assessment = RiskAssessment(
        trade              = signal,
        status             = status,
        reasons            = reasons,
        approved_qty       = approved_qty if status != ApprovalStatus.REJECTED else 0,
        position_size_usd  = position_value,
        max_loss_usd       = max_loss_usd,
        risk_pct_account   = risk_pct,
        kelly_fraction     = kf,
        var_1day_99        = var_1d,
        expected_shortfall = es,
    )

    logger.info("Risk assessment: %s — %s", status.value, "; ".join(reasons) or "all clear")
    return assessment


def record_trade_result(pnl_usd: float) -> None:
    """Update the drawdown tracker after a trade closes."""
    _drawdown_tracker.record_trade(pnl_usd)
    logger.info(
        "PnL recorded: $%.0f | Equity: $%.0f | Drawdown: %.2f%% | Consec losses: %d",
        pnl_usd,
        _drawdown_tracker.current_equity,
        _drawdown_tracker.drawdown_pct * 100,
        _drawdown_tracker.consecutive_losses,
    )


def get_risk_summary() -> Dict:
    """Return current risk state as a plain dict for logging / display."""
    return {
        "account_equity"           : round(_drawdown_tracker.current_equity, 2),
        "peak_equity"              : round(_drawdown_tracker.peak_equity, 2),
        "drawdown_pct"             : round(_drawdown_tracker.drawdown_pct * 100, 2),
        "max_drawdown_pct"         : round(_drawdown_tracker.max_historical_drawdown * 100, 2),
        "daily_loss_today_usd"     : round(_drawdown_tracker.daily_loss_today, 2),
        "consecutive_losses"       : _drawdown_tracker.consecutive_losses,
        "max_risk_per_trade_usd"   : MAX_RISK_PER_TRADE_USD,
        "max_portfolio_risk_usd"   : MAX_PORTFOLIO_RISK_USD,
        "max_daily_loss_usd"       : MAX_DAILY_LOSS_USD,
        "max_var_1d_99_usd"        : MAX_VAR_USD,
    }


# ============================================================================
# SECTION 7 — BACKWARD-COMPATIBLE ALIASES
# (main.py was written against risk_agent.py; these keep it runnable)
# ============================================================================

FUND_EQUITY_USD    = ACCOUNT_EQUITY_USD   # name used throughout main.py & CLAUDE.md
DrawdownState      = DrawdownTracker       # old class name
record_pnl         = record_trade_result  # old function name


# ============================================================================
# SECTION 8 — STRESS TESTING
# 8 named energy-market scenarios (Hull Ch.17 / Basel III / PPM constraint)
# ============================================================================

@dataclass
class StressScenario:
    """Single named stress scenario result.

    Hull Ch.17: stress tests must cover historically observed extreme moves
    AND plausible future shocks not seen in recent data.
    """
    name:          str
    price_shock:   float   # $/bbl change applied to spot
    shock_pct:     float   # percentage price change
    pnl_usd:       float   # estimated P&L on net_long_bbls position
    nav_impact_pct:float   # P&L as % of fund equity
    exceeds_limit: bool    # True if |nav_impact_pct| > 15% (PPM gate)
    description:   str     = ""


# 8 canonical energy stress scenarios sourced from historical precedent
_STRESS_SCENARIOS: List[Tuple[str, float, str]] = [
    # (name, price_shock_pct, description)
    ("WTI 2014 Supply Glut",       -0.60, "OPEC no-cut; US shale surge — WTI fell from $107 to $43"),
    ("COVID Demand Collapse 2020",  -0.70, "Global lockdown; WTI Apr-20 contract settled at -$37/bbl"),
    ("Gulf War Supply Shock 1990",  +0.70, "Iraq invades Kuwait; WTI doubled in 3 months"),
    ("Russia-Ukraine Spike 2022",   +0.45, "EU sanctions; Brent touched $139/bbl (Mar 2022)"),
    ("OPEC+ Surprise Cut",          +0.25, "Coordinated 1 mbpd+ unannounced cut"),
    ("Hurricane Katrina Disruption",+0.20, "US Gulf Coast offline; 25% of US production lost"),
    ("USD Strengthening Shock",     -0.15, "DXY +10%; commodities historically -15% correlated"),
    ("Demand Recession Slowdown",   -0.30, "Global GDP -2%; IEA demand forecast cut 2 mbpd"),
]


def run_stress_tests(
    spot:           float,
    net_long_bbls:  float,
    sigma_annual:   float,
    fund_equity:    float,
    max_nav_loss:   float = 0.15,
) -> List[StressScenario]:
    """
    Apply all 8 named stress scenarios to the current position.

    Hull Ch.17 / PPM constraint: maximum tolerated NAV drawdown = 15%.
    The stress P&L is computed as a simple linear shock (no Greeks).
    For options positions, a full vega/gamma adjustment is needed at
    the strategy layer — this function provides the futures-equivalent floor.

    Args:
        spot:          Current WTI spot price ($/bbl).
        net_long_bbls: Net barrel exposure (positive = long, negative = short).
        sigma_annual:  Annualised GARCH volatility (for vol-scaled scenarios).
        fund_equity:   Total fund equity in USD.
        max_nav_loss:  NAV loss threshold that triggers exceeds_limit flag (default 15%).
    """
    results: List[StressScenario] = []
    for name, shock_pct, description in _STRESS_SCENARIOS:
        price_shock  = spot * shock_pct
        pnl_usd      = net_long_bbls * price_shock
        nav_impact   = pnl_usd / fund_equity if fund_equity > 0 else 0.0
        results.append(StressScenario(
            name           = name,
            price_shock    = price_shock,
            shock_pct      = shock_pct,
            pnl_usd        = pnl_usd,
            nav_impact_pct = nav_impact,
            exceeds_limit  = abs(nav_impact) > max_nav_loss,
            description    = description,
        ))
    return results


def stress_test_report(results: List[StressScenario]) -> str:
    """Format stress test results as a printable report string."""
    sep   = "─" * 72
    lines = [
        "═" * 72,
        "  STRESS TEST REPORT  (8 Named Energy Scenarios)",
        "  Hull Ch.17 · Basel III · PPM 15% NAV Limit",
        sep,
        f"  {'Scenario':<35} {'Shock':>7} {'P&L':>12} {'NAV %':>7}  {'Status':>8}",
        sep,
    ]
    for r in results:
        flag   = "⚠ BREACH" if r.exceeds_limit else "  ok"
        lines.append(
            f"  {r.name:<35} {r.shock_pct:>+6.0%} "
            f"${r.pnl_usd:>+11,.0f} {r.nav_impact_pct:>+6.1%}  {flag}"
        )
    breach_count = sum(r.exceeds_limit for r in results)
    lines += [
        sep,
        f"  Scenarios breaching 15% NAV limit: {breach_count} / {len(results)}",
        "═" * 72,
    ]
    return "\n".join(lines)


def performance_report() -> str:
    """Return a formatted risk-state and performance string for display."""
    s   = get_risk_summary()
    sep = "─" * 60
    lines = [
        "═" * 60,
        "  RISK & PERFORMANCE REPORT",
        sep,
        f"  Account Equity      : ${s['account_equity']:>12,.2f}",
        f"  Peak Equity         : ${s['peak_equity']:>12,.2f}",
        f"  Current Drawdown    : {s['drawdown_pct']:>11.2f}%",
        f"  Max Historical DD   : {s['max_drawdown_pct']:>11.2f}%",
        f"  Daily Loss Today    : ${s['daily_loss_today_usd']:>12,.2f}",
        f"  Consecutive Losses  : {s['consecutive_losses']:>12d}",
        sep,
        f"  Max Risk / Trade    : ${s['max_risk_per_trade_usd']:>12,.2f}",
        f"  Max Portfolio Risk  : ${s['max_portfolio_risk_usd']:>12,.2f}",
        f"  Max Daily Loss Gate : ${s['max_daily_loss_usd']:>12,.2f}",
        f"  Max 1-Day 99% VaR   : ${s['max_var_1d_99_usd']:>12,.2f}",
        "═" * 60,
    ]
    return "\n".join(lines)
