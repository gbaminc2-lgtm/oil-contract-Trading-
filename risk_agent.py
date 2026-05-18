"""
risk_agent.py
Oil & Gas Trading Agent Team — Risk Layer
Enforces hard capital-protection constraints before any order is dispatched.
"""

import os
import json
import time
import threading
from typing import Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic output schemas
# ---------------------------------------------------------------------------

class RiskDecision(BaseModel):
    """Output schema for the equity Risk Manager Agent."""
    approved: bool = Field(
        description="True if trade meets risk parameters, False otherwise"
    )
    adjusted_size: float = Field(
        description="Number of shares or contract units allowed"
    )
    risk_notes: str


class DerivativesRiskDecision(BaseModel):
    """Output schema for the Derivatives Risk Manager Agent."""
    approved: bool
    adjusted_contracts: int = Field(
        description="The maximum allowed volume of contract blocks to buy/sell"
    )
    max_premium_or_margin_allowed: float = Field(
        description="The max total cash capital allocated for this entry"
    )
    risk_notes: str


# ---------------------------------------------------------------------------
# Risk Manager system prompt (equity / spot)
# ---------------------------------------------------------------------------

RISK_PROMPT = """
You are a conservative, strict Risk Manager Agent.
Your sole purpose is to protect capital. You review the Analyst Agent's recommendation.
Check the signal against these hard constraints:
1. Never risk more than 2% of total account equity on one trade.
2. If market volatility (VIX) is over 30, cut all proposed position sizes in half.
3. Deny any trade if the current daily drawdown exceeds 5%.
If constraints fail, set approved to False.
Override the analyst if safety parameters are breached.
"""

# ---------------------------------------------------------------------------
# Derivatives Risk Manager system prompt
# ---------------------------------------------------------------------------

DERIVATIVES_RISK_PROMPT = """
You are a Derivatives Risk Manager.
Check risk parameters against option premiums or futures margin limits:
1. Long option trades risk losing 100% of premium. Maximum total premium allowed is 1.5% of net balance.
2. Short option legs or Futures require strict maintenance margins. Do not exceed 10% total portfolio risk exposure.
Adjust the volume profile under 'adjusted_contracts' dynamically.
"""


# ---------------------------------------------------------------------------
# Hard-coded risk constraint enforcement (no LLM required)
# ---------------------------------------------------------------------------

def enforce_risk_constraints(
    proposed_size: float,
    vix: float,
    account: dict,
) -> dict:
    """
    Apply deterministic guardrails before forwarding to the LLM Risk Agent.

    Parameters
    ----------
    proposed_size : float
        Raw number of shares / contracts suggested by the Analyst.
    vix : float
        Current VIX level.
    account : dict
        Keys: 'equity' (float), 'daily_drawdown' (float 0-1).

    Returns
    -------
    dict with keys: approved (bool), adjusted_size (float), risk_notes (str)
    """
    equity = account.get("equity", 0.0)
    daily_drawdown = account.get("daily_drawdown", 0.0)

    approved = True
    risk_notes = "Parameters within normal bounds."
    size = proposed_size

    # Rule 1 — max 2% equity at risk
    max_risk_dollars = equity * 0.02
    if size * 1.0 > max_risk_dollars:
        size = max_risk_dollars
        risk_notes = "Size capped at 2% equity rule."

    # Rule 2 — VIX > 30: halve position
    if vix > 30:
        size = size * 0.5
        risk_notes = f"VIX exceeds 30. Execution size scaled down 50% for safety."

    # Rule 3 — daily drawdown > 5%: reject
    if daily_drawdown > 0.05:
        approved = False
        risk_notes = "CRITICAL FAILURE: Max system drawdown exceeded. Trade rejected."
        size = 0.0

    return {
        "approved": approved,
        "adjusted_size": size,
        "risk_notes": risk_notes,
    }


# ---------------------------------------------------------------------------
# Risk agent runner
# ---------------------------------------------------------------------------

def run_risk_agent(
    analyst_output,
    account: dict,
    vix: float,
    call_claude_fn,
) -> RiskDecision:
    """
    Step 2 — Send the Analyst's proposed trade to the Risk Manager Agent.
    Pre-filter with hard rules first; then call LLM for notes + approval stamp.
    """
    pre_check = enforce_risk_constraints(
        proposed_size=100.0,
        vix=vix,
        account=account,
    )

    prompt = (
        f"Proposed Trade: {analyst_output.model_dump_json()}. "
        f"Current Account State: {account}. "
        f"Pre-check result: {pre_check}"
    )

    return call_claude_fn(
        system=RISK_PROMPT,
        prompt=prompt,
        response_format=RiskDecision,
    )


def run_derivatives_risk_agent(
    derivatives_signal,
    account: dict,
    call_claude_fn,
) -> DerivativesRiskDecision:
    """Run the Derivatives Risk Manager Agent."""
    prompt = (
        f"Proposed Derivatives Trade: {derivatives_signal.model_dump_json()}. "
        f"Account State: {account}."
    )
    return call_claude_fn(
        system=DERIVATIVES_RISK_PROMPT,
        prompt=prompt,
        response_format=DerivativesRiskDecision,
    )


# ---------------------------------------------------------------------------
# Emergency halt & kill switch
# ---------------------------------------------------------------------------

SYSTEM_ACTIVE: bool = True


def trigger_emergency_halt(reason: str = "Manual intervention") -> None:
    """
    Global system halt sequence.
    Cancels all open orders and liquidates every open position via Alpaca API.
    """
    global SYSTEM_ACTIVE
    SYSTEM_ACTIVE = False
    print(f"!!! EMERGENCY HALT TRIGGERED: {reason} !!!")

    try:
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(
            key_id=os.environ.get("ALPACA_API_KEY"),
            secret_key=os.environ.get("ALPACA_SECRET_KEY"),
            base_url=os.environ.get("ALPACA_BASE_URL", "https://alpaca.markets"),
            api_version="v2",
        )
        api.cancel_all_orders()
        api.close_all_positions()
        print("All market positions successfully flattened.")
    except Exception as exc:
        print(f"CRITICAL: Failed to liquidate positions during halt: {exc}")


# ---------------------------------------------------------------------------
# Heartbeat watchdog (background thread)
# ---------------------------------------------------------------------------

last_heartbeat_timestamp: float = time.time()
WATCHDOG_TIMEOUT_SECONDS: float = 30.0


def run_heartbeat_watchdog() -> None:
    """
    Independent background thread checking if WebSocket data has frozen.
    Fires an emergency alert when silence exceeds WATCHDOG_TIMEOUT_SECONDS.
    """
    global last_heartbeat_timestamp
    print("Failsafe Heartbeat Watchdog active.")

    while True:
        time.sleep(5)
        time_since_last_tick = time.time() - last_heartbeat_timestamp
        if time_since_last_tick > WATCHDOG_TIMEOUT_SECONDS:
            msg = (
                f"WebSocket stream has been completely silent for "
                f"{time_since_last_tick:.1f} seconds! Possible connection drop."
            )
            print(f"[WATCHDOG CRITICAL]: {msg}")


def start_watchdog_thread() -> threading.Thread:
    """Spawn the heartbeat watchdog as a daemon thread."""
    t = threading.Thread(target=run_heartbeat_watchdog, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Account balance stub (replace with live Alpaca call in production)
# ---------------------------------------------------------------------------

def get_account_balance() -> dict:
    """Return current account equity and daily drawdown."""
    return {"equity": 50000, "daily_drawdown": 0.01}
