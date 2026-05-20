"""
risk_validation.py
Oil Contract Trading — Risk Management & Validation Layer
Hardcoded risk parameters and sequential multi-gate signal validation.

All threshold constants are defined here.  To adjust risk tolerance,
modify only this file — no other module needs to change.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Hardcoded Risk Parameters
# ---------------------------------------------------------------------------

# Gate 1 — CAPM Beta Threshold
# Minimum rolling 20-day Beta required for systemic-risk alignment.
BETA_MIN_THRESHOLD: float = 1.1

# Gate 2 — Black-Scholes Volatility Expansion Threshold
# The candle body must EXCEED the BSM 1-day expected move.
BSM_BODY_MULTIPLIER: float = 1.0

# Rolling window used for Beta and Volatility calculations (trading days)
ROLLING_WINDOW: int = 20

# Maximum allowable portfolio Beta exposure (informational — not enforced here)
MAX_PORTFOLIO_BETA: float = 2.0

# Stop-loss expressed as a multiple of the BSM expected daily move
STOP_LOSS_BSM_MULTIPLE: float = 1.5

# Profit target expressed as a multiple of the BSM expected daily move
PROFIT_TARGET_BSM_MULTIPLE: float = 3.0


# ---------------------------------------------------------------------------
# Beta Validation Gate
# ---------------------------------------------------------------------------

def apply_beta_gate(df: pd.DataFrame,
                    beta_min: float = BETA_MIN_THRESHOLD) -> pd.DataFrame:
    """
    Gate 1: Validates that the rolling 20-day Beta exceeds the minimum
    systemic-risk threshold.

    A Beta > BETA_MIN_THRESHOLD confirms the asset is a high-velocity,
    market-leading instrument — required for oil futures momentum plays.

    Parameters
    ----------
    df       : DataFrame containing a 'Beta' column
    beta_min : Minimum Beta required to pass this gate

    Returns
    -------
    pd.DataFrame with additional boolean column 'Beta_Validated'
    """
    df = df.copy()
    df["Beta_Validated"] = df["Beta"] > beta_min
    return df


# ---------------------------------------------------------------------------
# BSM Volatility Expansion Gate
# ---------------------------------------------------------------------------

def apply_bsm_gate(df: pd.DataFrame,
                   multiplier: float = BSM_BODY_MULTIPLIER) -> pd.DataFrame:
    """
    Gate 2: Validates that the candle body size exceeds the Black-Scholes
    1-day expected move scaled by BSM_BODY_MULTIPLIER.

    A body expansion beyond the BSM threshold confirms that price action is
    statistically significant — not mere noise — justifying entry.

    Parameters
    ----------
    df         : DataFrame with 'Candle_Body_Size' and 'BSM_Expected_1D_Move'
    multiplier : Scale factor applied to the BSM expected move floor

    Returns
    -------
    pd.DataFrame with additional boolean column 'BSM_Shield_Validated'
    """
    df = df.copy()
    df["BSM_Shield_Validated"] = (
        df["Candle_Body_Size"] > df["BSM_Expected_1D_Move"] * multiplier
    )
    return df


# ---------------------------------------------------------------------------
# CAPM / Beta Computation
# ---------------------------------------------------------------------------

def compute_rolling_beta(df: pd.DataFrame,
                         rolling_window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Computes the 20-day rolling CAPM Beta for the asset relative to the
    market benchmark.

    Beta = Cov(Asset, Market) / Var(Market)

    Parameters
    ----------
    df             : DataFrame with 'Asset_Return' and 'Market_Return' columns
    rolling_window : Lookback window in trading days

    Returns
    -------
    pd.DataFrame with additional column 'Beta'
    """
    df = df.copy()
    covariance      = df["Asset_Return"].rolling(rolling_window).cov(df["Market_Return"])
    market_variance = df["Market_Return"].rolling(rolling_window).var()
    df["Beta"]      = covariance / market_variance
    return df


# ---------------------------------------------------------------------------
# Combined Risk Validation Pipeline
# ---------------------------------------------------------------------------

def run_risk_validation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes the full sequential multi-gate risk validation pipeline.

    Gate order
    ----------
    1. Compute rolling CAPM Beta
    2. Apply Beta threshold gate
    3. Apply BSM volatility expansion gate
    4. Combine both gates into the final 'High_Edge_Signal' flag

    Parameters
    ----------
    df : DataFrame enriched by strategy_agent.run_strategy()

    Returns
    -------
    pd.DataFrame with columns:
        Beta                 — rolling 20-day CAPM Beta
        Beta_Validated       — Gate 1 boolean
        BSM_Shield_Validated — Gate 2 boolean
        High_Edge_Signal     — Final combined signal flag
    """
    df = compute_rolling_beta(df)
    df = apply_beta_gate(df)
    df = apply_bsm_gate(df)

    df["High_Edge_Signal"] = (
        df["Bullish_Engulfing"]
        & df["Beta_Validated"]
        & df["BSM_Shield_Validated"]
    )

    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Stop-Loss & Profit Target Helpers
# ---------------------------------------------------------------------------

def calculate_stop_loss(entry_price: float, bsm_daily_move: float) -> float:
    """
    Returns the stop-loss price level as a function of BSM daily move.

    Stop = Entry - (BSM_Daily_Move x STOP_LOSS_BSM_MULTIPLE)
    """
    return entry_price - (bsm_daily_move * STOP_LOSS_BSM_MULTIPLE)


def calculate_profit_target(entry_price: float, bsm_daily_move: float) -> float:
    """
    Returns the profit-target price level as a function of BSM daily move.

    Target = Entry + (BSM_Daily_Move x PROFIT_TARGET_BSM_MULTIPLE)
    """
    return entry_price + (bsm_daily_move * PROFIT_TARGET_BSM_MULTIPLE)
