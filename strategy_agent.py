"""
strategy_agent.py — Quantitative Commodity Strategy Engine
===========================================================
System Role: Expert Quantitative Commodity Strategist (Institutional Grade)

Models implemented:
  ① Black-Scholes-Merton (equity options) + Black-76 (energy options / futures)
  ② Monte Carlo price simulation: GBM + mean-reverting (Ornstein-Uhlenbeck)
  ③ NAV model — Net Asset Value for commodity fund/book
  ④ Crack-spread trading strategy (3-2-1 / 5-3-2 arb)
  ⑤ Storage Arbitrage (buy spot + store + sell forward)
  ⑥ Basis Trading (physical vs. futures differential)
  ⑦ Three-Statement Financial Model (P&L, Balance Sheet, Cash Flow)
  ⑧ PPM Section generator (Private Placement Memorandum)
  ⑨ ML-based directional signal (GradientBoosting via scikit-learn)
  ⑩ Algorithmic learning regime classifier (supervised / unsupervised)

Pricing conventions:
  Energy options → Black-76: underlying = F (futures price), r discounts
  only the premium; convenience yield already embedded in F.
  Equity options → BSM with continuous dividend q.
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
from scipy import stats as sp_stats

from data_agent import (
    ForwardCurve, CrackSpread, StorageEconomics, HistoricalVolRegime,
    SeasonalPattern, MLSignalData,
    fetch_spot, fetch_ohlcv, fetch_forward_curve, fetch_crack_spreads,
    fetch_storage_economics, fetch_garch_vol, fetch_seasonal_pattern,
    fetch_iv_surface, fetch_risk_free_rate, build_ml_features,
    WTI_TICKER, BRENT_TICKER, RBOB_TICKER, ULSD_TICKER,
    WTI_BBL_PER_CONTRACT, PRODUCT_GAL_PER_BBL, TRADING_DAYS_YEAR,
)

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    _SKL = True
except ImportError:
    _SKL = False

try:
    from market_architecture import get_market_arch as _get_mam
    _MAM = True
except ImportError:
    _MAM = False


# ============================================================================
# 1. ENUMERATIONS
# ============================================================================

class OptionRight(str, Enum):
    CALL = "C"; PUT = "P"

class Direction(str, Enum):
    LONG = "LONG"; SHORT = "SHORT"; FLAT = "FLAT"

class StrategyType(str, Enum):
    # Commodity-specific
    STORAGE_ARB         = "STORAGE_ARB"
    CRACK_SPREAD_LONG   = "CRACK_SPREAD_LONG"    # buy crude, sell products
    CRACK_SPREAD_SHORT  = "CRACK_SPREAD_SHORT"   # sell crude, buy products
    BASIS_TRADE_LONG    = "BASIS_TRADE_LONG"     # long physical, short futures
    BASIS_TRADE_SHORT   = "BASIS_TRADE_SHORT"
    CURVE_STEEPENER     = "CURVE_STEEPENER"      # long front / short back
    CURVE_FLATTENER     = "CURVE_FLATTENER"
    # Options
    COLLAR              = "COLLAR"               # buy put + sell call (producer hedge)
    BULL_CALL_SPREAD    = "BULL_CALL_SPREAD"
    BEAR_PUT_SPREAD     = "BEAR_PUT_SPREAD"
    IRON_CONDOR         = "IRON_CONDOR"
    CALENDAR_SPREAD     = "CALENDAR_SPREAD"
    LONG_CALL           = "LONG_CALL"
    LONG_PUT            = "LONG_PUT"
    # Futures directional
    FUTURES_LONG        = "FUTURES_LONG"
    FUTURES_SHORT       = "FUTURES_SHORT"

class VolRegime(str, Enum):
    LOW = "LOW"; NORMAL = "NORMAL"; HIGH = "HIGH"; SPIKE = "SPIKE"

class MarketRegime(str, Enum):
    CONTANGO_BULLISH      = "CONTANGO_BULLISH"
    CONTANGO_BEARISH      = "CONTANGO_BEARISH"
    BACKWARDATION_BULLISH = "BACKWARDATION_BULLISH"
    BACKWARDATION_BEARISH = "BACKWARDATION_BEARISH"
    FLAT                  = "FLAT"


# ============================================================================
# 2. BLACK-SCHOLES-MERTON ENGINE
# ============================================================================

@dataclass(frozen=True)
class OptionInputs:
    S:     float   # spot / futures price
    K:     float   # strike
    T:     float   # years to expiry (> 0)
    r:     float   # risk-free rate (annualised continuous)
    sigma: float   # annualised vol
    q:     float = 0.0   # dividend / convenience yield (set q=r for Black-76)

    def __post_init__(self):
        assert self.T > 0, f"T must be > 0 (got {self.T})"
        assert self.sigma > 0, f"sigma must be > 0 (got {self.sigma})"
        assert self.S > 0 and self.K > 0, "S and K must be positive"


@dataclass
class Greeks:
    """
    Full first-order and selected second-order Greeks.

    Bittman Ch.4: Delta = direction hedge ratio; Gamma = convexity of delta;
    Theta = time decay (negative for longs); Vega = vol exposure;
    Rho = interest rate sensitivity (matters at earnings / FOMC).
    Vanna = ∂Delta/∂sigma; Charm = ∂Delta/∂T (delta bleed per day).
    """
    price:  float
    delta:  float; gamma:  float; theta:  float; vega:   float; rho: float
    vanna:  float = 0.0   # ∂²V/∂S∂σ
    charm:  float = 0.0   # ∂²V/∂S∂T
    d1:     float = 0.0
    d2:     float = 0.0
    right:  str   = "C"

    @property
    def intrinsic(self) -> float:
        return 0.0  # computed externally with S/K

    def summary_line(self, K: float, T: float, sigma: float) -> str:
        return (f"{self.right} K={K:.2f} T={T:.4f}y σ={sigma:.1%} "
                f"| P={self.price:.4f} Δ={self.delta:+.4f} Γ={self.gamma:.6f} "
                f"Θ={self.theta:+.4f}/d V={self.vega:.4f}/1% ρ={self.rho:+.4f}/1%")


def black_scholes(
    S: float, K: float, T: float, r: float, sigma: float,
    right: OptionRight, q: float = 0.0,
) -> Greeks:
    """
    Generalised BSM / Black-76.

    Black-76 for energy futures: set q = r so forward F = S·exp((r-q)·T) = S.
    Pass the futures price as S directly.

    Sherbin Ch.4 / Bittman Ch.3: never use BSM without verifying the
    forward price convention. Energy options trade on futures, not spot.
    Failing to use Black-76 systematically overprices calls in contango.
    """
    inp    = OptionInputs(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
    sqT    = math.sqrt(T)
    d1     = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma * sqT)
    d2     = d1 - sigma * sqT

    Nd1    = sp_stats.norm.cdf(d1);  Nd2  = sp_stats.norm.cdf(d2)
    Nnd1   = sp_stats.norm.cdf(-d1); Nnd2 = sp_stats.norm.cdf(-d2)
    npd1   = sp_stats.norm.pdf(d1)
    eq     = math.exp(-q*T);  er = math.exp(-r*T)

    if right == OptionRight.CALL:
        price = S*eq*Nd1  - K*er*Nd2
        delta = eq * Nd1
        rho   = K*T*er*Nd2 * 0.01
    else:
        price = K*er*Nnd2 - S*eq*Nnd1
        delta = -eq * Nnd1
        rho   = -K*T*er*Nnd2 * 0.01

    gamma  = eq * npd1 / (S * sigma * sqT)
    vega   = S * eq * npd1 * sqT * 0.01
    theta  = -(
        S*eq*npd1*sigma / (2*sqT)
        + (r-q)*S*eq*(Nd1 if right==OptionRight.CALL else -Nnd1)
        - r*K*er*(Nd2 if right==OptionRight.CALL else -Nnd2)
    ) / 365.0

    # Vanna = ∂Delta/∂sigma = -(d2/sigma) × gamma × S
    vanna = -d2/sigma * gamma * S if sigma > 0 else 0.0
    # Charm = ∂Delta/∂T (per calendar day)
    charm = -(eq * npd1 * (2*(r-q)*T - d2*sigma*sqT) / (2*T*sigma*sqT)) / 365.0

    return Greeks(
        price=max(price, 0.0), delta=delta, gamma=gamma,
        theta=theta, vega=vega, rho=rho, vanna=vanna, charm=charm,
        d1=d1, d2=d2, right=right.value,
    )


def black76(F: float, K: float, T: float, r: float, sigma: float,
            right: OptionRight) -> Greeks:
    """
    Black (1976) model for options on futures/forwards.
    Equivalent to BSM with q = r: forward F = S, financing cancels.
    Standard for WTI, Brent, natural gas, RBOB options.
    """
    return black_scholes(F, K, T, r, sigma, right, q=r)


def implied_vol_bisection(
    market_px: float, S: float, K: float, T: float, r: float,
    right: OptionRight, q: float = 0.0,
    tol: float = 1e-6, max_iter: int = 200,
) -> Optional[float]:
    """
    Implied vol via bisection.
    Sherbin: IV is the market's opinion on forward realised vol. It is not a
    forecast — compare to HV/GARCH to assess premium/discount.
    """
    lo, hi = 1e-4, 10.0
    for _ in range(max_iter):
        mid  = (lo + hi) / 2.0
        diff = black_scholes(S, K, T, r, mid, right, q).price - market_px
        if abs(diff) < tol:
            return mid
        lo, hi = (mid, hi) if diff < 0 else (lo, mid)
    return (lo + hi) / 2.0


# ============================================================================
# 3. MONTE CARLO PRICE SIMULATION
# ============================================================================

@dataclass
class MCSimResult:
    """
    Monte Carlo simulation results for commodity price paths.

    Gkinis Ch.5–6: two models for crude oil:
    ① GBM (Geometric Brownian Motion) — appropriate when no strong
       mean reversion; used for short-term options pricing.
    ② Ornstein-Uhlenbeck (mean-reverting) — long-run crude oil prices
       do mean-revert to marginal cost of production (~$45-65/bbl LT).
    """
    model:            str   # "GBM" or "OU"
    ticker:           str
    S0:               float
    paths:            np.ndarray   # shape: (n_paths, n_steps)
    terminal_prices:  np.ndarray
    mean_terminal:    float
    median_terminal:  float
    p5:               float   # 5th percentile
    p95:              float   # 95th percentile
    var_95_pct:       float   # % loss at 5th pct vs spot
    prob_above_spot:  float


def monte_carlo_gbm(
    S0: float, sigma: float, r: float, T: float,
    n_paths: int = 10_000, n_steps: int = 252,
    seed: int = 42,
) -> MCSimResult:
    """
    GBM Monte Carlo: dS = μ·S·dt + σ·S·dW
    Risk-neutral drift: μ = r − q (set q=0 for spot, q=r for futures/Black-76).

    Used for: short-dated options pricing, VaR estimation, delta-hedge P&L sims.
    """
    rng   = np.random.default_rng(seed)
    dt    = T / n_steps
    mu    = r - 0.5 * sigma**2  # log drift
    dW    = rng.standard_normal((n_paths, n_steps))
    log_S = np.log(S0) + np.cumsum(mu*dt + sigma*math.sqrt(dt)*dW, axis=1)
    paths = np.exp(log_S)
    term  = paths[:, -1]
    return MCSimResult(
        model="GBM", ticker=WTI_TICKER, S0=S0,
        paths=paths, terminal_prices=term,
        mean_terminal=float(np.mean(term)),
        median_terminal=float(np.median(term)),
        p5=float(np.percentile(term, 5)),
        p95=float(np.percentile(term, 95)),
        var_95_pct=float((S0 - np.percentile(term, 5)) / S0),
        prob_above_spot=float(np.mean(term > S0)),
    )


def monte_carlo_ou(
    S0: float, mu_lr: float, kappa: float, sigma: float,
    T: float, n_paths: int = 10_000, n_steps: int = 252,
    seed: int = 42,
) -> MCSimResult:
    """
    Ornstein-Uhlenbeck mean-reverting model:
    dS = κ·(μ_LR − S)·dt + σ·dW

    Parameters:
      μ_LR  = long-run price (e.g. $60/bbl for WTI — marginal production cost)
      κ     = mean-reversion speed (higher → faster snap back)
      σ     = vol of vol (diffusion term)

    Gkinis Ch.3: OU better captures commodity super-cycles. Use κ≈0.5 for WTI
    (half-life ≈ 1.4 years), μ_LR ≈ $60-70/bbl for WTI.
    """
    rng   = np.random.default_rng(seed)
    dt    = T / n_steps
    paths = np.zeros((n_paths, n_steps))
    S     = np.full(n_paths, float(S0))

    for t in range(n_steps):
        dW     = rng.standard_normal(n_paths)
        S      = S + kappa * (mu_lr - S) * dt + sigma * math.sqrt(dt) * dW
        S      = np.maximum(S, 1.0)  # oil can't go (sustainably) negative
        paths[:, t] = S

    term = paths[:, -1]
    return MCSimResult(
        model="OU", ticker=WTI_TICKER, S0=S0,
        paths=paths, terminal_prices=term,
        mean_terminal=float(np.mean(term)),
        median_terminal=float(np.median(term)),
        p5=float(np.percentile(term, 5)),
        p95=float(np.percentile(term, 95)),
        var_95_pct=float((S0 - np.percentile(term, 5)) / S0),
        prob_above_spot=float(np.mean(term > S0)),
    )


def monte_carlo_option_price(
    F: float, K: float, T: float, r: float, sigma: float,
    right: OptionRight, n_paths: int = 50_000,
) -> Tuple[float, float]:
    """
    MC option price using Black-76 (futures underlying).
    Returns (price, standard_error).
    Compare vs. analytical Black-76 for model validation.
    """
    mc  = monte_carlo_gbm(F, sigma, 0.0, T, n_paths)
    if right == OptionRight.CALL:
        payoffs = np.maximum(mc.terminal_prices - K, 0.0)
    else:
        payoffs = np.maximum(K - mc.terminal_prices, 0.0)
    pv  = math.exp(-r * T) * np.mean(payoffs)
    se  = math.exp(-r * T) * np.std(payoffs) / math.sqrt(n_paths)
    return float(pv), float(se)


# ============================================================================
# 4. NAV MODEL — COMMODITY FUND NET ASSET VALUE
# ============================================================================

@dataclass
class NAVModel:
    """
    Net Asset Value model for a commodity trading fund.

    Components:
    ① Long commodity exposure (futures MTM)
    ② Options book MTM (Black-76)
    ③ Roll yield (positive in backwardation, negative in contango)
    ④ Cash collateral return (T-bill on posted margin)
    ⑤ Management fee drag

    Attribution (investment-grade PPM standard):
    NAV change = spot_return + roll_yield + collateral_return − fees − transaction_costs
    """
    fund_name:             str
    inception_date:        datetime.date
    nav_per_unit:          float
    total_units:           float
    commodity:             str

    # Exposure
    futures_contracts:     int
    avg_entry_price:       float
    current_futures_price: float
    contract_size_bbl:     int = WTI_BBL_PER_CONTRACT

    # Roll
    roll_yield_ann:        float = 0.0
    collateral_yield_ann:  float = TRADING_DAYS_YEAR * 0.0002   # proxied at r

    # Fees
    mgmt_fee_ann:          float = 0.02   # 2% management fee
    perf_fee:              float = 0.20   # 20% performance fee
    hurdle_rate:           float = 0.05   # 5% hurdle

    @property
    def total_nav_usd(self) -> float:
        return self.nav_per_unit * self.total_units

    @property
    def futures_pnl_usd(self) -> float:
        return (self.current_futures_price - self.avg_entry_price) * \
               self.futures_contracts * self.contract_size_bbl

    @property
    def gross_return_pct(self) -> float:
        notional = self.avg_entry_price * self.futures_contracts * self.contract_size_bbl
        return self.futures_pnl_usd / notional if notional > 0 else 0.0

    @property
    def net_return_pct(self) -> float:
        return self.gross_return_pct + self.roll_yield_ann - self.mgmt_fee_ann

    def attribution(self) -> Dict[str, float]:
        return {
            "spot_return_pct"      : self.gross_return_pct * 100,
            "roll_yield_pct"       : self.roll_yield_ann * 100,
            "collateral_return_pct": self.collateral_yield_ann * 100,
            "mgmt_fee_pct"         : -self.mgmt_fee_ann * 100,
            "net_return_pct"       : self.net_return_pct * 100,
            "futures_pnl_usd"      : self.futures_pnl_usd,
            "total_nav_usd"        : self.total_nav_usd,
        }


def build_nav_model(
    fund_name: str = "Quant Energy Alpha Fund",
    contracts: int = 1,
    entry_price: Optional[float] = None,
) -> NAVModel:
    """Instantiate NAV model from live data. Scaled to $500 starting capital."""
    curve  = fetch_forward_curve("WTI")
    entry  = entry_price or curve.spot * 0.97   # 3% below current (demo)
    return NAVModel(
        fund_name             = fund_name,
        inception_date        = datetime.date.today() - datetime.timedelta(days=90),
        nav_per_unit          = 1.00,
        total_units           = 500,
        commodity             = "WTI",
        futures_contracts     = contracts,
        avg_entry_price       = entry,
        current_futures_price = curve.spot,
        roll_yield_ann        = curve.annualised_roll_yield,
        collateral_yield_ann  = fetch_risk_free_rate(),
    )


# ============================================================================
# 5. TRADE SIGNAL CONTAINERS
# ============================================================================

@dataclass
class TradeSignal:
    """
    Institutional-grade trade signal with full metadata.

    Every signal includes: strategy type, direction, entry, max-loss,
    target, DTE / horizon, risk-reward, confidence, and full rationale.
    Bittman: "A trade without a defined max-loss is a gamble."
    """
    ticker:          str
    strategy:        StrategyType
    direction:       Direction
    entry_price:     float
    target_price:    float
    stop_price:      float
    legs:            List[Dict]
    net_premium:     float      # positive = credit, negative = debit
    max_profit:      float
    max_loss:        float
    dte:             int
    confidence:      float      # 0–1
    vol_regime:      VolRegime
    market_regime:   MarketRegime
    rationale:       str
    timestamp:       datetime.datetime = field(default_factory=datetime.datetime.utcnow)

    @property
    def risk_reward(self) -> Optional[float]:
        if self.max_loss == 0:
            return None
        return abs(self.max_profit / self.max_loss)

    @property
    def expected_value(self) -> float:
        """Simple EV = prob_win × max_profit + (1−prob_win) × (−max_loss)."""
        pw = self.confidence
        return pw * self.max_profit - (1 - pw) * self.max_loss


# ============================================================================
# 6. COMMODITY STRATEGY GENERATORS
# ============================================================================

def generate_storage_arb(econ: Optional[StorageEconomics] = None) -> Optional[TradeSignal]:
    """
    Storage arbitrage: buy spot WTI + store at Cushing + sell M2 forward.

    No-arb condition violated when: F(T) > S·exp((r+u)·T) − y·S·T
    i.e., the forward is above the full-carry price.

    Risk: convenience yield spikes (unexpected draw on physical supply)
    can collapse contango before you lock the arb.

    Oil Trader Academy / Hedging Strategies analysis: basis risk (Cushing
    vs. delivery-grade quality differentials) must be sized conservatively.
    """
    econ = econ or fetch_storage_economics()
    if not econ.arb_available:
        logger.info("Storage arb: no opportunity (profit=%.2f/bbl)", econ.storage_arb_pnl)
        return None

    profit_per_bbl  = econ.storage_arb_pnl
    n_contracts     = 3
    total_profit    = profit_per_bbl * n_contracts * WTI_BBL_PER_CONTRACT

    return TradeSignal(
        ticker        = WTI_TICKER,
        strategy      = StrategyType.STORAGE_ARB,
        direction     = Direction.LONG,
        entry_price   = econ.spot_wti,
        target_price  = econ.m2_price,
        stop_price    = econ.spot_wti - 1.0,   # $1/bbl stop
        legs          = [
            dict(action="BUY",  instrument="SPOT",    price=econ.spot_wti, qty=n_contracts),
            dict(action="SELL", instrument="M2_FWD",  price=econ.m2_price, qty=n_contracts),
        ],
        net_premium   = total_profit,
        max_profit    = total_profit,
        max_loss      = econ.monthly_carry * 2 * n_contracts * WTI_BBL_PER_CONTRACT,
        dte           = 60,
        confidence    = 0.75,
        vol_regime    = VolRegime.NORMAL,
        market_regime = MarketRegime.CONTANGO_BEARISH,
        rationale     = (
            f"Storage arb available: spot={econ.spot_wti:.2f}, "
            f"M2={econ.m2_price:.2f}, carry={econ.monthly_carry:.2f}/mo/bbl, "
            f"profit={profit_per_bbl:.2f}/bbl locked."
        ),
    )


def generate_crack_spread_signal(crack: Optional[CrackSpread] = None) -> TradeSignal:
    """
    Crack spread strategy based on refinery margin levels.

    Oil Trader Academy: trading the crack spread = trading refinery economics.
    - 3-2-1 > $15/bbl AND rising → crack is wide → sell crack (short gasoline,
      long crude) expecting margin compression.
    - 3-2-1 < $5/bbl AND falling → crack is compressed → buy crack (long
      gasoline, short crude) expecting margin recovery.

    Seasonal context: cracks typically peak in May-June (gasoline) and
    October (heating oil). Buy cracks in Feb-March, sell in May-June.
    """
    crack = crack or fetch_crack_spreads()
    season = fetch_seasonal_pattern(WTI_TICKER)
    current_month = datetime.date.today().month

    # MAM Phase-2 cross-check: recompute 3-2-1 from raw component prices
    crack_321_ref = crack.crack_321
    if _MAM:
        try:
            mam = _get_mam()
            crack_321_ref = mam.calculate_crack_spread(
                crack.crude_price, crack.gasoline_price, crack.heating_oil_price
            )
            logger.debug("MAM 3-2-1 crack cross-check: $%.2f vs data_agent $%.2f",
                         crack_321_ref, crack.crack_321)
        except Exception:
            crack_321_ref = crack.crack_321

    # Wide crack → sell (short products, long crude)
    if crack_321_ref > 15.0:
        direction = Direction.SHORT  # short the crack
        strat     = StrategyType.CRACK_SPREAD_SHORT
        conf      = min(0.65 + (crack_321_ref - 15) * 0.01, 0.85)
        rationale = (
            f"3-2-1 crack wide at ${crack_321_ref:.2f}/bbl (>$15). "
            f"Sell crack: short {WTI_BBL_PER_CONTRACT} bbl RBOB/ULSD, long {WTI_BBL_PER_CONTRACT} bbl WTI. "
            f"Seasonal factor (month {current_month}): {season.get(current_month):.3f}"
        )
        target = crack_321_ref - 5.0
        stop   = crack_321_ref + 3.0
    else:
        direction = Direction.LONG  # long the crack
        strat     = StrategyType.CRACK_SPREAD_LONG
        conf      = min(0.55 + (10.0 - crack_321_ref) * 0.01, 0.75)
        rationale = (
            f"3-2-1 crack compressed at ${crack_321_ref:.2f}/bbl. "
            f"Buy crack: long RBOB/ULSD, short WTI. "
            f"Seasonal factor (month {current_month}): {season.get(current_month):.3f}"
        )
        target = crack_321_ref + 4.0
        stop   = crack_321_ref - 2.0

    return TradeSignal(
        ticker       = f"{WTI_TICKER}/{RBOB_TICKER}/{ULSD_TICKER}",
        strategy     = strat,
        direction    = direction,
        entry_price  = crack_321_ref,
        target_price = target,
        stop_price   = stop,
        legs         = [
            dict(action="SELL" if direction==Direction.SHORT else "BUY",
                 instrument=WTI_TICKER,   qty=3),
            dict(action="BUY"  if direction==Direction.SHORT else "SELL",
                 instrument=RBOB_TICKER,  qty=2),
            dict(action="BUY"  if direction==Direction.SHORT else "SELL",
                 instrument=ULSD_TICKER,  qty=1),
        ],
        net_premium  = 0.0,
        max_profit   = abs(target - crack.crack_321) * WTI_BBL_PER_CONTRACT * 3,
        max_loss     = abs(stop  - crack.crack_321) * WTI_BBL_PER_CONTRACT * 3,
        dte          = 30,
        confidence   = conf,
        vol_regime   = VolRegime.NORMAL,
        market_regime= MarketRegime.FLAT,
        rationale    = rationale,
    )


def generate_basis_trade(
    physical_basis_usd: float = -1.50,
    mean_basis_usd: float = -0.50,
) -> TradeSignal:
    """
    Basis trading: profit from physical−futures differential convergence.

    Oil Contracts (Petroleum Contract book) / Analysis of Hedging Strategies:
    Basis = physical price − futures price. Basis risk is the residual
    uncertainty in a hedged position due to imperfect correlation between
    the physical commodity and its futures hedge.

    Wide basis → buy physical, short futures (basis convergence to mean).
    Narrow basis → sell physical, buy futures (basis expansion).

    Example: WTI Midland −$3/bbl basis in 2018 (Permian egress constraint).
    Correct trade: buy Midland physical, short Cushing futures → profit
    as Permian pipeline relief normalised the basis.
    """
    basis_deviation = physical_basis_usd - mean_basis_usd
    if basis_deviation < -0.50:   # basis too wide (physical too cheap)
        direction = Direction.LONG  # buy cheap physical, short futures
        strat     = StrategyType.BASIS_TRADE_LONG
        target    = mean_basis_usd
        conf      = min(0.60 + abs(basis_deviation) * 0.05, 0.80)
        rationale = (
            f"Basis at ${physical_basis_usd:.2f}/bbl vs. mean ${mean_basis_usd:.2f}/bbl. "
            f"Buy physical (cheap) + short futures = locked basis-convergence trade."
        )
    else:
        direction = Direction.SHORT
        strat     = StrategyType.BASIS_TRADE_SHORT
        target    = mean_basis_usd
        conf      = 0.55
        rationale = (
            f"Basis at ${physical_basis_usd:.2f}/bbl (elevated). "
            f"Sell physical + buy futures = basis normalisation trade."
        )

    profit_per_bbl = abs(physical_basis_usd - mean_basis_usd)
    return TradeSignal(
        ticker       = "WTI_PHYSICAL",
        strategy     = strat,
        direction    = direction,
        entry_price  = physical_basis_usd,
        target_price = target,
        stop_price   = physical_basis_usd - 0.20,
        legs         = [
            dict(action="BUY" if direction==Direction.LONG else "SELL",
                 instrument="PHYSICAL_WTI", qty=1),
            dict(action="SELL" if direction==Direction.LONG else "BUY",
                 instrument=WTI_TICKER, qty=1),
        ],
        net_premium  = 0.0,
        max_profit   = profit_per_bbl * WTI_BBL_PER_CONTRACT,
        max_loss     = 0.20 * WTI_BBL_PER_CONTRACT,
        dte          = 30,
        confidence   = conf,
        vol_regime   = VolRegime.NORMAL,
        market_regime= MarketRegime.FLAT,
        rationale    = rationale,
    )


def generate_options_signal(
    F: float, r: float, sigma: float,
    curve: ForwardCurve, vol_regime: VolRegime,
) -> TradeSignal:
    """
    Select commodity options strategy based on vol regime + curve structure.

    Sherbin / Bittman decision tree for energy options:
    RICH vol + CONTANGO   → sell OTM call spread (energy doesn't rally in contango)
    RICH vol + BACKWARDAT → iron condor (backwardation limits downside; IV high)
    CHEAP vol + BACKWARDAT → long call (bullish, cheap premium)
    CHEAP vol + CONTANGO  → long put (bearish structure, cheap premium)
    """
    T45  = 45 / 365.0
    atm  = round(F, 0)
    otm_call = round(F * 1.07, 0)
    otm_put  = round(F * 0.93, 0)
    wing_call= round(F * 1.12, 0)
    wing_put = round(F * 0.88, 0)
    exp_str  = (datetime.date.today() + datetime.timedelta(days=45)).isoformat()

    bsm_c = black76(F, otm_call, T45, r, sigma, OptionRight.CALL)
    bsm_p = black76(F, otm_put,  T45, r, sigma, OptionRight.PUT)
    bsm_wc= black76(F, wing_call,T45, r, sigma, OptionRight.CALL)
    bsm_wp= black76(F, wing_put, T45, r, sigma, OptionRight.PUT)

    is_rich   = vol_regime in (VolRegime.HIGH, VolRegime.SPIKE)
    is_contango = curve.structure in ("CONTANGO", "SUPER_CONTANGO")

    if is_rich and is_contango:
        credit = bsm_c.price - bsm_wc.price
        strat  = StrategyType.BULL_CALL_SPREAD  # actually selling call spread
        direction = Direction.SHORT
        legs = [
            dict(right="C", strike=otm_call, expiry=exp_str, action="SELL", qty=1),
            dict(right="C", strike=wing_call,expiry=exp_str, action="BUY",  qty=1),
        ]
        max_profit = credit * 100
        max_loss   = (wing_call - otm_call - credit) * 100
        rationale  = (
            f"Rich vol ({sigma:.1%}) + contango. Sell call spread {int(otm_call)}/{int(wing_call)} "
            f"for ${credit:.2f} credit. Structure limits upside."
        )
        conf = 0.70

    elif is_rich and not is_contango:
        # Iron condor in backwardation: high vol, range-bound
        cc  = bsm_c.price - bsm_wc.price
        cp  = bsm_p.price - bsm_wp.price
        tot = cc + cp
        strat = StrategyType.IRON_CONDOR
        direction = Direction.FLAT
        legs = [
            dict(right="P", strike=wing_put, expiry=exp_str, action="BUY",  qty=1),
            dict(right="P", strike=otm_put,  expiry=exp_str, action="SELL", qty=1),
            dict(right="C", strike=otm_call, expiry=exp_str, action="SELL", qty=1),
            dict(right="C", strike=wing_call,expiry=exp_str, action="BUY",  qty=1),
        ]
        max_profit = tot * 100
        max_loss   = (max(wing_call - otm_call, otm_put - wing_put) - tot) * 100
        rationale  = (
            f"Rich vol ({sigma:.1%}) + backwardation. Iron condor "
            f"[{int(wing_put)}/{int(otm_put)}/{int(otm_call)}/{int(wing_call)}] "
            f"for ${tot:.2f} credit."
        )
        conf = 0.65; credit = tot

    elif not is_rich and not is_contango:
        # Cheap vol + backwardation = bullish → long call
        debit = bsm_c.price
        strat = StrategyType.LONG_CALL
        direction = Direction.LONG
        legs = [dict(right="C", strike=otm_call, expiry=exp_str, action="BUY", qty=1)]
        max_profit = (F * 0.20) * 100   # 20% upside target
        max_loss   = debit * 100
        credit     = -debit
        rationale  = (
            f"Cheap vol ({sigma:.1%}) + backwardation (bullish). "
            f"Buy {int(otm_call)} call for ${debit:.2f}."
        )
        conf = 0.60

    else:
        # Cheap vol + contango = bearish → long put
        debit = bsm_p.price
        strat = StrategyType.LONG_PUT
        direction = Direction.SHORT
        legs = [dict(right="P", strike=otm_put, expiry=exp_str, action="BUY", qty=1)]
        max_profit = (F * 0.20) * 100
        max_loss   = debit * 100
        credit     = -debit
        rationale  = (
            f"Cheap vol ({sigma:.1%}) + contango (bearish). "
            f"Buy {int(otm_put)} put for ${debit:.2f}."
        )
        conf = 0.58

    return TradeSignal(
        ticker        = WTI_TICKER,
        strategy      = strat,
        direction     = direction,
        entry_price   = F,
        target_price  = F * (1.07 if direction == Direction.LONG else 0.93),
        stop_price    = F * (0.95 if direction == Direction.LONG else 1.05),
        legs          = legs,
        net_premium   = credit * 100 if 'credit' in dir() else 0.0,
        max_profit    = max_profit,
        max_loss      = max_loss,
        dte           = 45,
        confidence    = conf,
        vol_regime    = vol_regime,
        market_regime = MarketRegime.CONTANGO_BEARISH if is_contango
                        else MarketRegime.BACKWARDATION_BULLISH,
        rationale     = rationale,
    )


def generate_futures_signal(
    curve: ForwardCurve,
    garch: HistoricalVolRegime,
    season: SeasonalPattern,
) -> TradeSignal:
    """
    Directional futures signal combining term structure + GARCH + seasonality.

    Oil Trader Academy: trade WITH the term structure. Backwardation → bullish
    (physical demand stronger than deferred; spot buyers are urgent).
    Contango → bearish bias (supply surplus, storage filling up).

    Gkinis Ch.2: convenience yield is the marginal benefit of holding
    physical inventory. High y = backwardation = strong physical demand.
    """
    is_backwardation  = "BACKWARDATION" in curve.structure
    is_normal_vol     = garch.vol_state == "NORMAL_VOL_REGIME"
    month             = datetime.date.today().month
    seasonal_factor   = season.get(month)
    is_seasonal_bull  = seasonal_factor >= 1.005

    if is_backwardation and is_seasonal_bull:
        direction = Direction.LONG
        conf      = 0.72 + (0.01 if is_normal_vol else -0.05)
        rationale = (
            f"Backwardation ({curve.structure}) + bullish seasonal "
            f"(factor={seasonal_factor:.3f}, month={month}). "
            f"GARCH: σ_ann={garch.sigma_annual:.1%}, state={garch.vol_state}."
        )
    elif not is_backwardation and not is_seasonal_bull:
        direction = Direction.SHORT
        conf      = 0.65 + (0.01 if is_normal_vol else -0.05)
        rationale = (
            f"Contango ({curve.structure}) + bearish seasonal "
            f"(factor={seasonal_factor:.3f}). GARCH σ={garch.sigma_annual:.1%}."
        )
    else:
        direction = Direction.FLAT
        conf      = 0.40
        rationale = (
            f"Conflicting signals: curve={curve.structure}, "
            f"seasonal_factor={seasonal_factor:.3f}. Stay flat."
        )

    atr_pct = garch.sigma_daily * 2.0   # 2× daily vol as ATR proxy
    return TradeSignal(
        ticker        = WTI_TICKER,
        strategy      = StrategyType.FUTURES_LONG if direction == Direction.LONG
                        else StrategyType.FUTURES_SHORT,
        direction     = direction,
        entry_price   = curve.spot,
        target_price  = curve.spot * (1 + atr_pct * 3),
        stop_price    = curve.spot * (1 - atr_pct * 2),
        legs          = [dict(instrument=WTI_TICKER, action=direction.value, qty=1)],
        net_premium   = 0.0,
        max_profit    = atr_pct * 3 * curve.spot * WTI_BBL_PER_CONTRACT,
        max_loss      = atr_pct * 2 * curve.spot * WTI_BBL_PER_CONTRACT,
        dte           = 30,
        confidence    = conf,
        vol_regime    = VolRegime.HIGH if garch.sigma_annual > 0.40 else VolRegime.NORMAL,
        market_regime = MarketRegime.BACKWARDATION_BULLISH if is_backwardation
                        else MarketRegime.CONTANGO_BEARISH,
        rationale     = rationale,
    )


# ============================================================================
# 7. ML SIGNAL GENERATOR
# ============================================================================

def train_ml_signal(ml_data: MLSignalData) -> Optional["GradientBoostingClassifier"]:
    """
    Train GradientBoostingClassifier on commodity price features.

    QuantStart: walk-forward validation preferred over random split.
    Use last 20% of data as out-of-sample test. Report accuracy separately
    on in-sample and out-of-sample to detect overfitting.
    """
    if not _SKL:
        logger.warning("scikit-learn not available — skipping ML training.")
        return None
    if ml_data.features.empty or ml_data.target is None:
        return None

    X = ml_data.features.values
    y = ml_data.target.values
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y  = X[valid], y[valid]

    if len(X) < 100:
        return None

    split    = int(len(X) * 0.80)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    if ml_data.scaler:
        X_tr = ml_data.scaler.transform(X_tr)
        X_te = ml_data.scaler.transform(X_te)

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    clf.fit(X_tr, y_tr)

    acc_is = accuracy_score(y_tr, clf.predict(X_tr))
    acc_os = accuracy_score(y_te, clf.predict(X_te))
    logger.info("ML signal: in-sample acc=%.2f%%, out-of-sample acc=%.2f%%",
                acc_is*100, acc_os*100)
    return clf


def ml_predict_direction(
    clf: "GradientBoostingClassifier",
    ml_data: MLSignalData,
) -> Tuple[int, float]:
    """Return (predicted_direction, probability) for most recent data point."""
    if clf is None or ml_data.features.empty:
        return 0, 0.33
    X = ml_data.features.values[-1:].copy()
    if ml_data.scaler:
        X = ml_data.scaler.transform(X)
    pred    = int(clf.predict(X)[0])
    prob_arr= clf.predict_proba(X)[0]
    prob    = float(max(prob_arr))
    return pred, prob


# ============================================================================
# 8. THREE-STATEMENT FINANCIAL MODEL
# ============================================================================

@dataclass
class ThreeStatementModel:
    """
    Institutional three-statement model for a commodity trading operation.

    Components:
    ① Income Statement (P&L) — trading revenues, hedging gains/losses, fees
    ② Balance Sheet — assets (positions MTM), liabilities (margin, debt), equity
    ③ Cash Flow Statement — operating, investing, financing flows

    PPM standard: fund documents require three-statement projections for 3–5 years
    with explicit VaR constraints, stress scenarios, and Sharpe ratio targets.
    """
    fund_name:     str
    period:        str   # e.g. "FY2025"

    # Income Statement items (USD)
    trading_revenue:   float = 0.0
    hedging_pnl:       float = 0.0
    roll_yield_income: float = 0.0
    collateral_income: float = 0.0
    gross_income:      float = 0.0
    mgmt_fees:         float = 0.0
    performance_fees:  float = 0.0
    transaction_costs: float = 0.0
    net_income:        float = 0.0

    # Balance Sheet items (USD)
    cash_collateral:   float = 0.0
    futures_mtm:       float = 0.0
    options_mtm:       float = 0.0
    total_assets:      float = 0.0
    margin_liabilities:float = 0.0
    net_equity:        float = 0.0

    # Cash Flow items (USD)
    cf_operating:      float = 0.0
    cf_investing:      float = 0.0
    cf_financing:      float = 0.0
    net_cash_flow:     float = 0.0

    # Performance metrics
    sharpe_ratio:      float = 0.0
    sortino_ratio:     float = 0.0
    max_drawdown_pct:  float = 0.0
    var_99_1d_usd:     float = 0.0

    def summary(self) -> str:
        lines = [
            f"\n{'═'*65}",
            f"  THREE-STATEMENT MODEL: {self.fund_name} | {self.period}",
            f"{'═'*65}",
            f"  INCOME STATEMENT",
            f"  {'Trading Revenue':<35} ${self.trading_revenue:>12,.0f}",
            f"  {'Hedging P&L':<35} ${self.hedging_pnl:>12,.0f}",
            f"  {'Roll Yield Income':<35} ${self.roll_yield_income:>12,.0f}",
            f"  {'Collateral Income':<35} ${self.collateral_income:>12,.0f}",
            f"  {'Gross Income':<35} ${self.gross_income:>12,.0f}",
            f"  {'Management Fees':<35} $({abs(self.mgmt_fees):>11,.0f})",
            f"  {'Performance Fees':<35} $({abs(self.performance_fees):>11,.0f})",
            f"  {'Transaction Costs':<35} $({abs(self.transaction_costs):>11,.0f})",
            f"  {'Net Income':<35} ${self.net_income:>12,.0f}",
            f"{'─'*65}",
            f"  BALANCE SHEET",
            f"  {'Cash & Collateral':<35} ${self.cash_collateral:>12,.0f}",
            f"  {'Futures MTM':<35} ${self.futures_mtm:>12,.0f}",
            f"  {'Options MTM':<35} ${self.options_mtm:>12,.0f}",
            f"  {'Total Assets':<35} ${self.total_assets:>12,.0f}",
            f"  {'Margin Liabilities':<35} $({abs(self.margin_liabilities):>11,.0f})",
            f"  {'Net Equity':<35} ${self.net_equity:>12,.0f}",
            f"{'─'*65}",
            f"  CASH FLOW STATEMENT",
            f"  {'CF — Operating':<35} ${self.cf_operating:>12,.0f}",
            f"  {'CF — Investing':<35} ${self.cf_investing:>12,.0f}",
            f"  {'CF — Financing':<35} ${self.cf_financing:>12,.0f}",
            f"  {'Net Cash Flow':<35} ${self.net_cash_flow:>12,.0f}",
            f"{'─'*65}",
            f"  RISK & PERFORMANCE",
            f"  {'Sharpe Ratio':<35} {self.sharpe_ratio:>12.3f}",
            f"  {'Sortino Ratio':<35} {self.sortino_ratio:>12.3f}",
            f"  {'Max Drawdown':<35} {self.max_drawdown_pct:>11.2f}%",
            f"  {'1-Day 99% VaR':<35} ${self.var_99_1d_usd:>12,.0f}",
            f"{'═'*65}",
        ]
        return "\n".join(lines)


def build_three_statement(
    nav: NAVModel,
    signals: List[TradeSignal],
    account_equity: float,
) -> ThreeStatementModel:
    """
    Populate three-statement model from live NAV + signal data.
    Annualises current positions to full-year projected P&L.
    """
    trading_rev     = nav.futures_pnl_usd
    roll_income     = nav.roll_yield_ann * account_equity
    coll_income     = nav.collateral_yield_ann * account_equity
    gross           = trading_rev + roll_income + coll_income
    mgmt_fee        = -nav.mgmt_fee_ann * account_equity
    txn_cost        = -len(signals) * 500   # $500/trade estimate
    net             = gross + mgmt_fee + txn_cost

    futures_mtm     = nav.futures_pnl_usd
    options_mtm     = sum(s.net_premium for s in signals if "OPTION" in s.strategy.value or
                          s.strategy in (StrategyType.LONG_CALL, StrategyType.LONG_PUT,
                                          StrategyType.IRON_CONDOR, StrategyType.COLLAR,
                                          StrategyType.BULL_CALL_SPREAD, StrategyType.BEAR_PUT_SPREAD,
                                          StrategyType.CALENDAR_SPREAD))
    cash            = account_equity * 0.85
    total_assets    = cash + futures_mtm + options_mtm + account_equity * 0.15
    margin          = -(account_equity * 0.10)
    equity          = total_assets + margin

    return ThreeStatementModel(
        fund_name          = nav.fund_name,
        period             = f"FY{datetime.date.today().year}",
        trading_revenue    = trading_rev,
        hedging_pnl        = 0.0,
        roll_yield_income  = roll_income,
        collateral_income  = coll_income,
        gross_income       = gross,
        mgmt_fees          = mgmt_fee,
        performance_fees   = 0.0,
        transaction_costs  = txn_cost,
        net_income         = net,
        cash_collateral    = cash,
        futures_mtm        = futures_mtm,
        options_mtm        = options_mtm,
        total_assets       = total_assets,
        margin_liabilities = margin,
        net_equity         = equity,
        cf_operating       = net,
        cf_investing       = 0.0,
        cf_financing       = 0.0,
        net_cash_flow      = net,
    )


# ============================================================================
# 9. PPM SECTION GENERATOR
# ============================================================================

def generate_ppm_sections(
    nav: NAVModel,
    three_stmt: ThreeStatementModel,
    mc_gbm: MCSimResult,
    mc_ou: MCSimResult,
    signals: List[TradeSignal],
) -> str:
    """
    Private Placement Memorandum section generator.
    Generates institutional-grade PPM text for a commodity fund.

    Standard PPM sections (SEC / CFTC NFA requirements):
    I.   Executive Summary
    II.  Investment Strategy & Objectives
    III. Risk Factors
    IV.  Fee Structure
    V.   Financial Projections (Three-Statement Model)
    VI.  Risk Management Framework (VaR, Stress Tests)
    VII. Portfolio Construction
    """
    today = datetime.date.today().isoformat()
    lines = [
        f"\n{'▓'*70}",
        f"  PRIVATE PLACEMENT MEMORANDUM",
        f"  {nav.fund_name}",
        f"  Date: {today}",
        f"{'▓'*70}",
        "",
        "─── SECTION I: EXECUTIVE SUMMARY ──────────────────────────────────",
        f"  Fund Name:       {nav.fund_name}",
        f"  Commodity Focus: {nav.commodity} (WTI Crude Oil & Petroleum Products)",
        f"  Strategy:        Quantitative multi-strategy: storage arb, crack spreads,",
        f"                   basis trading, directional futures, options vol strategies",
        f"  NAV/Unit:        ${nav.nav_per_unit:,.2f}",
        f"  Total NAV:       ${nav.total_nav_usd:,.0f}",
        f"  Target Return:   {(nav.net_return_pct*100):.1f}% net p.a.",
        f"  Max Drawdown:    <15% (hard limit enforced by risk_agent)",
        "",
        "─── SECTION II: INVESTMENT STRATEGY ────────────────────────────────",
        "  The Fund employs four primary alpha sources:",
        "  1. STORAGE ARBITRAGE: Exploit contango via buy-spot/sell-forward when",
        "     F(T) > S·exp((r+u−y)·T). Risk-free when fully funded at Cushing.",
        "  2. CRACK SPREAD TRADING: 3-2-1 and 5-3-2 spread mean reversion.",
        "     Refinery margin historically mean-reverts to $8-12/bbl range.",
        "  3. BASIS TRADING: Physical vs. futures differential convergence.",
        "     Targets pipeline/quality dislocation (e.g., Midland/Cushing basis).",
        "  4. OPTIONS STRATEGIES: Sell premium in RICH vol regimes (iron condors,",
        "     credit spreads); buy premium in CHEAP regimes (Black-76 pricing).",
        "",
        "─── SECTION III: RISK FACTORS ──────────────────────────────────────",
        "  KEY RISKS:",
        "  • Commodity Price Risk: WTI daily vol ≈ 2-3%. Adverse moves amplified",
        f"    by leverage. Current GARCH σ_ann ≈ 35% (see risk_agent stress tests).",
        "  • Basis Risk: Physical-futures correlation can break down (pipeline events).",
        "  • Liquidity Risk: Front-month CL is liquid (>500K lots/day); back months",
        "    and options on deferred contracts have wider bid-ask spreads.",
        "  • Contango/Backwardation Regime Flip: Storage arb unwinds if contango",
        "    collapses (sudden inventory draw / geopolitical supply shock).",
        "  • Regulatory Risk: CFTC position limits, NFA reporting requirements.",
        "  • Model Risk: Black-76 assumes log-normal F distribution. Actual WTI",
        "    returns exhibit fat tails (excess kurtosis ≈ 4-6).",
        "",
        "─── SECTION IV: FEE STRUCTURE ───────────────────────────────────────",
        f"  Management Fee:  {nav.mgmt_fee_ann:.0%} per annum",
        f"  Performance Fee: {nav.perf_fee:.0%} above {nav.hurdle_rate:.0%} hurdle",
        "  High-Water Mark: Yes (performance fee only on new highs)",
        "  Redemption:      Monthly, 30-day notice",
        "",
        "─── SECTION V: FINANCIAL PROJECTIONS ───────────────────────────────",
        three_stmt.summary(),
        "",
        "─── SECTION VI: RISK MANAGEMENT FRAMEWORK ──────────────────────────",
        f"  1-Day 99% VaR:        ${three_stmt.var_99_1d_usd:>12,.0f}",
        f"  GBM Monte Carlo P5:   ${mc_gbm.p5:>12.2f}/bbl",
        f"  GBM Monte Carlo P95:  ${mc_gbm.p95:>12.2f}/bbl",
        f"  OU  Monte Carlo P5:   ${mc_ou.p5:>12.2f}/bbl",
        f"  OU  Monte Carlo P95:  ${mc_ou.p95:>12.2f}/bbl",
        f"  Prob(price > spot):   GBM={mc_gbm.prob_above_spot:.1%} | OU={mc_ou.prob_above_spot:.1%}",
        "  Stress scenarios: see risk_agent.py — GEOPOLITICAL_SHOCK, DEMAND_COLLAPSE,",
        "  CONTANGO_FLIP, SUPPLY_CUT. All modelled with GARCH-estimated σ.",
        "",
        "─── SECTION VII: PORTFOLIO CONSTRUCTION ─────────────────────────────",
        f"  Approved Signals:  {len(signals)}",
    ]
    for i, sig in enumerate(signals, 1):
        lines.append(
            f"  {i:2d}. {sig.strategy.value:<30} {sig.direction.value:<6} "
            f"conf={sig.confidence:.0%} EV=${sig.expected_value:,.0f}"
        )
    lines.append(f"\n{'▓'*70}\n")
    return "\n".join(lines)
