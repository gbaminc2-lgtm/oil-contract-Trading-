"""
data_agent.py — Quantitative Energy Commodity Data Engine
==========================================================
System Role: Expert Quantitative Commodity Strategist (Institutional Grade)

Knowledge sources embedded:
  Oil Trader Academy · Hedging Strategies in Crude Oil Futures
  Hedging Strategies for Oil End-Users · Commodities Demystified (Trafigura)
  NYMEX Chapter 200 (WTI specs) · Modelling Energy Markets (Gkinis)
  Oil Contracts: How to Read Petroleum Contracts
  How to Price and Trade Options (Sherbin / Bloomberg)
  Trading Options as a Professional (Bittman / McGraw-Hill)
  Risk Management & Financial Institutions (Hull 4th Ed.)
  Successful Algorithmic Trading (QuantStart)
  The Art & Science of Technical Analysis (Grimes)

Learning modes:
  ① PAST:    Historical seasonal patterns, crack-spread history, geopolitical shocks
  ② REAL-TIME: Streaming spot prices, options chain, OI, basis
  ③ FUTURE:  Monte Carlo forward curves, delta-hedging sims, contango/backwardation

pip install yfinance pandas numpy scipy scikit-learn requests
"""

from __future__ import annotations

import datetime
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── optional guards ──────────────────────────────────────────────────────────
try:
    import yfinance as yf;  _YF = True
except ImportError:
    _YF = False;  logger.warning("yfinance missing. pip install yfinance")

try:
    import requests;  _REQ = True
except ImportError:
    _REQ = False

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    _SKL = True
except ImportError:
    _SKL = False;  logger.warning("scikit-learn missing. pip install scikit-learn")

try:
    from market_architecture import get_market_arch as _get_mam
    _MAM = True
except ImportError:
    _MAM = False

# ============================================================================
# 1. CONTRACT SPECIFICATIONS  (NYMEX Chapter 200 + ICE Brent + products)
# ============================================================================
WTI_TICKER          = "CL=F"
BRENT_TICKER        = "BZ=F"
RBOB_TICKER         = "RB=F"
ULSD_TICKER         = "HO=F"
NATGAS_TICKER       = "NG=F"

WTI_BBL_PER_CONTRACT    = 1_000     # NYMEX Ch.200 §200.00
WTI_TICK_USD            = 0.01      # minimum tick per barrel
WTI_TICK_VALUE          = 10.00     # tick value per contract ($)
WTI_DELIVERY_POINT      = "Cushing, Oklahoma Pipeline"
WTI_API_GRAVITY_MIN     = 37.0
WTI_SULFUR_MAX_PCT      = 0.42
PRODUCT_GAL_PER_BBL     = 42.0      # 1 bbl = 42 US gallons

TRADING_DAYS_YEAR       = 252
RISK_FREE_FALLBACK      = 0.053     # 5.3% — update from FRED at runtime

# ============================================================================
# 2. DATA CONTAINERS
# ============================================================================

@dataclass
class ForwardCurve:
    """
    Commodity forward price curve (term structure).

    Gkinis Ch.2–3: F(T) = S · exp((r + u − y) · T)
    r = risk-free rate, u = storage cost rate (annualised)
    y = convenience yield (annualised)

    Structure classification:
      Contango      F > S  →  carry trade / storage arbitrage possible
      Backwardation F < S  →  tight physical, producer hedge urgency
      Super-contango          Extreme contango: fill-all-storage signal
    """
    commodity:          str
    spot:               float
    timestamp:          datetime.datetime
    tenors_months:      List[int]
    prices:             List[float]
    convenience_yields: List[float]
    storage_cost_rate:  float = 0.042   # annualised $/bbl as % of spot
    financing_rate:     float = RISK_FREE_FALLBACK

    @property
    def structure(self) -> str:
        if len(self.prices) < 2:
            return "FLAT"
        slope = self.prices[-1] - self.spot
        if slope > 2.0:
            return "SUPER_CONTANGO"
        elif slope > 0.50:
            return "CONTANGO"
        elif slope < -2.0:
            return "STRONG_BACKWARDATION"
        elif slope < -0.50:
            return "BACKWARDATION"
        return "FLAT"

    @property
    def m1_m2_spread(self) -> float:
        """Prompt vs. second-month roll differential."""
        if len(self.prices) < 2:
            return 0.0
        return self.prices[0] - self.prices[1]

    @property
    def annualised_roll_yield(self) -> float:
        """
        Roll yield = − (F₁₂ − S) / S (positive in backwardation).
        Key input for NAV model and fund performance attribution.
        """
        if not self.prices or self.spot <= 0:
            return 0.0
        return -(self.prices[-1] - self.spot) / self.spot

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "tenor_months"     : self.tenors_months,
            "price"            : self.prices,
            "basis_to_spot"    : [p - self.spot for p in self.prices],
            "conv_yield_ann"   : self.convenience_yields,
            "ann_roll_yield_pct": [
                -(p - self.spot) / self.spot * 100 for p in self.prices
            ],
        })


@dataclass
class CrackSpread:
    """
    Petroleum product crack spread — proxy for refinery gross margin.

    Trafigura (Commodities Demystified) Ch.4:
    3-2-1 crack: buy 3 bbl crude → sell 2 bbl gasoline + 1 bbl distillate
    5-3-2 crack: buy 5 bbl crude → sell 3 bbl gasoline + 2 bbl distillate

    Trading implication (Oil Trader Academy):
    Rising crack spread → refiners buy more crude → WTI demand support
    Falling crack spread → refinery run-cuts → bearish crude demand
    """
    timestamp:              datetime.datetime
    wti_spot:               float
    brent_spot:             float
    rbob_per_bbl:           float
    ulsd_per_bbl:           float
    crack_321:              float   # $/bbl
    crack_532:              float   # $/bbl
    wti_brent_diff:         float   # WTI − Brent (negative = Brent premium)
    ho_rbob_spread:         float   # heating oil vs. gasoline (seasonal)

    @property
    def refinery_margin_healthy(self) -> bool:
        """3-2-1 > $8/bbl = refineries running at full rates."""
        return self.crack_321 > 8.0

    @property
    def distillate_premium(self) -> bool:
        return self.ho_rbob_spread > 0


@dataclass
class StorageEconomics:
    """
    Cushing, OK storage arbitrage economics.

    Oil Trader Academy: NYMEX WTI delivery at Cushing. Capacity ~76 MMbbl.
    Storage arb condition (Gkinis Ch.2 / no-arbitrage):
      F(T) > S + u·T + r·S·T − y·S·T
    Simplified monthly: arb_profit = F(2m) − spot − carry_cost_2m > 0
    """
    timestamp:               datetime.datetime
    spot_wti:                float
    m1_price:                float
    m2_price:                float
    storage_cost_bbl_mo:     float = 0.35
    financing_rate_ann:      float = RISK_FREE_FALLBACK
    convenience_yield_ann:   float = 0.025
    cushing_utilisation_pct: Optional[float] = None

    @property
    def monthly_carry(self) -> float:
        return self.storage_cost_bbl_mo + self.spot_wti * self.financing_rate_ann / 12

    @property
    def theoretical_m1(self) -> float:
        r  = self.financing_rate_ann
        u  = self.storage_cost_bbl_mo * 12 / max(self.spot_wti, 1)
        y  = self.convenience_yield_ann
        return self.spot_wti * math.exp((r + u - y) / 12)

    @property
    def storage_arb_pnl(self) -> float:
        """Buy spot + store 2 months + sell M2 forward = locked $/bbl."""
        return self.m2_price - self.spot_wti - self.monthly_carry * 2

    @property
    def arb_available(self) -> bool:
        return self.storage_arb_pnl > 0.15


@dataclass
class HistoricalVolRegime:
    """
    GARCH(1,1) volatility term structure for energy commodities.

    Hull Ch.10: GARCH(1,1) — σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)
    Long-run variance VL = ω / (1 − α − β)
    Persistence = α + β. Near 1 = long vol memory (energy markets typical).
    """
    ticker:             str
    sigma_daily:        float
    sigma_annual:       float
    sigma_30d_forecast: float
    sigma_long_run:     float
    alpha:              float
    beta:               float
    omega:              float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def vol_state(self) -> str:
        if self.sigma_annual > self.sigma_long_run * 1.3:
            return "HIGH_VOL_REGIME"
        elif self.sigma_annual < self.sigma_long_run * 0.7:
            return "LOW_VOL_REGIME"
        return "NORMAL_VOL_REGIME"


@dataclass
class SeasonalPattern:
    """
    Monthly seasonal price factors derived from 5+ year history.
    Factor = 1.0 → no seasonal; > 1.0 → seasonally bullish.
    """
    commodity: str
    factors:   Dict[int, float]

    def get(self, month: int) -> float:
        return self.factors.get(month, 1.0)

    def bullish_months(self, threshold: float = 1.01) -> List[int]:
        return [m for m, f in self.factors.items() if f >= threshold]

    def bearish_months(self, threshold: float = 0.99) -> List[int]:
        return [m for m, f in self.factors.items() if f <= threshold]


@dataclass
class MLSignalData:
    """Feature matrix + target for supervised ML signal generation."""
    timestamp:     datetime.datetime
    features:      pd.DataFrame
    feature_names: List[str]
    target:        Optional[pd.Series] = None
    scaler:        object = None


# ============================================================================
# 3. CORE FETCH FUNCTIONS
# ============================================================================

def fetch_spot(ticker: str) -> float:
    """Fetch latest price; synthetic fallback for offline use."""
    if not _YF:
        return _synth_spot(ticker)
    try:
        tkr = yf.Ticker(ticker)
        fi  = tkr.fast_info
        for k in ("lastPrice", "last_price", "regularMarketPrice"):
            p = fi.get(k) if hasattr(fi, "get") else getattr(fi, k, None)
            if p and float(p) > 0:
                return float(p)
        h = tkr.history(period="1d")
        if not h.empty:
            return float(h["Close"].iloc[-1])
    except Exception as e:
        logger.error("fetch_spot(%s): %s", ticker, e)
    return _synth_spot(ticker)


def fetch_ohlcv(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """2-year OHLCV minimum for seasonal analysis and GARCH estimation."""
    if not _YF:
        return _synth_ohlcv(ticker, 504)
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception as e:
        logger.error("fetch_ohlcv(%s): %s", ticker, e)
    return _synth_ohlcv(ticker, 504)


def fetch_forward_curve(commodity: str = "WTI") -> ForwardCurve:
    """
    Build no-arbitrage forward curve via F(T) = S·exp((r+u−y)·T).

    Gkinis: calibrate u and y to observable market spreads when available.
    Default values: u=4.2%/yr (Cushing storage), y=2.5%/yr (avg convenience).
    Production: replace with CME DataMine or broker API for full curve.
    """
    ticker = WTI_TICKER if commodity == "WTI" else BRENT_TICKER
    S      = fetch_spot(ticker)
    r      = fetch_risk_free_rate()
    u, y   = 0.042, 0.025

    tenors, prices, cy_list = [1, 2, 3, 6, 9, 12], [], []
    for T_mo in tenors:
        T   = T_mo / 12.0
        F   = S * math.exp((r + u - y) * T)
        cy  = r + u - math.log(F / S) / T if F > 0 and T > 0 else y
        prices.append(round(F, 2))
        cy_list.append(round(cy, 4))

    return ForwardCurve(
        commodity=commodity, spot=S,
        timestamp=datetime.datetime.utcnow(),
        tenors_months=tenors, prices=prices, convenience_yields=cy_list,
        financing_rate=r,
    )


def fetch_crack_spreads() -> CrackSpread:
    """
    Calculate petroleum product crack spreads.
    yfinance quotes RBOB & ULSD in $/gal → convert to $/bbl × 42.
    """
    wti    = fetch_spot(WTI_TICKER)
    brent  = fetch_spot(BRENT_TICKER)
    rbob_g = fetch_spot(RBOB_TICKER)
    ulsd_g = fetch_spot(ULSD_TICKER)

    rbob_bbl = rbob_g * PRODUCT_GAL_PER_BBL
    ulsd_bbl = ulsd_g * PRODUCT_GAL_PER_BBL

    crack_321 = (2/3 * rbob_bbl + 1/3 * ulsd_bbl) - wti
    crack_532 = (3/5 * rbob_bbl + 2/5 * ulsd_bbl) - wti

    logger.info("Crack 3-2-1: $%.2f/bbl | 5-3-2: $%.2f/bbl | WTI-Brent: $%.2f",
                crack_321, crack_532, wti - brent)
    return CrackSpread(
        timestamp=datetime.datetime.utcnow(),
        wti_spot=wti, brent_spot=brent,
        rbob_per_bbl=rbob_bbl, ulsd_per_bbl=ulsd_bbl,
        crack_321=crack_321, crack_532=crack_532,
        wti_brent_diff=wti - brent,
        ho_rbob_spread=ulsd_bbl - rbob_bbl,
    )


def fetch_storage_economics() -> StorageEconomics:
    curve = fetch_forward_curve("WTI")
    m1 = curve.prices[0] if curve.prices else curve.spot
    m2 = curve.prices[1] if len(curve.prices) > 1 else m1
    return StorageEconomics(
        timestamp=datetime.datetime.utcnow(),
        spot_wti=curve.spot, m1_price=m1, m2_price=m2,
        financing_rate_ann=curve.financing_rate,
    )


def fetch_garch_vol(ticker: str = WTI_TICKER) -> HistoricalVolRegime:
    """
    GARCH(1,1) with variance targeting.
    Hull Ch.10 / Gkinis Ch.4: for crude oil α≈0.10, β≈0.85 is typical.
    """
    df  = fetch_ohlcv(ticker, period="2y")
    close = df["Close"].squeeze()
    ret = np.log(close / close.shift(1)).dropna().values

    long_var = float(np.var(ret[-252:]))
    alpha, beta = 0.10, 0.85
    omega = long_var * (1 - alpha - beta)

    h = np.full(len(ret), long_var)
    for t in range(1, len(ret)):
        h[t] = omega + alpha * ret[t-1]**2 + beta * h[t-1]

    sigma_d  = math.sqrt(h[-1])
    sigma_a  = sigma_d * math.sqrt(TRADING_DAYS_YEAR)
    sigma_lr = math.sqrt(long_var * TRADING_DAYS_YEAR)
    kappa    = 1 - alpha - beta
    sigma_30 = math.sqrt(
        long_var + (h[-1] - long_var) * (1 - kappa**30) / max(30*kappa, 1e-9)
    ) * math.sqrt(TRADING_DAYS_YEAR)

    return HistoricalVolRegime(
        ticker=ticker,
        sigma_daily=sigma_d, sigma_annual=sigma_a,
        sigma_30d_forecast=sigma_30, sigma_long_run=sigma_lr,
        alpha=alpha, beta=beta, omega=omega,
    )


def fetch_seasonal_pattern(ticker: str = WTI_TICKER) -> SeasonalPattern:
    """Extract 12-month seasonal factors from 5-year price history."""
    df = fetch_ohlcv(ticker, period="5y")
    if len(df) < 252:
        return _default_seasonal()

    df = df.copy()
    df["month"] = df.index.month
    df["ret"]   = df["Close"].pct_change()
    monthly_avg = df.groupby("month")["ret"].mean()
    grand_mean  = monthly_avg.mean()

    factors = {}
    for m in range(1, 13):
        raw = monthly_avg.get(m, grand_mean)
        factors[m] = round(1.0 + (raw - grand_mean) * 10, 4)
    return SeasonalPattern(commodity=ticker, factors=factors)


def build_ml_features(ticker: str = WTI_TICKER) -> MLSignalData:
    """
    Build feature matrix for ML-based commodity signal generation.

    QuantStart: all features lagged by 1 day. Target = next-day direction
    (1 = up > 0.5%, 0 = flat ±0.5%, -1 = down < -0.5%).
    Features: momentum (5/10/20/60d), vol ratio, RSI, seasonality,
    volume ratio, 52-week position, GARCH vol proxy.
    """
    df  = fetch_ohlcv(ticker, period="3y")
    if len(df) < 150:
        return MLSignalData(datetime.datetime.utcnow(), pd.DataFrame(), [])

    c   = df["Close"].squeeze()
    v   = df["Volume"].squeeze()
    ret = np.log(c / c.shift(1))
    feat = pd.DataFrame(index=df.index)

    for n in [5, 10, 20, 60]:
        feat[f"mom_{n}d"] = c.pct_change(n)
        feat[f"rvol_{n}d"] = ret.rolling(n).std() * math.sqrt(TRADING_DAYS_YEAR)

    lo52 = c.rolling(252, min_periods=100).min()
    hi52 = c.rolling(252, min_periods=100).max()
    feat["pos_52wk"] = (c - lo52) / (hi52 - lo52 + 1e-9)
    feat["vol_ratio"] = feat["rvol_5d"] / (feat["rvol_20d"] + 1e-9)

    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    feat["rsi14"] = 100 - 100 / (1 + gain / (loss + 1e-9))
    feat["vol_ratio_20ma"] = v / (v.rolling(20).mean() + 1e-9)
    feat["season_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    feat["season_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    feat["dow"]         = df.index.dayofweek

    next_ret = ret.shift(-1)
    tgt      = np.where(next_ret > 0.005, 1, np.where(next_ret < -0.005, -1, 0))
    feat     = feat.dropna()
    target   = pd.Series(tgt, index=df.index, name="target").reindex(feat.index)

    scaler = None
    if _SKL:
        scaler = StandardScaler()
        scaler.fit(feat.values)

    return MLSignalData(
        timestamp=datetime.datetime.utcnow(),
        features=feat, feature_names=list(feat.columns),
        target=target, scaler=scaler,
    )


def fetch_iv_surface(ticker: str = WTI_TICKER) -> Dict[Tuple[int, float], float]:
    """
    Build implied-volatility surface.
    Energy options use Black-76 (futures as underlying, q = r).
    Negative vol skew normal for crude: put wing elevated (crash risk).
    """
    if not _YF:
        return _synth_iv_surface()
    try:
        tkr   = yf.Ticker(ticker)
        spot  = fetch_spot(ticker)
        today = datetime.date.today()
        surf  = {}
        buckets = [0.90, 0.95, 1.00, 1.05, 1.10]
        for exp in (tkr.options or [])[:5]:
            chain = tkr.option_chain(exp)
            dte   = (datetime.date.fromisoformat(exp) - today).days
            if dte <= 0:
                continue
            for _, row in pd.concat([chain.calls, chain.puts]).iterrows():
                iv = float(row.get("impliedVolatility") or 0)
                K  = float(row.get("strike") or 0)
                if iv <= 0 or K <= 0 or spot <= 0:
                    continue
                nb = min(buckets, key=lambda b: abs(b - K / spot))
                surf[(dte, nb)] = iv
        return surf or _synth_iv_surface()
    except Exception as e:
        logger.error("fetch_iv_surface: %s", e)
        return _synth_iv_surface()


def fetch_risk_free_rate() -> float:
    """3-month T-bill from FRED; fallback 5.3%."""
    if not _REQ:
        return RISK_FREE_FALLBACK
    try:
        url  = ("https://fred.stlouisfed.org/graph/fredgraph.csv"
                "?id=DTB3&vintage_date=" + datetime.date.today().isoformat())
        r    = requests.get(url, timeout=5)
        if r.status_code == 200:
            return float(r.text.strip().split("\n")[-1].split(",")[-1]) / 100.0
    except Exception:
        pass
    return RISK_FREE_FALLBACK


def fetch_basis(physical_price: Optional[float] = None) -> Dict[str, float]:
    """
    Basis = physical − futures. Captures location/quality/timing premium.
    Oil Contracts book: physical deal pricing always references a benchmark
    plus a negotiated differential (basis).
    """
    futs = fetch_spot(WTI_TICKER)
    phys = physical_price or futs
    return {
        "futures": futs, "physical": phys,
        "basis": phys - futs,
        "basis_pct": (phys - futs) / futs * 100 if futs > 0 else 0.0,
        "type": "PREMIUM" if phys > futs else "DISCOUNT" if phys < futs else "FLAT",
    }


def fetch_exchange_latency() -> Dict[str, Dict]:
    """
    Return microwave-vs-fiber latency advantage for all known exchange routes.

    Market Architecture Math (Phase 1): microwave travels at ~0.67c in free
    air; fiber at ~0.67c through glass but with routing overhead → net
    microwave advantage of ~35-40% per route.

    Returns dict keyed by route name with advantage_microseconds,
    microwave_microseconds, fiber_microseconds, and route metadata.
    Falls back to empty dict if MAM unavailable.
    """
    if not _MAM:
        logger.debug("market_architecture not available; skipping latency fetch.")
        return {}
    mam = _get_mam()
    return mam.all_exchange_latencies()


# ============================================================================
# 4. SYNTHETIC FALLBACKS
# ============================================================================

def _synth_spot(ticker: str) -> float:
    seed = abs(hash(ticker)) % 1000
    if any(x in ticker for x in ["CL", "BZ"]):
        return 72.0 + seed * 0.04
    if "RB" in ticker:
        return 2.35 + seed * 0.0004
    if "HO" in ticker:
        return 2.55 + seed * 0.0004
    return 50.0 + seed * 0.12


def _synth_ohlcv(ticker: str, n: int = 504) -> pd.DataFrame:
    rng   = np.random.default_rng(abs(hash(ticker)) % 2**32)
    s0    = 72.0 if "CL" in ticker else 100.0
    close = s0 * np.cumprod(1 + rng.normal(0.00015, 0.019, n))
    dates = pd.date_range(end=datetime.date.today(), periods=n, freq="B")
    return pd.DataFrame({
        "Open": close*(1-rng.uniform(0,.005,n)), "High": close*(1+rng.uniform(0,.012,n)),
        "Low": close*(1-rng.uniform(0,.012,n)),  "Close": close,
        "Volume": rng.integers(150_000, 2_000_000, n),
    }, index=dates)


def _synth_iv_surface() -> Dict[Tuple[int, float], float]:
    surf = {}
    for dte in [21, 45, 60, 90, 120]:
        atm = 0.30 + dte / 365 * 0.03
        for m, skew in [(0.90, 0.12), (0.95, 0.05), (1.00, 0.00), (1.05, 0.02), (1.10, 0.04)]:
            surf[(dte, m)] = round(atm + skew, 4)
    return surf


def _default_seasonal() -> SeasonalPattern:
    return SeasonalPattern("WTI", {
        1:0.985, 2:0.990, 3:1.002, 4:1.008, 5:1.015, 6:1.012,
        7:1.008, 8:1.005, 9:0.998, 10:0.995, 11:0.998, 12:1.002,
    })
