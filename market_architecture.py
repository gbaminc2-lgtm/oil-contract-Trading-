"""
market_architecture.py — Market Microstructure Math Engine
===========================================================
Four-phase quantitative market microstructure calculator.

  Phase 1 — Latency engine: microsecond advantage of microwave over fiber-optic
  Phase 2 — Contract math: notional value, tick value, 3:2:1 crack spread
  Phase 3 — Black-Scholes: European call price + Greeks (EQUITY ONLY)
  Phase 4 — Risk engine: 1%-rule position sizing, parametric VaR

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠  PHASE 3 IS FOR EQUITY OPTIONS ONLY.
   WTI / Brent / RBOB / ULSD options MUST use black76() in strategy_agent.py.
   Black-Scholes assumes a spot underlying; energy options trade on futures
   and require Black-76 (q = r collapses to the forward price F).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Integration map:
  data_agent.py        ← fetch_exchange_latency()      (Phase 1)
  risk_engine.py       ← mam_position_size(), mam_var() (Phase 4)
  micro_futures.py     ← calculate_notional_and_tick()  (Phase 2)
  strategy_agent.py    ← calculate_crack_spread()       (Phase 2)
  vsa_agents.py        ← calculate_position_size()      (Phase 4)
  autonomous_agent.py  ← all_exchange_latencies()       (Phase 1)
  crew_agent.py        ← full context string            (Phase 1+2+4)
  global_ecosystem.py  ← exchange latency context       (Phase 1)
  main.py              ← --market-arch demo             (all phases)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

try:
    from scipy.stats import norm as _norm
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ============================================================================
# KNOWN EXCHANGE MICROWAVE ROUTES
# ============================================================================

EXCHANGE_ROUTES: Dict[str, Dict] = {
    "NYC_CHICAGO": {
        "description": "NYSE / NASDAQ → CME Globex (WTI CL, S&P 500 ES)",
        "distance_km": 1_180,
        "exchanges":   ["NYSE", "CME/NYMEX"],
    },
    "CHICAGO_LONDON": {
        "description": "CME Globex → ICE London (Brent BRN, TTF Gas)",
        "distance_km": 7_500,
        "exchanges":   ["CME/NYMEX", "ICE London"],
    },
    "NYC_HOUSTON": {
        "description": "NYSE → Physical WTI Cushing pipeline hub",
        "distance_km": 2_200,
        "exchanges":   ["NYSE", "Cushing Hub"],
    },
    "LONDON_FRANKFURT": {
        "description": "ICE London → Eurex (energy derivatives)",
        "distance_km":   650,
        "exchanges":   ["ICE", "Eurex"],
    },
}

# ============================================================================
# CONTRACT SPECIFICATIONS (CME / NYMEX — sourced from CLAUDE.md)
# ============================================================================

WTI_CL_SPEC: Dict = {
    "symbol":      "CL",
    "name":        "WTI Crude Oil",
    "multiplier":  1_000,   # bbl / contract
    "tick_size":   0.01,    # $/bbl
    "tick_value":  10.00,   # $/tick
    "currency":    "USD",
    "exchange":    "NYMEX",
}
MCL_SPEC: Dict = {
    "symbol":      "MCL",
    "name":        "Micro WTI Crude Oil",
    "multiplier":  100,     # bbl / contract
    "tick_size":   0.01,
    "tick_value":  1.00,
    "currency":    "USD",
    "exchange":    "NYMEX",
}
BRENT_SPEC: Dict = {
    "symbol":      "BRN",
    "name":        "Brent Crude Oil",
    "multiplier":  1_000,
    "tick_size":   0.01,
    "tick_value":  10.00,
    "currency":    "USD",
    "exchange":    "ICE",
}
RBOB_SPEC: Dict = {
    "symbol":      "RB",
    "name":        "RBOB Gasoline",
    "multiplier":  42_000,  # gallons / contract (1 000 bbl × 42 gal/bbl)
    "tick_size":   0.0001,  # $/gal
    "tick_value":  4.20,
    "currency":    "USD",
    "exchange":    "NYMEX",
}
ULSD_SPEC: Dict = {
    "symbol":      "HO",
    "name":        "ULSD Heating Oil",
    "multiplier":  42_000,
    "tick_size":   0.0001,
    "tick_value":  4.20,
    "currency":    "USD",
    "exchange":    "NYMEX",
}

ALL_CONTRACTS: Dict[str, Dict] = {
    "CL":  WTI_CL_SPEC,
    "MCL": MCL_SPEC,
    "BRN": BRENT_SPEC,
    "RB":  RBOB_SPEC,
    "HO":  ULSD_SPEC,
}


# ============================================================================
# MARKET ARCHITECTURE MATH ENGINE
# ============================================================================

class MarketArchitectureMath:
    """
    Four-phase quantitative market microstructure calculator.

    Stateless: all methods are pure functions — safe to use as a module
    singleton via get_market_arch().
    """

    def __init__(self) -> None:
        # Physical constants
        self.c       = 299_792   # Speed of light in vacuum (km/s)
        self.n_fiber = 1.5       # Refractive index of fiber-optic glass
        # Derived velocities
        self.v_fiber     = self.c / self.n_fiber    # ≈ 199 861 km/s
        self.v_microwave = self.c                   # ≈ 299 792 km/s (through air ≈ vacuum)

    # ── Phase 1 — Latency Engine ──────────────────────────────────────────────

    def calculate_latency_advantage(self, distance_km: float) -> Dict:
        """
        Microsecond edge of microwave over fiber-optic for a given route.

        Microwave travels at ≈c through air; fiber at c/n_fiber (n≈1.5) due to
        total internal reflection — microwave is ~33% faster over the same distance.

        HFT context: at NYC→CME (1 180 km), microwave saves ~2.62 μs one-way.
        A 1 μs advantage allows cancellation of ~1 000 limit orders before
        a competitor's stale quote arrives.
        """
        t_fiber     = (distance_km / self.v_fiber)     * 1e6   # microseconds
        t_microwave = (distance_km / self.v_microwave) * 1e6
        advantage   = t_fiber - t_microwave
        return {
            "distance_km":            round(distance_km, 1),
            "fiber_microseconds":     round(t_fiber,     2),
            "microwave_microseconds": round(t_microwave, 2),
            "advantage_microseconds": round(advantage,   2),
        }

    def all_exchange_latencies(self) -> Dict[str, Dict]:
        """Compute microwave vs fiber advantage for every entry in EXCHANGE_ROUTES."""
        result = {}
        for route, meta in EXCHANGE_ROUTES.items():
            lat = self.calculate_latency_advantage(meta["distance_km"])
            result[route] = {**meta, **lat}
        return result

    # ── Phase 2 — Contract Math ────────────────────────────────────────────────

    def calculate_notional_and_tick(self,
                                    price:      float,
                                    multiplier: int,
                                    tick_size:  float,
                                    contracts:  int = 1) -> Dict:
        """
        Total contract exposure and exact dollar-per-tick value.

        Example — WTI CL at $75.00, 1 contract:
          multiplier   = 1 000 bbl/contract
          notional     = $75 × 1 000 = $75 000
          tick_value   = $0.01 × 1 000 = $10.00/tick
        """
        notional   = price * multiplier * contracts
        tick_value = tick_size * multiplier
        return {
            "notional_value_usd":  round(notional,   2),
            "tick_value_usd":      round(tick_value,  4),
            "price":               price,
            "contracts":           contracts,
        }

    def contract_notional(self, symbol: str, price: float,
                          contracts: int = 1) -> Dict:
        """Convenience wrapper using the pre-defined ALL_CONTRACTS specs."""
        spec = ALL_CONTRACTS.get(symbol.upper(), MCL_SPEC)
        return self.calculate_notional_and_tick(
            price=price,
            multiplier=spec["multiplier"],
            tick_size=spec["tick_size"],
            contracts=contracts,
        )

    def calculate_crack_spread(self,
                               crude_price:        float,
                               gasoline_price:     float,
                               heating_oil_price:  float) -> float:
        """
        3:2:1 refinery profit margin per barrel of crude processed.

        Formula  (Oil Trader Academy / Trafigura):
          crack = (2 × gasoline_$/gal × 42 + 1 × heat_oil_$/gal × 42
                   − 3 × crude_$/bbl) ÷ 3

        Inputs:  crude in $/bbl; gasoline & heating oil in $/gallon.
        Output:  $/bbl refinery gross margin.
        """
        gas_value  = 2 * gasoline_price    * 42   # $/bbl equivalent
        oil_value  = 1 * heating_oil_price * 42
        crude_cost = 3 * crude_price
        return round((gas_value + oil_value - crude_cost) / 3, 4)

    def calculate_sp500_index(self, market_caps: List[float],
                              divisor: float) -> float:
        """Market-cap-weighted index level (price-weighted if market_caps = prices)."""
        return round(sum(market_caps) / divisor, 4)

    # ── Phase 3 — Options Pricing (Black-Scholes — EQUITY ONLY) ───────────────
    #
    # ⚠  DO NOT USE FOR ENERGY OPTIONS.
    #    WTI / Brent / RBOB / ULSD  →  use black76() from strategy_agent.py.
    #    The Black-Scholes PDE assumes a continuously-traded spot asset; energy
    #    options are written on futures and require the Black-76 forward-price
    #    convention: S·e^{(r-q)T} = F when q = r.
    # ──────────────────────────────────────────────────────────────────────────

    def black_scholes_call(self,
                           S:     float,
                           K:     float,
                           T:     float,
                           r:     float,
                           sigma: float) -> Dict:
        """
        European call price, Delta, and daily Theta via Black-Scholes PDE.

        Partial derivatives:
          ∂C/∂S  = N(d1)                               ← Delta
          ∂C/∂t  = −S·N'(d1)·σ/(2√T) − r·K·e^{-rT}·N(d2)  ← Theta (annual)

        Args:
            S:     underlying SPOT price (equity only)
            K:     strike price
            T:     time to expiry in years
            r:     risk-free rate (annual, continuously compounded)
            sigma: implied volatility (annual)

        ⚠  EQUITY USE ONLY. Energy options → black76() in strategy_agent.py.
        """
        if T <= 0:
            intrinsic = max(0.0, S - K)
            return {
                "call_price":        round(intrinsic, 4),
                "delta":             1.0 if S > K else 0.0,
                "theta_daily_decay": 0.0,
            }

        sq_T = math.sqrt(T)
        d1   = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sq_T)
        d2   = d1 - sigma * sq_T

        if _SCIPY:
            nd1,  nd2  = float(_norm.cdf(d1)),  float(_norm.cdf(d2))
            pfd1       = float(_norm.pdf(d1))
        else:
            nd1,  nd2  = _ncdf(d1), _ncdf(d2)
            pfd1       = math.exp(-d1**2 / 2) / math.sqrt(2 * math.pi)

        call_price   = S * nd1 - K * math.exp(-r * T) * nd2
        delta        = nd1
        theta_annual = (-(S * pfd1 * sigma) / (2 * sq_T)
                        - r * K * math.exp(-r * T) * nd2)
        theta_daily  = theta_annual / 365

        return {
            "call_price":        round(call_price,  4),
            "delta":             round(delta,       4),
            "theta_daily_decay": round(theta_daily, 4),
        }

    # ── Phase 4 — Risk Engine ─────────────────────────────────────────────────

    def calculate_position_size(self,
                                balance:    float,
                                risk_pct:   float,
                                stop_ticks: int,
                                tick_value: float) -> Dict:
        """
        Maximum contract count under the 1%-rule risk ceiling.

        Formula:
          max_contracts = floor(balance × risk_pct/100
                               / (stop_ticks × tick_value))

        Example — $500 account, 2% risk, 20-tick stop, MCL ($1.00/tick):
          max_loss          = $10.00
          risk_per_contract = 20 × $1.00 = $20.00
          max_contracts     = floor($10 / $20) = 0  → paper-only at $500

        Consistent with CLAUDE.md risk limits (MAX_RISK_PER_TRADE_PCT = 2%).
        """
        if stop_ticks <= 0 or tick_value <= 0:
            return {"max_loss_allowed_usd": 0.0,
                    "risk_per_contract_usd": 0.0,
                    "max_contracts": 0}
        max_loss          = balance * (risk_pct / 100)
        risk_per_contract = stop_ticks * tick_value
        max_contracts     = math.floor(max_loss / risk_per_contract)
        return {
            "max_loss_allowed_usd":  round(max_loss,          2),
            "risk_per_contract_usd": round(risk_per_contract,  2),
            "max_contracts":          max_contracts,
        }

    def calculate_parametric_var(self,
                                  portfolio_value: float,
                                  daily_mean:      float,
                                  daily_vol:       float,
                                  confidence:      int = 95,
                                  days:            int = 1) -> float:
        """
        Parametric (variance-covariance) Value at Risk.

        VaR = portfolio × [μ·T + z·σ·√T]   (positive = dollar loss)

        z-scores: 95% → 1.645 | 99% → 2.326

        Consistent with Hull, Risk Management & Financial Institutions Ch.12.
        """
        z_map = {95: 1.645, 99: 2.326}
        z     = z_map.get(confidence, 1.645)
        var   = portfolio_value * (daily_mean * days + z * daily_vol * math.sqrt(days))
        return round(var, 2)

    def run_demo(self) -> None:
        """Print a full 4-phase demo to stdout. Called by main.py --market-arch."""
        print("=" * 70)
        print("  MARKET ARCHITECTURE MATH ENGINE")
        print("  Phase 1 (Latency) · Phase 2 (Contract) · "
              "Phase 3 (BSM Equity) · Phase 4 (Risk)")
        print("=" * 70)

        print("\n── PHASE 1: LATENCY ENGINE ──────────────────────────────────────────")
        for route, meta in EXCHANGE_ROUTES.items():
            lat = self.calculate_latency_advantage(meta["distance_km"])
            print(f"  {route} ({meta['distance_km']} km)")
            print(f"    Fiber:     {lat['fiber_microseconds']:>7.2f} µs")
            print(f"    Microwave: {lat['microwave_microseconds']:>7.2f} µs")
            print(f"    Advantage: {lat['advantage_microseconds']:>7.2f} µs  ← HFT edge")

        print("\n── PHASE 2: CONTRACT MATH ───────────────────────────────────────────")
        for sym, spec in ALL_CONTRACTS.items():
            ct = self.calculate_notional_and_tick(
                price=75.0, multiplier=spec["multiplier"], tick_size=spec["tick_size"]
            )
            print(f"  {sym:4s} ({spec['name']})  notional=${ct['notional_value_usd']:>12,.2f}"
                  f"  tick=${ct['tick_value_usd']:.4f}")
        cs = self.calculate_crack_spread(75.0, 2.10, 2.50)
        print(f"\n  3:2:1 Crack spread @ WTI=$75 RBOB=$2.10 HO=$2.50 → ${cs:.4f}/bbl")

        print("\n── PHASE 3: BLACK-SCHOLES (EQUITY ONLY — NOT FOR ENERGY) ───────────")
        opt = self.black_scholes_call(S=75.0, K=75.0, T=30/365, r=0.04, sigma=0.35)
        print(f"  ATM call $75 / 30d / σ=35%:  Price=${opt['call_price']}  "
              f"Δ={opt['delta']}  Θ/day=${opt['theta_daily_decay']}")

        print("\n── PHASE 4: RISK ENGINE ─────────────────────────────────────────────")
        ct_mcl = self.calculate_notional_and_tick(
            price=75.0, multiplier=MCL_SPEC["multiplier"], tick_size=MCL_SPEC["tick_size"]
        )
        pos = self.calculate_position_size(
            balance=500, risk_pct=2.0, stop_ticks=20, tick_value=ct_mcl["tick_value_usd"]
        )
        print(f"  MCL $500 · 2% · 20-tick stop: {pos['max_contracts']} contracts "
              f"(loss cap=${pos['max_loss_allowed_usd']:.2f})")
        var95 = self.calculate_parametric_var(500, 0.0, 0.018, 95)
        var99 = self.calculate_parametric_var(500, 0.0, 0.018, 99)
        print(f"  1-day VaR $500 portfolio: 95%=${var95:.2f}  99%=${var99:.2f}")
        print("=" * 70)


# ============================================================================
# SCIPY FALLBACK
# ============================================================================

def _ncdf(x: float) -> float:
    """Standard normal CDF via math.erf (no scipy required)."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


# ============================================================================
# MODULE-LEVEL SINGLETON
# ============================================================================

_mam_instance: Optional[MarketArchitectureMath] = None


def get_market_arch() -> MarketArchitectureMath:
    """Lazily initialise and return the module-level MAM singleton."""
    global _mam_instance
    if _mam_instance is None:
        _mam_instance = MarketArchitectureMath()
    return _mam_instance


# ============================================================================
# STANDALONE DEMO
# ============================================================================

if __name__ == "__main__":
    import sys as _sys
    import io as _io
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8")

    mam = MarketArchitectureMath()

    print("=" * 70)
    print("  MARKET ARCHITECTURE MATH ENGINE")
    print("  Phase 1 (Latency) · Phase 2 (Contract) · "
          "Phase 3 (BSM) · Phase 4 (Risk)")
    print("=" * 70)

    print("\n── PHASE 1: LATENCY ENGINE ──────────────────────────────────────────")
    for route, meta in EXCHANGE_ROUTES.items():
        lat = mam.calculate_latency_advantage(meta["distance_km"])
        print(f"  {route} ({meta['distance_km']} km)")
        print(f"    Fiber:     {lat['fiber_microseconds']:>7.2f} μs")
        print(f"    Microwave: {lat['microwave_microseconds']:>7.2f} μs")
        print(f"    Advantage: {lat['advantage_microseconds']:>7.2f} μs  ← HFT edge")

    print("\n── PHASE 2: CONTRACT MATH ───────────────────────────────────────────")
    for sym, spec in ALL_CONTRACTS.items():
        ct = mam.calculate_notional_and_tick(
            price=75.0, multiplier=spec["multiplier"], tick_size=spec["tick_size"]
        )
        print(f"  {sym:4s} ({spec['name']})  notional=${ct['notional_value_usd']:>12,.2f}"
              f"  tick=${ct['tick_value_usd']:.4f}")
    cs = mam.calculate_crack_spread(crude_price=75.0,
                                    gasoline_price=2.10,
                                    heating_oil_price=2.50)
    print(f"\n  3:2:1 Crack spread @ WTI=$75  RBOB=$2.10  HO=$2.50 → ${cs:.4f}/bbl")

    print("\n── PHASE 3: BLACK-SCHOLES (EQUITY ONLY — NOT FOR ENERGY) ───────────")
    opt = mam.black_scholes_call(S=75.0, K=75.0, T=30/365, r=0.04, sigma=0.35)
    print(f"  ATM call $75 / 30d / σ=35%:")
    print(f"    Price: ${opt['call_price']}  Δ={opt['delta']}  "
          f"Θ/day=${opt['theta_daily_decay']}")

    print("\n── PHASE 4: RISK ENGINE ─────────────────────────────────────────────")
    ct_cl = mam.calculate_notional_and_tick(price=75.0,
                                             multiplier=WTI_CL_SPEC["multiplier"],
                                             tick_size=WTI_CL_SPEC["tick_size"])
    pos = mam.calculate_position_size(balance=50_000, risk_pct=1.0,
                                       stop_ticks=80,
                                       tick_value=ct_cl["tick_value_usd"])
    print(f"  CL  $50k · 1% · 80-tick stop: {pos['max_contracts']} contracts  "
          f"(loss cap=${pos['max_loss_allowed_usd']:.0f})")

    ct_mcl = mam.calculate_notional_and_tick(price=75.0,
                                              multiplier=MCL_SPEC["multiplier"],
                                              tick_size=MCL_SPEC["tick_size"])
    pos_m = mam.calculate_position_size(balance=500, risk_pct=2.0,
                                         stop_ticks=20,
                                         tick_value=ct_mcl["tick_value_usd"])
    print(f"  MCL $500  · 2% · 20-tick stop: {pos_m['max_contracts']} contracts  "
          f"(loss cap=${pos_m['max_loss_allowed_usd']:.2f})")

    var95 = mam.calculate_parametric_var(portfolio_value=500, daily_mean=0.0,
                                          daily_vol=0.018, confidence=95)
    var99 = mam.calculate_parametric_var(portfolio_value=500, daily_mean=0.0,
                                          daily_vol=0.018, confidence=99)
    print(f"  1-day VaR $500 portfolio: 95%=${var95:.2f}  99%=${var99:.2f}")
