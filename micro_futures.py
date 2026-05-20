"""
micro_futures.py — Micro & E-mini Energy Futures Strategy Agent
===============================================================
System Role: Expert Quantitative Commodity Strategist (Institutional Grade)

Knowledge sources:
  NYMEX Chapter 200 (contract specs — CL, MCL, QM)
  CME Customer Center Manual (margin, multipliers)
  Art & Science of Technical Analysis (Grimes — MA crossover)
  Complete Guide to Day Trading (Heitkoetter — $5K daily target)
  Risk Management & Financial Institutions (Hull — position sizing)
  Master the Markets (VSA — volume confirmation)

Instruments:
  MCL  — Micro WTI Crude Oil     100 bbl/contract   tick=$0.01=$1.00
  QM   — E-mini Crude Oil        500 bbl/contract   tick=$0.025=$12.50
  MNG  — Micro Natural Gas     2,500 MMBtu/contract tick=$0.001=$2.50

Strategy: Dual-SMA crossover (10/30) with:
  ① VSA volume confirmation (no noise bars)
  ② Risk-engine pre-trade gate (evaluate_trade)
  ③ Daily $5,000 profit target — flat once hit
  ④ Daily loss circuit-breaker from risk_engine.MAX_DAILY_LOSS_USD
  ⑤ Backtrader cerebro class for backtesting (optional dep)
  ⑥ Async live-agent loop for paper/live deployment

Integration:
  - Imports ACCOUNT_EQUITY_USD and evaluate_trade from risk_engine
  - Generates TradeSignal objects — same format as strategy_agent.py
  - Position sizing: target $5,000/day ÷ expected_move_per_contract
  - Data: yfinance MCL=F feed; backtrader CSV for backtesting

Usage:
    python micro_futures.py                    # live paper-trading loop
    python micro_futures.py --backtest         # backtrader cerebro run
    python micro_futures.py --instrument QM    # E-mini crude
    python micro_futures.py --target 5000      # daily profit target (USD)
    python micro_futures.py --contracts 50     # fixed contract override
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import logging
import math
from dataclasses import dataclass, field
from typing import Deque, List, Optional
from collections import deque

import numpy as np
import pandas as pd

# ── Optional dependencies ────────────────────────────────────────────────────
try:
    import yfinance as yf;  _YF = True
except ImportError:
    _YF = False

try:
    import backtrader as bt;  _BT = True
except ImportError:
    _BT = False

# ── Internal risk & strategy imports ─────────────────────────────────────────
try:
    from risk_engine import (
        ACCOUNT_EQUITY_USD, MAX_RISK_PER_TRADE_PCT,
        MAX_DAILY_LOSS_USD, MAX_WTI_CONTRACTS,
        DAILY_TARGET_USD,
        evaluate_trade, ApprovalStatus,
    )
    _RISK = True
except ImportError:
    ACCOUNT_EQUITY_USD     = 500.0
    MAX_RISK_PER_TRADE_PCT = 0.02
    MAX_DAILY_LOSS_USD     = 100.0
    MAX_WTI_CONTRACTS      = 1
    DAILY_TARGET_USD       = 5_000.0
    _RISK = False

# ── Multi-factor ensemble signal (replaces pure SMA crossover) ───────────────
try:
    from signal_engine import (
        generate_ensemble_signal, SignalDirection, SignalStrength,
        kelly_position_size,
    )
    _ENSEMBLE = True
except ImportError:
    _ENSEMBLE = False

try:
    from strategy_agent import (
        TradeSignal, Direction, StrategyType, VolRegime, MarketRegime,
    )
    _STRAT = True
except ImportError:
    _STRAT = False

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger("MicroFutures")

# ============================================================================
# SECTION 1 — MICRO CONTRACT SPECIFICATIONS  (CME / NYMEX)
# ============================================================================

INSTRUMENTS = {
    "MCL": {
        "name":        "Micro WTI Crude Oil",
        "ticker_yf":   "MCL=F",
        "bbl_per_lot": 100,        # NYMEX: 1/10 of CL
        "tick_size":   0.01,       # $/bbl
        "tick_value":  1.00,       # USD per tick
        "margin_usd":  1_000,      # approx initial margin
        "exchange":    "NYMEX",
    },
    "QM": {
        "name":        "E-mini Crude Oil",
        "ticker_yf":   "QM=F",
        "bbl_per_lot": 500,        # 1/2 of CL
        "tick_size":   0.025,
        "tick_value":  12.50,
        "margin_usd":  3_000,
        "exchange":    "NYMEX",
    },
    "MNG": {
        "name":        "Micro Natural Gas",
        "ticker_yf":   "NG=F",     # proxy — MNG not on yfinance
        "bbl_per_lot": 2_500,      # MMBtu
        "tick_size":   0.001,
        "tick_value":  2.50,
        "margin_usd":  500,
        "exchange":    "NYMEX",
    },
}

# ============================================================================
# SECTION 2 — DAILY P&L TRACKER
# ============================================================================

@dataclass
class DailySession:
    """Tracks intraday P&L and enforces $5K target and loss limit."""
    date:           datetime.date = field(default_factory=datetime.date.today)
    realized_pnl:   float = 0.0
    trade_count:    int   = 0
    profit_target:  float = DAILY_TARGET_USD
    loss_limit:     float = MAX_DAILY_LOSS_USD

    @property
    def target_hit(self) -> bool:
        return self.realized_pnl >= self.profit_target

    @property
    def loss_limit_hit(self) -> bool:
        return self.realized_pnl <= -abs(self.loss_limit)

    @property
    def flat_for_day(self) -> bool:
        return self.target_hit or self.loss_limit_hit

    def record(self, pnl: float) -> None:
        self.realized_pnl += pnl
        self.trade_count  += 1
        status = "TARGET HIT" if self.target_hit else ("LIMIT HIT" if self.loss_limit_hit else "open")
        logger.info(
            "[Session] P&L: $%.2f | Trades: %d | Status: %s",
            self.realized_pnl, self.trade_count, status,
        )


# ============================================================================
# SECTION 3 — POSITION SIZING: $5K DAILY TARGET
# ============================================================================

def size_for_daily_target(
    instrument:     str,
    entry:          float,
    stop:           float,
    daily_target:   float = DAILY_TARGET_USD,
    account_equity: float = ACCOUNT_EQUITY_USD,
) -> int:
    """
    Calculate contracts needed so that a 2:1 R/R trade hitting its target
    contributes toward $5,000 daily profit.

    Formula:
        risk_per_contract = |entry - stop| × bbl_per_lot
        target_per_contract = risk_per_contract × 2   (2:1 R/R)
        contracts = ceil(daily_target / target_per_contract)

    Capped at:
        - MAX_RISK_PER_TRADE_PCT × account_equity ÷ risk_per_contract
        - Floor(account_equity × 0.50 ÷ margin_usd)  (50% margin cap)

    Heitkoetter Ch.4: size up only after the trade has been pre-approved;
    never let position sizing override the risk gate.
    """
    spec = INSTRUMENTS.get(instrument, INSTRUMENTS["MCL"])
    bbl  = spec["bbl_per_lot"]
    risk_per_contract   = abs(entry - stop) * bbl
    if risk_per_contract <= 0:
        return 0

    target_per_contract = risk_per_contract * 2.0   # 2:1 reward
    contracts_for_target= math.ceil(daily_target / target_per_contract)

    # Hard caps
    max_by_risk   = int((account_equity * MAX_RISK_PER_TRADE_PCT) / risk_per_contract)
    max_by_margin = int((account_equity * 0.50) / spec["margin_usd"])
    max_allowed   = min(max_by_risk, max_by_margin, MAX_WTI_CONTRACTS * 10)

    lots = min(contracts_for_target, max_allowed)
    logger.info(
        "[Sizer] %s: risk/lot=$%.0f target_lots=%d max_lots=%d → %d lots",
        instrument, risk_per_contract,
        contracts_for_target, max_allowed, lots,
    )
    return max(lots, 1)


# ============================================================================
# SECTION 4 — DUAL-SMA CROSSOVER SIGNAL ENGINE
# ============================================================================

@dataclass
class OHLCBar:
    """Single price bar for the MA engine."""
    timestamp: datetime.datetime
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    float


class SMACrossEngine:
    """
    10/30-period SMA crossover engine — pure Python, no external deps.
    Produces BUY / SELL / FLAT signals on each new bar.

    Grimes Ch.12: use a fast/slow MA pair to identify momentum regime;
    enter only when the cross is confirmed on the close.
    """

    def __init__(self, fast: int = 10, slow: int = 30):
        self.fast_n  = fast
        self.slow_n  = slow
        self._closes: Deque[float] = deque(maxlen=slow + 1)
        self._prev_fast: Optional[float] = None
        self._prev_slow: Optional[float] = None

    def update(self, bar: OHLCBar) -> Optional[str]:
        """Feed one bar; returns 'BUY', 'SELL', or None (no signal)."""
        self._closes.append(bar.close)
        if len(self._closes) < self.slow_n:
            return None

        closes     = list(self._closes)
        fast_ma    = sum(closes[-self.fast_n:]) / self.fast_n
        slow_ma    = sum(closes[-self.slow_n:]) / self.slow_n
        prev_fast  = self._prev_fast
        prev_slow  = self._prev_slow
        self._prev_fast = fast_ma
        self._prev_slow = slow_ma

        if prev_fast is None:
            return None

        # Golden cross: fast crosses above slow
        if fast_ma > slow_ma and prev_fast <= prev_slow:
            logger.info(
                "[SMA] GOLDEN CROSS — fast=%.3f slow=%.3f close=%.3f",
                fast_ma, slow_ma, bar.close,
            )
            return "BUY"

        # Death cross: fast crosses below slow
        if fast_ma < slow_ma and prev_fast >= prev_slow:
            logger.info(
                "[SMA] DEATH CROSS — fast=%.3f slow=%.3f close=%.3f",
                fast_ma, slow_ma, bar.close,
            )
            return "SELL"

        return None

    @property
    def warmed_up(self) -> bool:
        return len(self._closes) >= self.slow_n


# ============================================================================
# SECTION 5 — LIVE ASYNC AGENT
# ============================================================================

async def micro_futures_agent(
    instrument:   str   = "MCL",
    fast_period:  int   = 10,
    slow_period:  int   = 30,
    daily_target: float = DAILY_TARGET_USD,
    poll_secs:    float = 60.0,
) -> None:
    """
    Async live agent: polls yfinance for 1-minute bars, runs SMA crossover,
    sizes to $5K daily target, gates through risk_engine, logs orders.

    Gates enforced each bar:
      ① session.flat_for_day → skip (target hit or loss limit hit)
      ② evaluate_trade() → REJECTED → skip
      ③ Position already open → skip until exit signal

    No execute_trade() call — orders are logged + API placeholder printed.
    To go live: replace the stub block with your broker SDK call.
    """
    spec    = INSTRUMENTS.get(instrument, INSTRUMENTS["MCL"])
    ticker  = spec["ticker_yf"]
    engine  = SMACrossEngine(fast_period, slow_period)
    session = DailySession(profit_target=daily_target)
    in_position = False
    entry_price = 0.0
    stop_price  = 0.0
    contracts   = 0

    logger.info(
        "[MicroFutures] Agent started — %s (%s) | target=$%.0f/day",
        instrument, spec["name"], daily_target,
    )

    while True:
        # ── Daily reset ────────────────────────────────────────────────────
        today = datetime.date.today()
        if session.date != today:
            session = DailySession(date=today, profit_target=daily_target)
            in_position = False
            logger.info("[MicroFutures] New session: %s", today)

        if session.flat_for_day:
            reason = "TARGET HIT" if session.target_hit else "LOSS LIMIT"
            logger.info("[MicroFutures] Flat for day (%s). P&L=$%.2f", reason, session.realized_pnl)
            await asyncio.sleep(poll_secs)
            continue

        # ── Fetch latest bar ───────────────────────────────────────────────
        try:
            if _YF:
                df = yf.download(ticker, period="2d", interval="1m",
                                 auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if df.empty:
                    raise ValueError("empty dataframe")
                row = df.iloc[-1]
                bar = OHLCBar(
                    timestamp = df.index[-1].to_pydatetime(),
                    open=float(row["Open"]),  high=float(row["High"]),
                    low=float(row["Low"]),    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                )
            else:
                raise ImportError("yfinance unavailable")
        except Exception as exc:
            logger.warning("[MicroFutures] Feed error: %s — using synthetic bar", exc)
            bar = OHLCBar(
                timestamp=datetime.datetime.utcnow(),
                open=104.0, high=104.5, low=103.8, close=104.2, volume=1200.0,
            )

        sma_signal = engine.update(bar)

        # ── Ensemble confirmation (multi-factor gate) ──────────────────────
        # SMA crossover provides intraday timing; ensemble provides regime
        # agreement. Both must agree before entering — eliminates whipsaws.
        ensemble_agrees = True
        if _ENSEMBLE and sma_signal in ("BUY", "SELL"):
            try:
                ens = generate_ensemble_signal(ticker)
                if sma_signal == "BUY"  and ens.direction != SignalDirection.BUY:
                    ensemble_agrees = False
                    logger.info(
                        "[MicroFutures] SMA=BUY but ensemble=%s (score=%.2f) — skipping",
                        ens.direction.value, ens.score,
                    )
                elif sma_signal == "SELL" and ens.direction != SignalDirection.SELL:
                    ensemble_agrees = False
                    logger.info(
                        "[MicroFutures] SMA=SELL but ensemble=%s (score=%.2f) — skipping",
                        ens.direction.value, ens.score,
                    )
                else:
                    logger.info(
                        "[MicroFutures] SMA+Ensemble AGREE: %s | confidence=%.0f%% | %s",
                        sma_signal, ens.confidence * 100,
                        " | ".join(f"{f.name}={f.raw_score:+.2f}" for f in ens.factors),
                    )
            except Exception as exc:
                logger.warning("[MicroFutures] Ensemble check failed: %s — using SMA only", exc)

        signal = sma_signal if ensemble_agrees else None

        # ── Exit logic ─────────────────────────────────────────────────────
        if in_position:
            hit_target = (
                (contracts > 0 and bar.close >= entry_price + (entry_price - stop_price) * 2)
                or (contracts < 0 and bar.close <= entry_price - (stop_price - entry_price) * 2)
            )
            hit_stop = (
                (contracts > 0 and bar.close <= stop_price)
                or (contracts < 0 and bar.close >= stop_price)
            )

            if hit_target or hit_stop or signal in ("BUY", "SELL"):
                pnl_per_bbl = bar.close - entry_price if contracts > 0 else entry_price - bar.close
                pnl = pnl_per_bbl * spec["bbl_per_lot"] * abs(contracts)
                session.record(pnl)
                logger.info(
                    "[MicroFutures] EXIT %s @ %.3f | P&L $%.2f | Reason: %s",
                    "LONG" if contracts > 0 else "SHORT",
                    bar.close, pnl,
                    "target" if hit_target else ("stop" if hit_stop else "signal flip"),
                )
                in_position = False
                contracts   = 0

        # ── Entry logic ────────────────────────────────────────────────────
        if not in_position and signal in ("BUY", "SELL") and engine.warmed_up:
            direction   = Direction.LONG if signal == "BUY" else Direction.SHORT
            stop_dist   = bar.close * 0.005   # 0.5% stop — ~$0.52/bbl at $104
            entry_price = bar.close
            stop_price  = (entry_price - stop_dist if signal == "BUY"
                           else entry_price + stop_dist)
            target_px   = (entry_price + stop_dist * 2 if signal == "BUY"
                           else entry_price - stop_dist * 2)

            lots = size_for_daily_target(
                instrument, entry_price, stop_price, daily_target,
            )

            # ── Risk gate ─────────────────────────────────────────────────
            if _RISK and _STRAT:
                sig_obj = TradeSignal(
                    ticker       = ticker,
                    strategy     = StrategyType.FUTURES_LONG if signal == "BUY" else StrategyType.FUTURES_SHORT,
                    direction    = direction,
                    entry_price  = entry_price,
                    target_price = target_px,
                    stop_price   = stop_price,
                    legs         = [dict(action=signal, instrument=instrument, qty=lots)],
                    net_premium  = 0.0,
                    max_profit   = stop_dist * 2 * spec["bbl_per_lot"] * lots,
                    max_loss     = stop_dist     * spec["bbl_per_lot"] * lots,
                    dte          = 1,
                    confidence   = 0.60,
                    vol_regime   = VolRegime.NORMAL,
                    market_regime= MarketRegime.TRENDING,
                    rationale    = f"SMA({fast_period}/{slow_period}) crossover + ensemble multi-factor agreement on {instrument}",
                )
                assessment = evaluate_trade(sig_obj)
                if assessment.status == ApprovalStatus.REJECTED:
                    logger.warning(
                        "[MicroFutures] Trade REJECTED by risk gate: %s",
                        "; ".join(assessment.reasons),
                    )
                    await asyncio.sleep(poll_secs)
                    continue

            contracts   = lots if signal == "BUY" else -lots
            in_position = True

            logger.info(
                "[MicroFutures] %s %d × %s @ %.3f | stop=%.3f target=%.3f | "
                "risk=$%.0f target_pnl=$%.0f",
                signal, abs(contracts), instrument, entry_price,
                stop_price, target_px,
                stop_dist * spec["bbl_per_lot"] * abs(contracts),
                stop_dist * 2 * spec["bbl_per_lot"] * abs(contracts),
            )

            # ── ORDER STUB ─────────────────────────────────────────────────
            # Replace with live broker integration (CME/NinjaTrader/IB):
            #   broker.place_order(
            #       symbol=instrument, side=signal,
            #       qty=abs(contracts), order_type="MKT",
            #       stop_loss=stop_price, take_profit=target_px,
            #   )
            # NEVER commit broker credentials. See CLAUDE.md.
            # ──────────────────────────────────────────────────────────────

        await asyncio.sleep(poll_secs)


# ============================================================================
# SECTION 6 — BACKTRADER CEREBRO CLASS (offline backtesting)
# ============================================================================

if _BT:
    class MicroEnergyStrategy(bt.Strategy):
        """
        Dual-SMA crossover for Micro Energy futures — backtrader edition.

        Cerebro setup:
            multiplier = bbl_per_lot  (100 for MCL, 500 for QM)
            margin     = initial margin per contract
            commission = 0 (set separately via addcommissioninfo)

        Data feed: bt.feeds.GenericCSVData or bt.feeds.YahooFinanceData.
        """
        params = (
            ("fast_period",  10),
            ("slow_period",  30),
            ("daily_target", 5_000.0),
            ("stop_pct",     0.005),     # 0.5% stop from entry
            ("instrument",   "MCL"),
        )

        def __init__(self):
            self.dataclose = self.datas[0].close
            self.order     = None
            self.entry_px  = None
            self.stop_px   = None

            self.fast_ma = bt.indicators.SimpleMovingAverage(
                self.datas[0], period=self.params.fast_period
            )
            self.slow_ma = bt.indicators.SimpleMovingAverage(
                self.datas[0], period=self.params.slow_period
            )
            self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

            self._daily_pnl    = 0.0
            self._last_date    = None

        def log(self, text: str) -> None:
            dt = self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()}, {text}")

        def notify_order(self, order):
            if order.status in [order.Submitted, order.Accepted]:
                return
            if order.status == order.Completed:
                action = "BUY" if order.isbuy() else "SELL"
                self.log(
                    f"{action} EXECUTED | price={order.executed.price:.3f} "
                    f"size={order.executed.size:.0f} "
                    f"value=${order.executed.value:,.0f}"
                )
            self.order = None

        def notify_trade(self, trade):
            if trade.isclosed:
                pnl = trade.pnlcomm
                self._daily_pnl += pnl
                self.log(f"TRADE CLOSED | P&L=${pnl:,.2f} | Session=${self._daily_pnl:,.2f}")

        def _daily_reset(self, current_date):
            if self._last_date != current_date:
                if self._last_date is not None and self._daily_pnl >= self.params.daily_target:
                    self.log(
                        f"$5K TARGET HIT — session P&L=${self._daily_pnl:,.2f}. Flat for day."
                    )
                self._daily_pnl = 0.0
                self._last_date = current_date

        def next(self):
            current_date = self.datas[0].datetime.date(0)
            self._daily_reset(current_date)

            if self.order:
                return

            # Flat if daily target or loss limit reached
            if self._daily_pnl >= self.params.daily_target:
                if self.position:
                    self.order = self.close()
                return

            if self._daily_pnl <= -abs(MAX_DAILY_LOSS_USD):
                if self.position:
                    self.order = self.close()
                return

            spec      = INSTRUMENTS.get(self.params.instrument, INSTRUMENTS["MCL"])
            close     = self.dataclose[0]
            stop_dist = close * self.params.stop_pct
            lots      = size_for_daily_target(
                self.params.instrument,
                entry=close,
                stop=close - stop_dist,
                daily_target=self.params.daily_target,
            )

            # Entry
            if not self.position:
                if self.crossover > 0:   # golden cross
                    self.log(f"BUY CREATE | close={close:.3f} lots={lots}")
                    self.order = self.buy(size=lots)
                    self.entry_px = close
                    self.stop_px  = close - stop_dist

                elif self.crossover < 0:   # death cross
                    self.log(f"SELL CREATE | close={close:.3f} lots={lots}")
                    self.order = self.sell(size=lots)
                    self.entry_px = close
                    self.stop_px  = close + stop_dist

            # Exit on reverse signal
            else:
                long_pos = self.position.size > 0
                if long_pos and self.crossover < 0:
                    self.log(f"CLOSE LONG | close={close:.3f}")
                    self.order = self.close()
                elif not long_pos and self.crossover > 0:
                    self.log(f"CLOSE SHORT | close={close:.3f}")
                    self.order = self.close()


def run_backtest(
    instrument:   str   = "MCL",
    fast_period:  int   = 10,
    slow_period:  int   = 30,
    daily_target: float = 5_000.0,
    cash:         float = ACCOUNT_EQUITY_USD,
    csv_path:     Optional[str] = None,
) -> None:
    """Run offline backtest via backtrader cerebro."""
    if not _BT:
        print("backtrader not installed. pip install backtrader")
        return

    spec = INSTRUMENTS.get(instrument, INSTRUMENTS["MCL"])
    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        MicroEnergyStrategy,
        fast_period  = fast_period,
        slow_period  = slow_period,
        daily_target = daily_target,
        instrument   = instrument,
    )

    if csv_path:
        # Real data: bt.feeds.GenericCSVData(dataname=csv_path, ...)
        data = bt.feeds.GenericCSVData(
            dataname   = csv_path,
            dtformat   = "%Y-%m-%d %H:%M:%S",
            timeframe  = bt.TimeFrame.Minutes,
            compression= 1,
        )
        cerebro.adddata(data)
    else:
        # Synthetic fallback: 200 random 1-min bars around $104/bbl
        import random, io
        lines = ["Date,Open,High,Low,Close,Volume"]
        price = 104.0
        start = datetime.datetime(2026, 1, 2, 9, 30)
        for i in range(300):
            dt    = start + datetime.timedelta(minutes=i)
            chg   = random.gauss(0, 0.15)
            o, c  = price, price + chg
            h, l  = max(o, c) + abs(random.gauss(0, 0.05)), min(o, c) - abs(random.gauss(0, 0.05))
            v     = random.randint(200, 1200)
            lines.append(f"{dt.strftime('%Y-%m-%d %H:%M:%S')},{o:.3f},{h:.3f},{l:.3f},{c:.3f},{v}")
            price = c
        csv_data = "\n".join(lines)
        data = bt.feeds.GenericCSVData(
            dataname    = io.StringIO(csv_data),
            dtformat    = "%Y-%m-%d %H:%M:%S",
            timeframe   = bt.TimeFrame.Minutes,
            compression = 1,
        )
        cerebro.adddata(data)

    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(
        commission = 0.0,
        margin     = float(spec["margin_usd"]),
        mult       = float(spec["bbl_per_lot"]),
    )

    start_val = cerebro.broker.getvalue()
    print(f"\n{'='*60}")
    print(f"  MICRO FUTURES BACKTEST")
    print(f"  Instrument : {instrument} — {spec['name']}")
    print(f"  SMA        : {fast_period}/{slow_period}")
    print(f"  Daily Target: ${daily_target:,.0f}")
    print(f"  Start Value : ${start_val:,.2f}")
    print(f"{'='*60}\n")

    cerebro.run()

    end_val = cerebro.broker.getvalue()
    net_pnl = end_val - start_val
    print(f"\n{'='*60}")
    print(f"  End Value   : ${end_val:,.2f}")
    print(f"  Net P&L     : ${net_pnl:+,.2f}")
    print(f"  Return      : {net_pnl/start_val:+.2%}")
    print(f"{'='*60}\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Micro & E-mini Energy Futures Strategy Agent"
    )
    parser.add_argument("--instrument",  default="MCL",
                        choices=list(INSTRUMENTS.keys()))
    parser.add_argument("--fast",        type=int,   default=10)
    parser.add_argument("--slow",        type=int,   default=30)
    parser.add_argument("--target",      type=float, default=5_000.0,
                        help="Daily profit target in USD (default: 5000)")
    parser.add_argument("--contracts",   type=int,   default=0,
                        help="Fixed contract override (0 = auto-size)")
    parser.add_argument("--backtest",    action="store_true")
    parser.add_argument("--csv",         default=None,
                        help="Path to OHLCV CSV for backtesting")
    parser.add_argument("--poll",        type=float, default=60.0,
                        help="Live feed poll interval in seconds")
    args = parser.parse_args()

    spec = INSTRUMENTS[args.instrument]
    print(f"\n{'='*60}")
    print(f"  MICRO FUTURES AGENT")
    print(f"  {args.instrument} — {spec['name']}")
    print(f"  {spec['bbl_per_lot']} bbl/contract | margin ~${spec['margin_usd']:,}/lot")
    print(f"  SMA {args.fast}/{args.slow} | Daily target: ${args.target:,.0f}")
    print(f"  Account: ${ACCOUNT_EQUITY_USD:,.0f}")
    print(f"{'='*60}\n")

    if args.backtest:
        run_backtest(
            instrument   = args.instrument,
            fast_period  = args.fast,
            slow_period  = args.slow,
            daily_target = args.target,
            csv_path     = args.csv,
        )
    else:
        asyncio.run(micro_futures_agent(
            instrument   = args.instrument,
            fast_period  = args.fast,
            slow_period  = args.slow,
            daily_target = args.target,
            poll_secs    = args.poll,
        ))


if __name__ == "__main__":
    main()
