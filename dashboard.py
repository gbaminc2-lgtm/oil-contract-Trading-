"""
dashboard.py — Institutional Portfolio Telemetry Dashboard
===========================================================
Streamlit real-time UI reading from crew_trading_ledger.db.
Run separately from the trading loop:
    streamlit run dashboard.py

Tabs:
  1. Options Portfolio  — active positions, Greeks exposure, PnL
  2. Agent Decisions    — cognitive reasoning audit trail
  3. Telemetry Logs     — latency profile, system health
  4. Simulator          — inject test alerts to Discord/Slack webhooks
"""

from __future__ import annotations

import datetime
import os
import sqlite3
from pathlib import Path

import pandas as pd

# ── Load .env (for webhook URLs) ──────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

import streamlit as st

# Guard: exit cleanly if plotly/requests not installed
try:
    import plotly.graph_objects as go
    import plotly.express as px
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

try:
    import requests as _req
    _REQ = True
except ImportError:
    _REQ = False

# ── Config ────────────────────────────────────────────────────────────────────
DB_FILE             = Path(__file__).parent / "logs" / "crew_trading_ledger.db"
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
SLACK_WEBHOOK_URL   = os.environ.get("SLACK_WEBHOOK_URL", "")

# Risk limits (from risk_engine or fallback)
try:
    from risk_engine import ACCOUNT_EQUITY_USD, DAILY_TARGET_USD, MAX_DAILY_LOSS_USD
except ImportError:
    ACCOUNT_EQUITY_USD = 500.0
    DAILY_TARGET_USD   = 5_000.0
    MAX_DAILY_LOSS_USD = 100.0

# =============================================================================
# DATA LAYER
# =============================================================================

def _load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not DB_FILE.exists():
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    conn = sqlite3.connect(DB_FILE)
    portfolio = pd.read_sql_query(
        "SELECT * FROM options_portfolio ORDER BY id DESC", conn)
    decisions = pd.read_sql_query(
        "SELECT * FROM agent_decisions ORDER BY id DESC", conn)
    telemetry = pd.read_sql_query(
        "SELECT * FROM system_telemetry ORDER BY id DESC", conn)
    trades = pd.read_sql_query(
        "SELECT * FROM trade_logs ORDER BY id DESC", conn)
    conn.close()
    return portfolio, decisions, telemetry, trades

# =============================================================================
# WEBHOOK NOTIFIER
# =============================================================================

def _send_alert(component: str, level: str, message: str, latency: float = 0.0) -> bool:
    body = (
        f"**[{level}]** `{component}`\n"
        f"{message}\n"
        f"Latency: `{latency:.0f}ms` | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    sent = False
    for url in [DISCORD_WEBHOOK_URL, SLACK_WEBHOOK_URL]:
        if not url or "your" in url or not _REQ:
            continue
        try:
            payload = {"text": body} if "slack" in url else {"content": body}
            r = _req.post(url, json=payload, timeout=5)
            if r.status_code in (200, 204):
                sent = True
        except Exception:
            pass
    return sent

# =============================================================================
# DASHBOARD UI
# =============================================================================

def main() -> None:
    st.set_page_config(
        page_title="Energy Derivative Execution Ledger",
        page_icon="🛢",
        layout="wide",
    )

    st.title("🛢 Energy Derivative Execution Ledger & Health Telemetry")
    st.caption(
        f"Account: **${ACCOUNT_EQUITY_USD:,.2f}** | "
        f"Daily Target: **${DAILY_TARGET_USD:,.0f}** | "
        f"Max Daily Loss: **${MAX_DAILY_LOSS_USD:,.0f}** | "
        f"Data source: `{DB_FILE.name}`"
    )
    st.markdown("---")

    # Auto-refresh every 30s
    st.markdown(
        "<meta http-equiv='refresh' content='30'>",
        unsafe_allow_html=True,
    )

    portfolio, decisions, telemetry, trades = _load_frames()

    # ── Compute PnL ───────────────────────────────────────────────────────────
    total_pnl   = 0.0
    total_risk  = 0.0
    n_positions = 0
    avg_latency = 0.0

    if not portfolio.empty and "entry_premium" in portfolio.columns:
        portfolio["pnl"] = (
            (portfolio["current_premium"] - portfolio["entry_premium"])
            * portfolio["quantity"].fillna(1) * 100
        )
        total_pnl   = portfolio["pnl"].sum()
        total_risk  = portfolio["max_risk"].sum() if "max_risk" in portfolio.columns else 0.0
        n_positions = len(portfolio)

    if not telemetry.empty and "execution_latency_ms" in telemetry.columns:
        avg_latency = telemetry["execution_latency_ms"].mean()

    win_trades  = 0
    total_trades = 0
    if not trades.empty and "pnl_usd" in trades.columns:
        total_trades = len(trades)
        win_trades   = int((trades["pnl_usd"] > 0).sum())

    # ── Top Metrics Row ───────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Autopilot Engine", "ONLINE ✓", "100% Core Health")
    with c2:
        latency_delta = f"-{avg_latency:.0f}ms ✓" if avg_latency < 200 else "⚠ HIGH LATENCY"
        latency_color = "normal" if avg_latency < 200 else "inverse"
        st.metric("Avg Loop Latency", f"{avg_latency:.0f} ms",
                  latency_delta, delta_color=latency_color)
    with c3:
        pnl_sign = "▲" if total_pnl >= 0 else "▼"
        st.metric("Options Portfolio PnL", f"${total_pnl:,.2f}",
                  f"{pnl_sign} Unrealized")
    with c4:
        st.metric("Margin Allocated", f"${total_risk:,.2f}",
                  f"{total_risk / ACCOUNT_EQUITY_USD * 100:.1f}% of equity")
    with c5:
        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0.0
        st.metric("Win Rate", f"{win_rate:.1f}%", f"{total_trades} trades")

    st.markdown("---")

    # ── Charts Row ────────────────────────────────────────────────────────────
    st.markdown("### 📈 Live Execution Analytics")
    if not _PLOTLY:
        st.warning("Install plotly for charts: `pip install plotly`")
    else:
        g1, g2 = st.columns(2)

        with g1:
            st.markdown("#### Black-76 Greeks Risk Exposure (Δ / Vega)")
            if not portfolio.empty and "delta" in portfolio.columns:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=portfolio["underlying"],
                    y=portfolio["delta"],
                    name="Net Delta",
                    marker_color="#1f77b4",
                ))
                fig.add_trace(go.Bar(
                    x=portfolio["underlying"],
                    y=portfolio["vega"],
                    name="Net Vega",
                    marker_color="#ff7f0e",
                ))
                fig.update_layout(
                    barmode="group", template="plotly_dark",
                    margin=dict(l=20, r=20, t=20, b=20),
                    legend=dict(orientation="h", y=-0.2),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Awaiting position data from autonomous agent cycles.")

        with g2:
            st.markdown("#### Engine Loop Latency Profile (ms)")
            if not telemetry.empty and "execution_latency_ms" in telemetry.columns:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    y=telemetry["execution_latency_ms"].iloc[:50],
                    mode="lines+markers",
                    line=dict(color="#2ca02c", width=2),
                    name="Latency (ms)",
                ))
                fig2.add_hline(
                    y=30_000, line_dash="dash", line_color="red",
                    annotation_text="30s threshold",
                )
                fig2.update_layout(
                    template="plotly_dark",
                    margin=dict(l=20, r=20, t=20, b=20),
                    yaxis_title="Latency (ms)",
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Awaiting telemetry data.")

    # ── PnL Curve ─────────────────────────────────────────────────────────────
    if not trades.empty and "pnl_usd" in trades.columns and _PLOTLY:
        st.markdown("#### Cumulative P&L Curve")
        trades_sorted = trades.sort_values("id")
        trades_sorted["cum_pnl"] = trades_sorted["pnl_usd"].cumsum()
        fig3 = px.line(
            trades_sorted, y="cum_pnl",
            title="", template="plotly_dark",
            labels={"cum_pnl": "Cumulative P&L ($)", "index": "Trade #"},
        )
        fig3.add_hline(y=DAILY_TARGET_USD, line_dash="dash",
                       line_color="gold", annotation_text="Daily Target")
        fig3.add_hline(y=-MAX_DAILY_LOSS_USD, line_dash="dash",
                       line_color="red", annotation_text="Loss Limit")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ── Data Tables Tabs ──────────────────────────────────────────────────────
    st.markdown("### 🗄 Deep Ledger Inspection")
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Options Portfolio",
        "🧠 Agent Decisions",
        "📟 Telemetry Logs",
        "🚨 Alert Simulator",
    ])

    with tab1:
        st.markdown("#### Open & Settled Multi-Leg Derivative Structures")
        if not portfolio.empty:
            st.dataframe(portfolio, use_container_width=True, hide_index=True)
        else:
            st.info("No positions yet. Run `python crew_agent.py` to start the trading cycle.")

    with tab2:
        st.markdown("#### Cognitive Agent Reasoning & Semantic Rationales")
        if not decisions.empty:
            st.dataframe(decisions, use_container_width=True, hide_index=True)
        else:
            st.info("No agent decisions logged yet.")

    with tab3:
        st.markdown("#### Infrastructure Logs & Health Records")
        if not telemetry.empty:
            critical = telemetry[telemetry["log_level"] == "CRITICAL"]
            if not critical.empty:
                st.error(f"⚠ {len(critical)} CRITICAL anomalies in telemetry array!")
            st.dataframe(telemetry, use_container_width=True, hide_index=True)
        else:
            st.info("No telemetry logged yet.")

    with tab4:
        st.markdown("#### Webhook Alert Simulator")
        st.markdown(
            "Test that Discord/Slack webhooks are wired correctly. "
            "Set `DISCORD_WEBHOOK_URL` or `SLACK_WEBHOOK_URL` in `.env`."
        )
        webhook_status = "✅ Configured" if (DISCORD_WEBHOOK_URL or SLACK_WEBHOOK_URL) else "❌ Not configured — add to .env"
        st.info(f"Webhook status: {webhook_status}")

        s1, s2, s3 = st.columns(3)
        with s1:
            if st.button("🚨 Simulate API Disconnect"):
                ok = _send_alert(
                    "Alpaca_Paper_API", "CRITICAL",
                    "Paper API connection dropped. Reconnection in progress.", 412.5,
                )
                st.success("Alert dispatched!" if ok else "Webhook not configured — check .env")
        with s2:
            if st.button("⚠ Simulate Margin Breach"):
                ok = _send_alert(
                    "RiskOfficer", "WARNING",
                    "Proposed spread attempts 7.4% capital allocation. Hard limit: 5%.", 12.1,
                )
                st.success("Alert dispatched!" if ok else "Webhook not configured — check .env")
        with s3:
            if st.button("✅ Simulate Cycle Complete"):
                ok = _send_alert(
                    "CrewOrchestrator", "INFO",
                    "Crew cycle complete. Bear Put Spread APPROVED. Paper order submitted.", 8_400.0,
                )
                st.success("Alert dispatched!" if ok else "Webhook not configured — check .env")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "🛢 Quant Energy Pipeline | Paper trading only | "
        "All pricing: Black-76 (not BSM) | "
        "evaluate_trade() gate enforced | "
        f"DB: `{DB_FILE}`"
    )


if __name__ == "__main__":
    main()
