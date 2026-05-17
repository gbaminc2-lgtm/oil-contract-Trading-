"""
Oil Contract Trading - Visualization Module

Provides charting and visualization utilities for oil contract trading data:
price history, volume, positions, P&L, and market depth.
"""

import datetime
from typing import Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ---------------------------------------------------------------------------
# Data structures (plain dicts / dataclasses kept dependency-free)
# ---------------------------------------------------------------------------

def _require(lib_name: str, has_lib: bool) -> None:
    if not has_lib:
        raise ImportError(
            f"{lib_name} is required for this function. "
            f"Install it with: pip install {lib_name.lower()}"
        )


# ---------------------------------------------------------------------------
# Matplotlib-based charts
# ---------------------------------------------------------------------------

def plot_price_history(
    dates: list,
    prices: list,
    contract: str = "Crude Oil",
    title: Optional[str] = None,
    show: bool = True,
) -> "plt.Figure":
    """Line chart of contract closing prices over time."""
    _require("matplotlib", HAS_MATPLOTLIB)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, prices, linewidth=1.8, color="#1f77b4", label=contract)
    ax.fill_between(dates, prices, alpha=0.12, color="#1f77b4")

    ax.set_title(title or f"{contract} – Price History", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD/bbl)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_candlestick(
    dates: list,
    opens: list,
    highs: list,
    lows: list,
    closes: list,
    volumes: Optional[list] = None,
    contract: str = "Crude Oil",
    show: bool = True,
) -> "plt.Figure":
    """OHLC candlestick chart with optional volume panel."""
    _require("matplotlib", HAS_MATPLOTLIB)

    rows = 2 if volumes else 1
    fig = plt.figure(figsize=(14, 8 if volumes else 6))
    gs = GridSpec(rows, 1, figure=fig, height_ratios=([3, 1] if volumes else [1]))

    ax_price = fig.add_subplot(gs[0])

    for i, (d, o, h, l, c) in enumerate(zip(dates, opens, highs, lows, closes)):
        color = "#26a69a" if c >= o else "#ef5350"
        x = mdates.date2num(d) if isinstance(d, (datetime.date, datetime.datetime)) else i
        ax_price.plot([x, x], [l, h], color=color, linewidth=1)
        body_bottom = min(o, c)
        body_height = abs(c - o) or 0.01
        rect = mpatches.Rectangle(
            (x - 0.3, body_bottom), 0.6, body_height,
            facecolor=color, edgecolor=color
        )
        ax_price.add_patch(rect)

    if isinstance(dates[0], (datetime.date, datetime.datetime)):
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()

    ax_price.set_title(f"{contract} – OHLC", fontsize=14, fontweight="bold")
    ax_price.set_ylabel("Price (USD/bbl)")
    ax_price.grid(True, linestyle="--", alpha=0.4)

    if volumes:
        ax_vol = fig.add_subplot(gs[1], sharex=ax_price)
        colors = [
            "#26a69a" if c >= o else "#ef5350"
            for o, c in zip(opens, closes)
        ]
        xs = (
            [mdates.date2num(d) for d in dates]
            if isinstance(dates[0], (datetime.date, datetime.datetime))
            else list(range(len(dates)))
        )
        ax_vol.bar(xs, volumes, color=colors, alpha=0.7, width=0.6)
        ax_vol.set_ylabel("Volume")
        ax_vol.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_pnl(
    dates: list,
    pnl: list,
    contract: str = "Crude Oil",
    show: bool = True,
) -> "plt.Figure":
    """Cumulative P&L chart with green/red fill above/below zero."""
    _require("matplotlib", HAS_MATPLOTLIB)

    fig, ax = plt.subplots(figsize=(12, 5))
    pnl_arr = np.array(pnl)
    ax.plot(dates, pnl_arr, linewidth=1.8, color="#555", zorder=3)
    ax.fill_between(dates, pnl_arr, where=pnl_arr >= 0, alpha=0.3, color="#26a69a", label="Profit")
    ax.fill_between(dates, pnl_arr, where=pnl_arr < 0, alpha=0.3, color="#ef5350", label="Loss")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    ax.set_title(f"{contract} – Cumulative P&L", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("P&L (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_positions(
    contracts: list,
    long_positions: list,
    short_positions: list,
    show: bool = True,
) -> "plt.Figure":
    """Grouped bar chart comparing long vs. short positions per contract."""
    _require("matplotlib", HAS_MATPLOTLIB)

    x = np.arange(len(contracts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, long_positions, width, label="Long", color="#26a69a", alpha=0.85)
    ax.bar(x + width / 2, short_positions, width, label="Short", color="#ef5350", alpha=0.85)

    ax.set_title("Open Positions by Contract", fontsize=14, fontweight="bold")
    ax.set_xlabel("Contract")
    ax.set_ylabel("Contracts (lots)")
    ax.set_xticks(x)
    ax.set_xticklabels(contracts, rotation=20, ha="right")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_market_depth(
    bid_prices: list,
    bid_volumes: list,
    ask_prices: list,
    ask_volumes: list,
    show: bool = True,
) -> "plt.Figure":
    """Cumulative order book depth chart (bids vs. asks)."""
    _require("matplotlib", HAS_MATPLOTLIB)

    cum_bids = np.cumsum(list(reversed(bid_volumes)))
    cum_asks = np.cumsum(ask_volumes)
    bid_px = list(reversed(bid_prices))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.step(bid_px, cum_bids, where="post", color="#26a69a", linewidth=2, label="Bids")
    ax.fill_between(bid_px, cum_bids, step="post", alpha=0.2, color="#26a69a")
    ax.step(ask_prices, cum_asks, where="post", color="#ef5350", linewidth=2, label="Asks")
    ax.fill_between(ask_prices, cum_asks, step="post", alpha=0.2, color="#ef5350")

    ax.set_title("Market Depth (Order Book)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Price (USD/bbl)")
    ax.set_ylabel("Cumulative Volume")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_volatility(
    dates: list,
    prices: list,
    window: int = 20,
    contract: str = "Crude Oil",
    show: bool = True,
) -> "plt.Figure":
    """Rolling annualised volatility derived from daily log returns."""
    _require("matplotlib", HAS_MATPLOTLIB)

    prices_arr = np.array(prices, dtype=float)
    log_returns = np.diff(np.log(prices_arr))
    vol = np.full(len(log_returns), np.nan)
    for i in range(window - 1, len(log_returns)):
        vol[i] = np.std(log_returns[i - window + 1 : i + 1]) * np.sqrt(252) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(dates, prices, color="#1f77b4", linewidth=1.5)
    ax1.set_ylabel("Price (USD/bbl)")
    ax1.set_title(f"{contract} – Price & Rolling Volatility ({window}d)", fontsize=14, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(dates[1:], vol, color="#ff7f0e", linewidth=1.5)
    ax2.fill_between(dates[1:], vol, alpha=0.15, color="#ff7f0e")
    ax2.set_ylabel("Annualised Vol (%)")
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Plotly-based interactive charts
# ---------------------------------------------------------------------------

def plotly_candlestick(
    dates: list,
    opens: list,
    highs: list,
    lows: list,
    closes: list,
    volumes: Optional[list] = None,
    contract: str = "Crude Oil",
    show: bool = True,
) -> "go.Figure":
    """Interactive Plotly candlestick with optional volume panel."""
    _require("plotly", HAS_PLOTLY)

    rows = 2 if volumes else 1
    specs = [[{"type": "candlestick"}]] + ([[{"type": "bar"}]] if volumes else [])
    row_heights = [0.7, 0.3] if volumes else [1.0]

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=row_heights, specs=specs,
        vertical_spacing=0.03,
    )

    fig.add_trace(
        go.Candlestick(
            x=dates, open=opens, high=highs, low=lows, close=closes,
            name=contract,
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    if volumes:
        colors = [
            "#26a69a" if c >= o else "#ef5350"
            for o, c in zip(opens, closes)
        ]
        fig.add_trace(
            go.Bar(x=dates, y=volumes, marker_color=colors, name="Volume", opacity=0.7),
            row=2, col=1,
        )

    fig.update_layout(
        title=f"{contract} – Interactive OHLC",
        yaxis_title="Price (USD/bbl)",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
    )
    if show:
        fig.show()
    return fig


def plotly_pnl_dashboard(
    dates: list,
    pnl: list,
    prices: list,
    contract: str = "Crude Oil",
    show: bool = True,
) -> "go.Figure":
    """Interactive P&L + price dashboard with dual y-axis."""
    _require("plotly", HAS_PLOTLY)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=dates, y=prices, name="Price",
            line=dict(color="#1f77b4", width=2),
        ),
        secondary_y=False,
    )

    import numpy as _np
    pnl_arr = _np.array(pnl)
    fig.add_trace(
        go.Scatter(
            x=dates, y=pnl_arr, name="Cumulative P&L",
            line=dict(color="#ff7f0e", width=2, dash="dot"),
            fill="tozeroy",
            fillcolor="rgba(255,127,14,0.15)",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=f"{contract} – Price & P&L Dashboard",
        template="plotly_dark",
        height=500,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price (USD/bbl)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative P&L (USD)", secondary_y=True)
    if show:
        fig.show()
    return fig


def plotly_market_depth(
    bid_prices: list,
    bid_volumes: list,
    ask_prices: list,
    ask_volumes: list,
    show: bool = True,
) -> "go.Figure":
    """Interactive cumulative order-book depth chart."""
    _require("plotly", HAS_PLOTLY)

    import numpy as _np

    cum_bids = _np.cumsum(list(reversed(bid_volumes))).tolist()
    cum_asks = _np.cumsum(ask_volumes).tolist()
    bid_px = list(reversed(bid_prices))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bid_px, y=cum_bids, name="Bids",
        mode="lines", line=dict(color="#26a69a", width=2, shape="hv"),
        fill="tozeroy", fillcolor="rgba(38,166,154,0.2)",
    ))
    fig.add_trace(go.Scatter(
        x=ask_prices, y=cum_asks, name="Asks",
        mode="lines", line=dict(color="#ef5350", width=2, shape="hv"),
        fill="tozeroy", fillcolor="rgba(239,83,80,0.2)",
    ))

    fig.update_layout(
        title="Market Depth (Order Book)",
        xaxis_title="Price (USD/bbl)",
        yaxis_title="Cumulative Volume",
        template="plotly_dark",
        height=450,
        hovermode="x unified",
    )
    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

def _generate_demo_data(n: int = 120):
    """Generate synthetic oil price data for demonstration."""
    import random
    random.seed(42)
    base = datetime.date(2024, 1, 1)
    dates = [base + datetime.timedelta(days=i) for i in range(n)]

    price = 80.0
    opens, highs, lows, closes, volumes = [], [], [], [], []
    for _ in range(n):
        change = random.gauss(0, 1.2)
        o = price
        c = round(price + change, 2)
        h = round(max(o, c) + abs(random.gauss(0, 0.5)), 2)
        l = round(min(o, c) - abs(random.gauss(0, 0.5)), 2)
        v = int(random.gauss(50_000, 8_000))
        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        volumes.append(max(v, 1000))
        price = c

    pnl = [0.0]
    for i in range(1, n):
        pnl.append(round(pnl[-1] + (closes[i] - closes[i - 1]) * 1000, 2))

    return dates, opens, highs, lows, closes, volumes, pnl


def demo(backend: str = "matplotlib") -> None:
    """Run a quick visual demo using synthetic data.

    Args:
        backend: "matplotlib" or "plotly"
    """
    dates, opens, highs, lows, closes, volumes, pnl = _generate_demo_data()

    if backend == "plotly":
        plotly_candlestick(dates, opens, highs, lows, closes, volumes)
        plotly_pnl_dashboard(dates, pnl, closes)
    else:
        plot_price_history(dates, closes)
        plot_candlestick(dates, opens, highs, lows, closes, volumes)
        plot_pnl(dates, pnl)
        plot_volatility(dates, closes)

        contracts = ["WTI Dec24", "Brent Dec24", "WTI Mar25", "Brent Mar25"]
        longs = [120, 80, 45, 60]
        shorts = [30, 55, 90, 20]
        plot_positions(contracts, longs, shorts)

        mid = 82.50
        spread = 0.05
        bid_prices = [round(mid - spread * (i + 1), 2) for i in range(10)]
        bid_volumes = [500 + 50 * i for i in range(10)]
        ask_prices = [round(mid + spread * (i + 1), 2) for i in range(10)]
        ask_volumes = [480 + 40 * i for i in range(10)]
        plot_market_depth(bid_prices, bid_volumes, ask_prices, ask_volumes)


if __name__ == "__main__":
    demo()
