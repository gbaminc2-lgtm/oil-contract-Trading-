"""
Interactive Plotly chart builder for candlestick data with overlays and sub-plots.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BULLISH_COLOR = "#26a69a"
BEARISH_COLOR = "#ef5350"
PATTERN_BULL_COLOR = "rgba(38,166,154,0.25)"
PATTERN_BEAR_COLOR = "rgba(239,83,80,0.25)"


def _candle_trace(df):
    return go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price",
        increasing_line_color=BULLISH_COLOR,
        decreasing_line_color=BEARISH_COLOR,
        increasing_fillcolor=BULLISH_COLOR,
        decreasing_fillcolor=BEARISH_COLOR,
    )


def _volume_trace(df, row, col):
    colors = [BULLISH_COLOR if c >= o else BEARISH_COLOR
              for c, o in zip(df["Close"], df["Open"])]
    return go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=colors, opacity=0.7,
        row=row, col=col,
    )


def _add_ma_traces(fig, df, row=1):
    ma_styles = {
        "SMA_10":  ("SMA 10",  "#e91e63", "dot",   1),
        "SMA_20":  ("SMA 20",  "#9c27b0", "solid", 1),
        "SMA_50":  ("SMA 50",  "#3f51b5", "solid", 1.5),
        "SMA_200": ("SMA 200", "#ff9800", "solid", 2),
        "EMA_12":  ("EMA 12",  "#00bcd4", "dash",  1),
        "EMA_26":  ("EMA 26",  "#009688", "dash",  1),
        "EMA_50":  ("EMA 50",  "#4caf50", "dash",  1.5),
        "WMA_10":  ("WMA 10",  "#ff5722", "dashdot", 1),
        "WMA_20":  ("WMA 20",  "#795548", "dashdot", 1),
    }
    for col_name, (label, color, dash, width) in ma_styles.items():
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col_name], name=label,
                line=dict(color=color, dash=dash, width=width),
                visible="legendonly",
            ), row=row, col=1)


def _add_bb_traces(fig, df, row=1):
    if "BB_Upper" not in df.columns:
        return
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_Upper"], name="BB Upper",
        line=dict(color="rgba(100,100,200,0.6)", dash="dot", width=1),
        visible="legendonly",
    ), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_Lower"], name="BB Lower",
        line=dict(color="rgba(100,100,200,0.6)", dash="dot", width=1),
        fill="tonexty",
        fillcolor="rgba(100,100,200,0.05)",
        visible="legendonly",
    ), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_Mid"], name="BB Mid",
        line=dict(color="rgba(100,100,200,0.4)", dash="solid", width=1),
        visible="legendonly",
    ), row=row, col=1)


def _add_pattern_markers(fig, df, signals_df, row=1):
    """Add pattern markers as scatter points on the candlestick chart."""
    for pattern_name in signals_df.columns:
        sig = signals_df[pattern_name]
        bull_idx = sig[sig == 1].index
        bear_idx = sig[sig == -1].index

        if len(bull_idx):
            fig.add_trace(go.Scatter(
                x=bull_idx,
                y=df.loc[bull_idx, "Low"] * 0.995,
                mode="markers+text",
                name=f"▲ {pattern_name}",
                marker=dict(symbol="triangle-up", color=BULLISH_COLOR, size=10),
                text=[pattern_name[:6]] * len(bull_idx),
                textposition="bottom center",
                textfont=dict(size=7, color=BULLISH_COLOR),
                visible="legendonly",
            ), row=row, col=1)

        if len(bear_idx):
            fig.add_trace(go.Scatter(
                x=bear_idx,
                y=df.loc[bear_idx, "High"] * 1.005,
                mode="markers+text",
                name=f"▼ {pattern_name}",
                marker=dict(symbol="triangle-down", color=BEARISH_COLOR, size=10),
                text=[pattern_name[:6]] * len(bear_idx),
                textposition="top center",
                textfont=dict(size=7, color=BEARISH_COLOR),
                visible="legendonly",
            ), row=row, col=1)


def _rsi_traces(df):
    traces = []
    if "RSI" not in df.columns:
        return traces
    traces.append(go.Scatter(
        x=df.index, y=df["RSI"], name="RSI",
        line=dict(color="#7c4dff", width=1.5),
    ))
    traces.append(go.Scatter(
        x=df.index, y=[70] * len(df), name="OB (70)",
        line=dict(color="red", dash="dot", width=1), showlegend=False,
    ))
    traces.append(go.Scatter(
        x=df.index, y=[30] * len(df), name="OS (30)",
        line=dict(color="green", dash="dot", width=1),
        fill="tonexty", fillcolor="rgba(0,200,0,0.03)", showlegend=False,
    ))
    return traces


def _stoch_traces(df):
    traces = []
    if "STOCH_K" not in df.columns:
        return traces
    traces.append(go.Scatter(
        x=df.index, y=df["STOCH_K"], name="%K",
        line=dict(color="#2196f3", width=1.5),
    ))
    traces.append(go.Scatter(
        x=df.index, y=df["STOCH_D"], name="%D",
        line=dict(color="#ff9800", width=1.5, dash="dash"),
    ))
    traces.append(go.Scatter(
        x=df.index, y=[80] * len(df), name="OB (80)",
        line=dict(color="red", dash="dot", width=1), showlegend=False,
    ))
    traces.append(go.Scatter(
        x=df.index, y=[20] * len(df), name="OS (20)",
        line=dict(color="green", dash="dot", width=1),
        fill="tonexty", fillcolor="rgba(0,200,0,0.03)", showlegend=False,
    ))
    return traces


def _macd_traces(df):
    traces = []
    if "MACD" not in df.columns:
        return traces
    colors = [BULLISH_COLOR if v >= 0 else BEARISH_COLOR for v in df["MACD_Hist"].fillna(0)]
    traces.append(go.Bar(
        x=df.index, y=df["MACD_Hist"], name="MACD Hist",
        marker_color=colors, opacity=0.6,
    ))
    traces.append(go.Scatter(
        x=df.index, y=df["MACD"], name="MACD",
        line=dict(color="#2196f3", width=1.5),
    ))
    traces.append(go.Scatter(
        x=df.index, y=df["MACD_Signal"], name="Signal",
        line=dict(color="#ff9800", width=1.5, dash="dash"),
    ))
    return traces


def build_chart(df, signals_df, ticker, indicators=("rsi", "stoch", "volume"),
                show_ma=True, show_bb=True, compare_df=None, compare_ticker=None):
    panel_names = [i for i in indicators if i in ("rsi", "stoch", "macd", "volume")]
    n_sub = len(panel_names)
    rows = 1 + n_sub
    heights = [0.55] + [0.45 / n_sub] * n_sub if n_sub else [1.0]

    subplot_titles = [f"{ticker} — Candlestick"] + [p.upper() for p in panel_names]
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=heights,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles,
    )

    # ── main candlestick ──
    fig.add_trace(_candle_trace(df), row=1, col=1)
    if show_ma:
        _add_ma_traces(fig, df, row=1)
    if show_bb:
        _add_bb_traces(fig, df, row=1)
    if signals_df is not None and not signals_df.empty:
        _add_pattern_markers(fig, df, signals_df, row=1)

    # ── comparison overlay ──
    if compare_df is not None and not compare_df.empty and len(df) > 0:
        cmp_close = compare_df["Close"].reindex(df.index, method="ffill").dropna()
        if len(cmp_close) > 0:
            main_first = df["Close"].dropna().iloc[0]
            cmp_first = cmp_close.iloc[0]
            if cmp_first != 0:
                normalized = cmp_close * (main_first / cmp_first)
                fig.add_trace(go.Scatter(
                    x=normalized.index,
                    y=normalized.values,
                    name=compare_ticker,
                    line=dict(color="#ff9800", dash="dash", width=1.5),
                    opacity=0.85,
                ), row=1, col=1)

    # ── sub-panels ──
    for idx, panel in enumerate(panel_names, start=2):
        if panel == "rsi":
            for t in _rsi_traces(df):
                fig.add_trace(t, row=idx, col=1)
            fig.update_yaxes(range=[0, 100], row=idx, col=1)
        elif panel == "stoch":
            for t in _stoch_traces(df):
                fig.add_trace(t, row=idx, col=1)
            fig.update_yaxes(range=[0, 100], row=idx, col=1)
        elif panel == "macd":
            for t in _macd_traces(df):
                fig.add_trace(t, row=idx, col=1)
        elif panel == "volume":
            colors = [BULLISH_COLOR if c >= o else BEARISH_COLOR
                      for c, o in zip(df["Close"], df["Open"])]
            fig.add_trace(go.Bar(
                x=df.index, y=df["Volume"], name="Volume",
                marker_color=colors, opacity=0.7,
            ), row=idx, col=1)

    fig.update_layout(
        title=dict(text=f"<b>{ticker}</b> — Candlestick Chart", x=0.5),
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="v",
            x=1.01, y=1,
            bgcolor="rgba(0,0,0,0.4)",
            font=dict(size=10),
        ),
        hovermode="x unified",
        margin=dict(l=60, r=160, t=60, b=40),
        height=800 + 120 * n_sub,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")

    return fig


def pattern_summary_table(signals_df, df):
    """Return an HTML table string of detected patterns at each date."""
    rows = []
    for date in signals_df.index:
        active = signals_df.loc[date]
        bull = active[active == 1].index.tolist()
        bear = active[active == -1].index.tolist()
        if bull or bear:
            c = df.loc[date, "Close"]
            rows.append({
                "Date": str(date)[:10],
                "Close": f"{c:.2f}",
                "Bullish Patterns": ", ".join(bull) if bull else "—",
                "Bearish Patterns": ", ".join(bear) if bear else "—",
            })
    if not rows:
        return "<p>No patterns detected in this range.</p>"
    tbl = pd.DataFrame(rows)
    return tbl.to_html(index=False, classes="pattern-table", border=0)
