"""
Candlestick Chart Viewer with Pattern Detection & Technical Indicators
Based on "Candlestick Charting for Dummies" by Russell Rhoads (Wiley, 2008)

Run:  python candlestick_app.py
Then open  http://127.0.0.1:8050  in your browser.
"""

import json
from datetime import date as _date

import pandas as pd
import yfinance as yf
import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context, ALL
import dash_bootstrap_components as dbc

import indicators as ind
import patterns as pat
import visualization as viz

# ─── app setup ──────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Candlestick Pro",
)

PERIOD_OPTIONS = [
    {"label": "1 Day",    "value": "1d"},
    {"label": "5 Days",   "value": "5d"},
    {"label": "1 Month",  "value": "1mo"},
    {"label": "3 Months", "value": "3mo"},
    {"label": "6 Months", "value": "6mo"},
    {"label": "1 Year",   "value": "1y"},
    {"label": "2 Years",  "value": "2y"},
    {"label": "5 Years",  "value": "5y"},
]

INTERVAL_OPTIONS = [
    {"label": "1 Min",   "value": "1m"},
    {"label": "5 Min",   "value": "5m"},
    {"label": "15 Min",  "value": "15m"},
    {"label": "30 Min",  "value": "30m"},
    {"label": "1 Hour",  "value": "1h"},
    {"label": "Daily",   "value": "1d"},
    {"label": "Weekly",  "value": "1wk"},
    {"label": "Monthly", "value": "1mo"},
]

INDICATOR_OPTIONS = [
    {"label": "RSI (14)",    "value": "rsi"},
    {"label": "Stochastics", "value": "stoch"},
    {"label": "MACD",        "value": "macd"},
    {"label": "Volume",      "value": "volume"},
]

MA_OPTIONS = [
    {"label": "SMA 10",  "value": "SMA_10"},
    {"label": "SMA 20",  "value": "SMA_20"},
    {"label": "SMA 50",  "value": "SMA_50"},
    {"label": "SMA 200", "value": "SMA_200"},
    {"label": "EMA 12",  "value": "EMA_12"},
    {"label": "EMA 26",  "value": "EMA_26"},
    {"label": "EMA 50",  "value": "EMA_50"},
    {"label": "WMA 10",  "value": "WMA_10"},
    {"label": "WMA 20",  "value": "WMA_20"},
]

PATTERN_OPTIONS = [{"label": name, "value": name} for name in pat.ALL_PATTERNS]

DEFAULT_WATCHLIST = ["SPY", "QQQ", "GLD", "BTC-USD", "CL=F"]


# ─── layout ─────────────────────────────────────────────────────────────────

sidebar = dbc.Col([
    html.H4("Candlestick Pro", className="text-warning fw-bold mb-3"),
    html.Hr(),

    # ── Ticker ──
    html.Label("Ticker Symbol"),
    dbc.InputGroup([
        dbc.Input(id="ticker-input", value="AAPL", debounce=True,
                  placeholder="e.g. AAPL, MSFT, TSLA"),
        dbc.Button("Load", id="load-btn", color="warning", n_clicks=0),
    ], className="mb-2"),

    # ── Compare ──
    html.Label("Compare With"),
    dbc.InputGroup([
        dbc.Input(id="compare-input", value="", debounce=True,
                  placeholder="e.g. SPY, QQQ"),
        dbc.Button("✕", id="compare-clear-btn", color="secondary",
                   n_clicks=0, size="sm"),
    ], className="mb-3"),

    html.Hr(),

    # ── Date Mode Toggle ──
    dbc.RadioItems(
        id="date-mode",
        options=[
            {"label": "Preset", "value": "preset"},
            {"label": "Custom", "value": "custom"},
        ],
        value="preset",
        inline=True,
        className="mb-2",
    ),

    dbc.Collapse(id="preset-collapse", is_open=True, children=[
        html.Label("Period"),
        dcc.Dropdown(id="period-dd", options=PERIOD_OPTIONS, value="6mo",
                     clearable=False, className="mb-2"),
    ]),

    dbc.Collapse(id="custom-collapse", is_open=False, children=[
        html.Label("Date Range"),
        dcc.DatePickerRange(
            id="date-range",
            start_date="2024-01-01",
            end_date=str(_date.today()),
            display_format="YYYY-MM-DD",
            style={"fontSize": "11px"},
        ),
        html.Br(),
        html.Br(),
    ]),

    html.Label("Interval"),
    dcc.Dropdown(id="interval-dd", options=INTERVAL_OPTIONS, value="1d",
                 clearable=False, className="mb-3"),

    html.Hr(),
    html.Label("Sub-panels"),
    dcc.Checklist(id="indicator-check", options=INDICATOR_OPTIONS,
                  value=["rsi", "volume"], labelClassName="d-block mb-1",
                  className="mb-3"),

    html.Label("Moving Averages"),
    dcc.Checklist(id="ma-check", options=MA_OPTIONS,
                  value=["SMA_20", "SMA_50"],
                  labelClassName="d-block mb-1", className="mb-3"),

    html.Label("Bollinger Bands"),
    dbc.Switch(id="bb-switch", label="Show BB (20, 2σ)", value=True,
               className="mb-3"),

    html.Hr(),
    html.Label("Pattern Groups"),
    dcc.Checklist(
        id="pattern-group-check",
        options=[
            {"label": "Single-stick", "value": "single"},
            {"label": "Double-stick", "value": "double"},
            {"label": "Three-stick",  "value": "triple"},
        ],
        value=["single", "double", "triple"],
        labelClassName="d-block mb-1",
        className="mb-2",
    ),

    html.Label("Specific Patterns"),
    dcc.Dropdown(id="pattern-dd", options=PATTERN_OPTIONS,
                 value=[], multi=True, placeholder="All patterns",
                 className="mb-3"),

    html.Hr(),

    # ── Watchlist ──
    html.H6("Watchlist", className="text-warning mb-1"),
    dbc.InputGroup([
        dbc.Input(id="wl-add-input", placeholder="Add ticker...",
                  debounce=True, size="sm"),
        dbc.Button("+", id="wl-add-btn", color="success",
                   size="sm", n_clicks=0),
    ], className="mb-2"),
    html.Div(id="watchlist-display", className="mb-2"),

    html.Hr(),
    html.Div(id="status-msg", className="text-muted small"),

], width=2, className="bg-dark p-3 vh-100 overflow-auto position-fixed",
   style={"top": 0, "left": 0, "zIndex": 100})


main_content = dbc.Col([
    dcc.Loading(
        id="loading-chart",
        type="circle",
        children=dcc.Graph(
            id="main-chart",
            config={"scrollZoom": True, "displayModeBar": True,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
            style={"height": "78vh"},
        ),
    ),

    html.Hr(),

    dbc.Row([
        dbc.Col([
            html.H5("Pattern Detection Summary", className="text-warning"),
            html.Div(id="pattern-table"),
        ], width=8),
        dbc.Col([
            html.H5("Stop Levels", className="text-warning"),
            html.Div(id="stop-levels"),
        ], width=4),
    ]),
], width={"size": 10, "offset": 2}, className="p-3")


app.layout = dbc.Container([
    dbc.Row([sidebar, main_content], className="g-0"),
    dcc.Store(id="watchlist-store", data=DEFAULT_WATCHLIST),
    dcc.Interval(id="watchlist-interval", interval=60_000, n_intervals=0),
], fluid=True, className="bg-dark text-light")


# ─── helpers ────────────────────────────────────────────────────────────────

SINGLE_PATTERNS = {
    "Long Candle", "Marubozu", "Closing Marubozu", "Opening Marubozu",
    "Doji", "Dragonfly Doji", "Gravestone Doji", "Long-Legged Doji",
    "Spinning Top", "Hammer", "Hanging Man", "Belt Hold",
}
DOUBLE_PATTERNS = {
    "Engulfing", "Harami", "Harami Cross", "Piercing / Dark Cloud",
    "Meeting Lines", "Inverted Hammer", "Doji Star",
    "Thrusting Lines", "Separating Lines", "On-Neck", "In-Neck",
}
TRIPLE_PATTERNS = set(pat.ALL_PATTERNS.keys()) - SINGLE_PATTERNS - DOUBLE_PATTERNS


def _active_patterns(group_check, pattern_dd):
    groups = set(group_check or [])
    specific = set(pattern_dd or [])
    allowed = set()
    if "single" in groups:
        allowed |= SINGLE_PATTERNS
    if "double" in groups:
        allowed |= DOUBLE_PATTERNS
    if "triple" in groups:
        allowed |= TRIPLE_PATTERNS
    if specific:
        allowed &= specific if allowed else specific
    return allowed


def _stop_level_card(df, signals_df):
    latest = signals_df.iloc[-5:]
    rows = []
    for date_idx, row in latest.iterrows():
        bulls = row[row == 1].index.tolist()
        bears = row[row == -1].index.tolist()
        bar = df.loc[date_idx]
        for p in bulls:
            stop = bar["Low"]
            rows.append(html.Div([
                html.Span(f"{str(date_idx)[:10]} ", className="text-muted small"),
                html.Span(f"▲ {p}", className="text-success small"),
                html.Span(f"  Sell stop: {stop:.2f}", className="text-warning small ms-2"),
            ], className="mb-1"))
        for p in bears:
            stop = bar["High"]
            rows.append(html.Div([
                html.Span(f"{str(date_idx)[:10]} ", className="text-muted small"),
                html.Span(f"▼ {p}", className="text-danger small"),
                html.Span(f"  Buy stop: {stop:.2f}", className="text-warning small ms-2"),
            ], className="mb-1"))
    if not rows:
        return html.P("No recent signals.", className="text-muted small")
    return html.Div(rows)


# ─── cache ──────────────────────────────────────────────────────────────────

_cache: dict = {}


def _fetch(ticker, period, interval, start=None, end=None):
    key = (ticker.upper(), period, interval, start, end)
    if key in _cache:
        return _cache[key]
    if start and end:
        raw = yf.download(ticker, start=start, end=end, interval=interval,
                          auto_adjust=True, progress=False)
    else:
        raw = yf.download(ticker, period=period, interval=interval,
                          auto_adjust=True, progress=False)
    if raw.empty:
        return None
    raw.index = pd.to_datetime(raw.index)
    if raw.index.tzinfo is not None:
        raw.index = raw.index.tz_localize(None)
    if hasattr(raw.columns, "levels"):
        raw.columns = raw.columns.get_level_values(0)
    df = ind.add_all(raw)
    _cache[key] = df
    return df


def _get_quote(ticker):
    """Return (last_price, pct_change_1d) for watchlist display."""
    try:
        fi = yf.Ticker(ticker).fast_info
        price = fi.last_price
        prev = fi.previous_close
        if price and prev and prev != 0:
            return price, (price - prev) / prev * 100
    except Exception:
        pass
    return None, None


# ─── callbacks ──────────────────────────────────────────────────────────────

@app.callback(
    Output("preset-collapse", "is_open"),
    Output("custom-collapse", "is_open"),
    Input("date-mode", "value"),
)
def toggle_date_mode(mode):
    return mode == "preset", mode == "custom"


@app.callback(
    Output("compare-input", "value"),
    Input("compare-clear-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_compare(_):
    return ""


@app.callback(
    Output("watchlist-store", "data"),
    Output("wl-add-input", "value"),
    Input("wl-add-btn", "n_clicks"),
    Input("wl-add-input", "n_submit"),
    State("wl-add-input", "value"),
    State("watchlist-store", "data"),
    prevent_initial_call=True,
)
def add_to_watchlist(_clicks, _submit, ticker, wl):
    wl = list(wl or DEFAULT_WATCHLIST)
    if ticker:
        t = ticker.upper().strip()
        if t and t not in wl:
            wl.append(t)
    return wl, ""


@app.callback(
    Output("watchlist-display", "children"),
    Input("watchlist-store", "data"),
    Input("watchlist-interval", "n_intervals"),
)
def refresh_watchlist(wl, _):
    wl = wl or DEFAULT_WATCHLIST
    rows = []
    for ticker in wl:
        price, chg = _get_quote(ticker)
        if price is None:
            price_str, chg_str, color = "N/A", "", "text-muted"
        else:
            price_str = f"{price:.2f}"
            chg_str = f"{chg:+.2f}%"
            color = "text-success" if chg >= 0 else "text-danger"

        rows.append(
            dbc.Button(
                [
                    html.Span(ticker, className="fw-bold",
                              style={"fontSize": "11px", "minWidth": "50px",
                                     "display": "inline-block"}),
                    html.Span(f" {price_str}", className="text-light",
                              style={"fontSize": "10px"}),
                    html.Span(f" {chg_str}", className=color,
                              style={"fontSize": "10px"}),
                ],
                id={"type": "wl-btn", "index": ticker},
                color="dark",
                outline=True,
                n_clicks=0,
                className="d-block w-100 text-start mb-1 p-1",
                style={"borderColor": "#333"},
            )
        )
    return rows


@app.callback(
    Output("ticker-input", "value"),
    Input({"type": "wl-btn", "index": ALL}, "n_clicks"),
    State({"type": "wl-btn", "index": ALL}, "id"),
    State("ticker-input", "value"),
    prevent_initial_call=True,
)
def watchlist_click(all_clicks, all_ids, current):
    ctx = callback_context
    if not ctx.triggered:
        return current
    prop = ctx.triggered[0]["prop_id"]
    try:
        idx = json.loads(prop.rsplit(".", 1)[0])["index"]
        return idx
    except (json.JSONDecodeError, KeyError, IndexError):
        return current


# ─── main callback ───────────────────────────────────────────────────────────

@app.callback(
    Output("main-chart", "figure"),
    Output("pattern-table", "children"),
    Output("stop-levels", "children"),
    Output("status-msg", "children"),
    Input("load-btn", "n_clicks"),
    Input("ticker-input", "value"),
    Input("indicator-check", "value"),
    Input("ma-check", "value"),
    Input("bb-switch", "value"),
    Input("pattern-group-check", "value"),
    Input("pattern-dd", "value"),
    Input("compare-input", "value"),
    Input("period-dd", "value"),
    Input("interval-dd", "value"),
    Input("date-mode", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    prevent_initial_call=False,
)
def update_chart(n_clicks, ticker, ind_vals, ma_vals, show_bb,
                 grp_check, pat_dd, compare_ticker,
                 period, interval, date_mode, range_start, range_end):
    ticker = (ticker or "AAPL").upper().strip()
    period = period or "6mo"
    interval = interval or "1d"
    use_custom = date_mode == "custom" and range_start and range_end

    df = _fetch(ticker, period, interval,
                start=range_start if use_custom else None,
                end=range_end if use_custom else None)

    if df is None:
        empty_fig = viz.build_chart(
            pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]),
            pd.DataFrame(), ticker, [], False, False,
        )
        return (empty_fig,
                html.P("No data returned.", className="text-danger"),
                "",
                f"Error: no data for {ticker}")

    # comparison overlay
    compare_df = None
    cmp = (compare_ticker or "").upper().strip()
    if cmp and cmp != ticker:
        compare_df = _fetch(cmp, period, interval,
                            start=range_start if use_custom else None,
                            end=range_end if use_custom else None)

    # active patterns
    allowed = _active_patterns(grp_check, pat_dd)
    sig_df = pat.detect_all(df)
    filtered = sig_df[[c for c in sig_df.columns if c in allowed]]

    df_view = df.copy()
    all_ma_cols = [c for c in df_view.columns if c.startswith(("SMA_", "EMA_", "WMA_"))]
    to_hide = [c for c in all_ma_cols if c not in (ma_vals or [])]
    df_view = df_view.drop(columns=to_hide, errors="ignore")

    fig = viz.build_chart(
        df_view, filtered, ticker,
        indicators=tuple(ind_vals or []),
        show_ma=bool(ma_vals),
        show_bb=show_bb,
        compare_df=compare_df,
        compare_ticker=cmp if compare_df is not None else None,
    )

    tbl_html = viz.pattern_summary_table(filtered, df)
    tbl = dash_table.DataTable(
        data=pd.read_html(tbl_html)[0].to_dict("records") if "<table" in tbl_html else [],
        columns=(
            [{"name": c, "id": c}
             for c in ["Date", "Close", "Bullish Patterns", "Bearish Patterns"]]
            if "<table" in tbl_html else []
        ),
        style_table={"overflowX": "auto", "maxHeight": "200px"},
        style_header={"backgroundColor": "#222", "color": "#ffc107", "fontWeight": "bold"},
        style_cell={"backgroundColor": "#111", "color": "#eee", "fontSize": "12px",
                    "textAlign": "left", "padding": "4px 8px"},
        style_data_conditional=[
            {"if": {"filter_query": "{Bullish Patterns} != '—'"},
             "color": "#26a69a"},
            {"if": {"filter_query": "{Bearish Patterns} != '—'"},
             "color": "#ef5350"},
        ],
        page_size=10,
    ) if "<table" in tbl_html else html.P("No patterns detected.", className="text-muted small")

    stops = _stop_level_card(df, filtered)
    cmp_note = f" | vs {cmp}" if compare_df is not None else ""
    intraday_note = " ⚠ intraday data limited to recent history" if interval in ("1m", "5m", "15m", "30m") else ""
    status = f"Loaded {ticker}{cmp_note}: {len(df)} bars | {len(allowed)} patterns active{intraday_note}"

    return fig, tbl, stops, status


# ─── run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=8050)
