"""
Candlestick pattern detection based on "Candlestick Charting for Dummies" by Russell Rhoads.
All patterns return a pandas Series with values: 1 (bullish), -1 (bearish), 0 (no pattern).
"""

import numpy as np
import pandas as pd


# ─── helpers ────────────────────────────────────────────────────────────────

def _body(df):
    return (df["Close"] - df["Open"]).abs()

def _range(df):
    return df["High"] - df["Low"]

def _upper_wick(df):
    return df["High"] - df[["Open", "Close"]].max(axis=1)

def _lower_wick(df):
    return df[["Open", "Close"]].min(axis=1) - df["Low"]

def _is_bullish(df):
    return df["Close"] > df["Open"]

def _is_bearish(df):
    return df["Close"] < df["Open"]

def _avg_body(df, window=14):
    return _body(df).rolling(window).mean()

def _gap_up(prev, curr):
    """True where curr Low > prev High."""
    return curr["Low"] > prev["High"]

def _gap_down(prev, curr):
    """True where curr High < prev Low."""
    return curr["High"] < prev["Low"]

def _in_downtrend(df, window=5):
    return df["Close"].rolling(window).mean().diff() < 0

def _in_uptrend(df, window=5):
    return df["Close"].rolling(window).mean().diff() > 0


# ─── single-stick patterns ───────────────────────────────────────────────────

def long_candle(df, body_factor=2.0):
    """Long white (bullish) or long black (bearish) candle."""
    avg = _avg_body(df)
    big = _body(df) >= body_factor * avg
    result = pd.Series(0, index=df.index)
    result[big & _is_bullish(df)] = 1
    result[big & _is_bearish(df)] = -1
    return result


def marubozu(df, wick_pct=0.01):
    """White/black marubozu — tiny or no wicks."""
    b = _body(df)
    r = _range(df)
    no_wick = ((_upper_wick(df) + _lower_wick(df)) / r.replace(0, np.nan)) < wick_pct
    result = pd.Series(0, index=df.index)
    result[no_wick & _is_bullish(df)] = 1
    result[no_wick & _is_bearish(df)] = -1
    return result


def closing_marubozu(df, wick_pct=0.01):
    """Closing marubozu — no wick on the closing end."""
    r = _range(df)
    no_upper_white = (_upper_wick(df) / r.replace(0, np.nan)) < wick_pct
    no_lower_black = (_lower_wick(df) / r.replace(0, np.nan)) < wick_pct
    result = pd.Series(0, index=df.index)
    result[no_upper_white & _is_bullish(df)] = 1
    result[no_lower_black & _is_bearish(df)] = -1
    return result


def opening_marubozu(df, wick_pct=0.01):
    """Opening marubozu — no wick on the opening end."""
    r = _range(df)
    no_lower_white = (_lower_wick(df) / r.replace(0, np.nan)) < wick_pct
    no_upper_black = (_upper_wick(df) / r.replace(0, np.nan)) < wick_pct
    result = pd.Series(0, index=df.index)
    result[no_lower_white & _is_bullish(df)] = 1
    result[no_upper_black & _is_bearish(df)] = -1
    return result


def doji(df, body_pct=0.05):
    """Doji — open ≈ close (body < body_pct of range)."""
    r = _range(df)
    small_body = (_body(df) / r.replace(0, np.nan)) < body_pct
    return pd.Series(small_body.astype(int), index=df.index)


def dragonfly_doji(df, body_pct=0.05, upper_pct=0.05):
    """Dragonfly doji — open=close≈high, long lower wick."""
    r = _range(df)
    is_doji = (_body(df) / r.replace(0, np.nan)) < body_pct
    tiny_upper = (_upper_wick(df) / r.replace(0, np.nan)) < upper_pct
    return pd.Series((is_doji & tiny_upper).astype(int), index=df.index)


def gravestone_doji(df, body_pct=0.05, lower_pct=0.05):
    """Gravestone doji — open=close≈low, long upper wick."""
    r = _range(df)
    is_doji = (_body(df) / r.replace(0, np.nan)) < body_pct
    tiny_lower = (_lower_wick(df) / r.replace(0, np.nan)) < lower_pct
    return pd.Series((is_doji & tiny_lower).astype(int), index=df.index)


def long_legged_doji(df, body_pct=0.05, wick_pct=0.2):
    """Long-legged doji — doji with long wicks both sides."""
    r = _range(df)
    is_doji = (_body(df) / r.replace(0, np.nan)) < body_pct
    long_upper = (_upper_wick(df) / r.replace(0, np.nan)) > wick_pct
    long_lower = (_lower_wick(df) / r.replace(0, np.nan)) > wick_pct
    return pd.Series((is_doji & long_upper & long_lower).astype(int), index=df.index)


def spinning_top(df, body_pct=0.25, wick_min=0.1):
    """Spinning top — small body, wicks longer than body on both sides."""
    r = _range(df)
    b = _body(df)
    small = (b / r.replace(0, np.nan)) < body_pct
    wicks_present = (
        (_upper_wick(df) / r.replace(0, np.nan)) > wick_min
    ) & (
        (_lower_wick(df) / r.replace(0, np.nan)) > wick_min
    )
    result = pd.Series(0, index=df.index)
    result[small & wicks_present & _is_bullish(df)] = 1
    result[small & wicks_present & _is_bearish(df)] = -1
    return result


def hammer(df, body_pct=0.35, lower_min=2.0, upper_max=0.1):
    """Hammer — small body at top, long lower wick (bullish in downtrend)."""
    r = _range(df)
    b = _body(df)
    lw = _lower_wick(df)
    uw = _upper_wick(df)
    small = (b / r.replace(0, np.nan)) < body_pct
    long_lower = lw >= lower_min * b
    tiny_upper = (uw / r.replace(0, np.nan)) < upper_max
    downtrend = _in_downtrend(df)
    signal = small & long_lower & tiny_upper & downtrend
    result = pd.Series(0, index=df.index)
    result[signal] = 1
    return result


def hanging_man(df, body_pct=0.35, lower_min=2.0, upper_max=0.1):
    """Hanging man — same shape as hammer but in uptrend (bearish)."""
    r = _range(df)
    b = _body(df)
    lw = _lower_wick(df)
    uw = _upper_wick(df)
    small = (b / r.replace(0, np.nan)) < body_pct
    long_lower = lw >= lower_min * b
    tiny_upper = (uw / r.replace(0, np.nan)) < upper_max
    uptrend = _in_uptrend(df)
    signal = small & long_lower & tiny_upper & uptrend
    result = pd.Series(0, index=df.index)
    result[signal] = -1
    return result


def belt_hold(df, wick_pct=0.05, body_factor=1.5):
    """Belt hold — long candle with no wick on opening end."""
    avg = _avg_body(df)
    r = _range(df)
    big = _body(df) >= body_factor * avg
    bullish_bh = big & _is_bullish(df) & ((_lower_wick(df) / r.replace(0, np.nan)) < wick_pct) & _in_downtrend(df)
    bearish_bh = big & _is_bearish(df) & ((_upper_wick(df) / r.replace(0, np.nan)) < wick_pct) & _in_uptrend(df)
    result = pd.Series(0, index=df.index)
    result[bullish_bh] = 1
    result[bearish_bh] = -1
    return result


# ─── double-stick patterns ───────────────────────────────────────────────────

def _prev(df):
    return df.shift(1)


def engulfing(df):
    """Bullish/bearish engulfing — second body fully covers first body."""
    p = _prev(df)
    bull = (
        _is_bearish(p) & _is_bullish(df)
        & (df["Open"] <= p["Close"])
        & (df["Close"] >= p["Open"])
    )
    bear = (
        _is_bullish(p) & _is_bearish(df)
        & (df["Open"] >= p["Close"])
        & (df["Close"] <= p["Open"])
    )
    result = pd.Series(0, index=df.index)
    result[bull] = 1
    result[bear] = -1
    return result


def harami(df, inner_pct=0.6):
    """Harami — second candle body inside first candle body."""
    p = _prev(df)
    p_body_high = p[["Open", "Close"]].max(axis=1)
    p_body_low = p[["Open", "Close"]].min(axis=1)
    c_body_high = df[["Open", "Close"]].max(axis=1)
    c_body_low = df[["Open", "Close"]].min(axis=1)
    inside = (c_body_high <= p_body_high) & (c_body_low >= p_body_low)
    bull = inside & _is_bearish(p) & _is_bullish(df)
    bear = inside & _is_bullish(p) & _is_bearish(df)
    result = pd.Series(0, index=df.index)
    result[bull] = 1
    result[bear] = -1
    return result


def harami_cross(df, body_pct=0.05):
    """Harami cross — harami where second candle is a doji."""
    r2 = _range(df)
    is_doji = (_body(df) / r2.replace(0, np.nan)) < body_pct
    p = _prev(df)
    p_body_high = p[["Open", "Close"]].max(axis=1)
    p_body_low = p[["Open", "Close"]].min(axis=1)
    c_mid = (df["Open"] + df["Close"]) / 2
    inside = (c_mid <= p_body_high) & (c_mid >= p_body_low)
    bull = inside & is_doji & _is_bearish(p)
    bear = inside & is_doji & _is_bullish(p)
    result = pd.Series(0, index=df.index)
    result[bull] = 1
    result[bear] = -1
    return result


def piercing_line(df):
    """Piercing line (bullish) / Dark cloud cover (bearish)."""
    p = _prev(df)
    mid_prev = (p["Open"] + p["Close"]) / 2
    bull = (
        _is_bearish(p) & _is_bullish(df)
        & (df["Open"] < p["Low"])
        & (df["Close"] > mid_prev)
        & (df["Close"] < p["Open"])
    )
    bear = (
        _is_bullish(p) & _is_bearish(df)
        & (df["Open"] > p["High"])
        & (df["Close"] < mid_prev)
        & (df["Close"] > p["Open"])
    )
    result = pd.Series(0, index=df.index)
    result[bull] = 1
    result[bear] = -1
    return result


def meeting_lines(df, close_pct=0.002):
    """Meeting lines — two candles close at same level."""
    p = _prev(df)
    same_close = (df["Close"] - p["Close"]).abs() / p["Close"] < close_pct
    bull = _is_bearish(p) & _is_bullish(df) & same_close
    bear = _is_bullish(p) & _is_bearish(df) & same_close
    result = pd.Series(0, index=df.index)
    result[bull] = 1
    result[bear] = -1
    return result


def inverted_hammer_double(df):
    """Inverted hammer in downtrend after bearish candle (bullish reversal)."""
    p = _prev(df)
    r = _range(df)
    b = _body(df)
    uw = _upper_wick(df)
    lw = _lower_wick(df)
    inv_hammer = (
        (uw >= 2 * b)
        & ((lw / r.replace(0, np.nan)) < 0.1)
        & _in_downtrend(df)
    )
    result = pd.Series(0, index=df.index)
    result[inv_hammer & _is_bearish(p)] = 1
    return result


def doji_star(df, body_pct=0.05):
    """Doji star — gap then doji (bullish after downtrend, bearish after uptrend)."""
    p = _prev(df)
    r2 = _range(df)
    is_doji = (_body(df) / r2.replace(0, np.nan)) < body_pct
    gap_up_cond = df["Low"] > p["High"]
    gap_down_cond = df["High"] < p["Low"]
    bull = is_doji & gap_up_cond & _is_bearish(p)   # after bearish with gap up = bullish star
    bear = is_doji & gap_down_cond & _is_bullish(p)
    result = pd.Series(0, index=df.index)
    result[bull] = 1
    result[bear] = -1
    return result


def thrusting_lines(df, body_pct=0.35):
    """Thrusting lines — second candle closes in lower half of first body (continuation)."""
    p = _prev(df)
    mid_prev = (p["Open"] + p["Close"]) / 2
    bull_thrust = (
        _is_bearish(p) & _is_bullish(df)
        & (df["Open"] < p["Low"])
        & (df["Close"] > p["Close"])
        & (df["Close"] < mid_prev)
    )
    bear_thrust = (
        _is_bullish(p) & _is_bearish(df)
        & (df["Open"] > p["High"])
        & (df["Close"] < p["Close"])
        & (df["Close"] > mid_prev)
    )
    result = pd.Series(0, index=df.index)
    result[bull_thrust] = 1
    result[bear_thrust] = -1
    return result


def separating_lines(df, open_pct=0.002):
    """Separating lines — same open, opposite color (continuation)."""
    p = _prev(df)
    same_open = (df["Open"] - p["Open"]).abs() / p["Open"] < open_pct
    bull = same_open & _is_bearish(p) & _is_bullish(df)
    bear = same_open & _is_bullish(p) & _is_bearish(df)
    result = pd.Series(0, index=df.index)
    result[bull] = 1
    result[bear] = -1
    return result


def on_neck(df, close_pct=0.002):
    """On-neck line — second candle closes near low of first (bearish continuation)."""
    p = _prev(df)
    near_low = (df["Close"] - p["Low"]).abs() / p["Low"] < close_pct
    bear = _is_bearish(p) & _is_bullish(df) & near_low & (df["Open"] < p["Low"])
    result = pd.Series(0, index=df.index)
    result[bear] = -1
    return result


def in_neck(df, close_pct=0.005):
    """In-neck line — second candle closes just inside first body (bearish continuation)."""
    p = _prev(df)
    just_in = (df["Close"] > p["Close"]) & ((df["Close"] - p["Close"]) / p["Close"] < close_pct)
    bear = _is_bearish(p) & _is_bullish(df) & just_in & (df["Open"] < p["Low"])
    result = pd.Series(0, index=df.index)
    result[bear] = -1
    return result


# ─── three-stick patterns ────────────────────────────────────────────────────

def _prev2(df):
    return df.shift(2)


def three_inside_up(df):
    """Three inside up — bearish, bullish harami, then confirming bullish."""
    p2 = _prev2(df)
    p1 = _prev(df)
    cond = (
        _is_bearish(p2) & _is_bullish(p1) & _is_bullish(df)
        & (p1["Open"] > p2["Close"]) & (p1["Close"] < p2["Open"])
        & (df["Close"] > p2["Open"])
    )
    result = pd.Series(0, index=df.index)
    result[cond] = 1
    return result


def three_inside_down(df):
    """Three inside down — bullish, bearish harami, then confirming bearish."""
    p2 = _prev2(df)
    p1 = _prev(df)
    cond = (
        _is_bullish(p2) & _is_bearish(p1) & _is_bearish(df)
        & (p1["Open"] < p2["Close"]) & (p1["Close"] > p2["Open"])
        & (df["Close"] < p2["Open"])
    )
    result = pd.Series(0, index=df.index)
    result[cond] = -1
    return result


def three_outside_up(df):
    """Three outside up — bearish, bullish engulfing, then confirming bullish."""
    p2 = _prev2(df)
    p1 = _prev(df)
    engulf = (
        _is_bearish(p2) & _is_bullish(p1)
        & (p1["Open"] <= p2["Close"]) & (p1["Close"] >= p2["Open"])
    )
    cond = engulf & _is_bullish(df) & (df["Close"] > p1["Close"])
    result = pd.Series(0, index=df.index)
    result[cond] = 1
    return result


def three_outside_down(df):
    """Three outside down — bullish, bearish engulfing, then confirming bearish."""
    p2 = _prev2(df)
    p1 = _prev(df)
    engulf = (
        _is_bullish(p2) & _is_bearish(p1)
        & (p1["Open"] >= p2["Close"]) & (p1["Close"] <= p2["Open"])
    )
    cond = engulf & _is_bearish(df) & (df["Close"] < p1["Close"])
    result = pd.Series(0, index=df.index)
    result[cond] = -1
    return result


def three_white_soldiers(df, body_factor=0.6):
    """Three white soldiers — three consecutive bullish candles, each closing higher."""
    p2 = _prev2(df)
    p1 = _prev(df)
    avg = _avg_body(df)
    decent_body = _body(df) > body_factor * avg
    cond = (
        _is_bullish(p2) & _is_bullish(p1) & _is_bullish(df)
        & (p1["Open"] > p2["Open"]) & (p1["Close"] > p2["Close"])
        & (df["Open"] > p1["Open"]) & (df["Close"] > p1["Close"])
        & decent_body
    )
    result = pd.Series(0, index=df.index)
    result[cond] = 1
    return result


def three_black_crows(df, body_factor=0.6):
    """Three black crows — three consecutive bearish candles, each closing lower."""
    p2 = _prev2(df)
    p1 = _prev(df)
    avg = _avg_body(df)
    decent_body = _body(df) > body_factor * avg
    cond = (
        _is_bearish(p2) & _is_bearish(p1) & _is_bearish(df)
        & (p1["Open"] < p2["Open"]) & (p1["Close"] < p2["Close"])
        & (df["Open"] < p1["Open"]) & (df["Close"] < p1["Close"])
        & decent_body
    )
    result = pd.Series(0, index=df.index)
    result[cond] = -1
    return result


def morning_star(df, body_pct=0.3):
    """Morning star — bearish, small/doji, bullish with gap transitions."""
    p2 = _prev2(df)
    p1 = _prev(df)
    r1 = p1["High"] - p1["Low"]
    small_mid = (_body(p1) / r1.replace(0, np.nan)) < body_pct
    cond = (
        _is_bearish(p2) & small_mid & _is_bullish(df)
        & (df["Close"] > (p2["Open"] + p2["Close"]) / 2)
    )
    result = pd.Series(0, index=df.index)
    result[cond] = 1
    return result


def evening_star(df, body_pct=0.3):
    """Evening star — bullish, small/doji, bearish with gap transitions."""
    p2 = _prev2(df)
    p1 = _prev(df)
    r1 = p1["High"] - p1["Low"]
    small_mid = (_body(p1) / r1.replace(0, np.nan)) < body_pct
    cond = (
        _is_bullish(p2) & small_mid & _is_bearish(df)
        & (df["Close"] < (p2["Open"] + p2["Close"]) / 2)
    )
    result = pd.Series(0, index=df.index)
    result[cond] = -1
    return result


def bullish_abandoned_baby(df, body_pct=0.05):
    """Bullish abandoned baby — bearish, gapped doji, gapped bullish."""
    p2 = _prev2(df)
    p1 = _prev(df)
    r1 = p1["High"] - p1["Low"]
    is_doji = (_body(p1) / r1.replace(0, np.nan)) < body_pct
    gap1 = p1["High"] < p2["Low"]
    gap2 = df["Low"] > p1["High"]
    cond = _is_bearish(p2) & is_doji & _is_bullish(df) & gap1 & gap2
    result = pd.Series(0, index=df.index)
    result[cond] = 1
    return result


def bearish_abandoned_baby(df, body_pct=0.05):
    """Bearish abandoned baby — bullish, gapped doji, gapped bearish."""
    p2 = _prev2(df)
    p1 = _prev(df)
    r1 = p1["High"] - p1["Low"]
    is_doji = (_body(p1) / r1.replace(0, np.nan)) < body_pct
    gap1 = p1["Low"] > p2["High"]
    gap2 = df["High"] < p1["Low"]
    cond = _is_bullish(p2) & is_doji & _is_bearish(df) & gap1 & gap2
    result = pd.Series(0, index=df.index)
    result[cond] = -1
    return result


def bullish_doji_star(df, body_pct=0.05):
    """Bullish doji star — bearish candle followed by gapped doji."""
    p = _prev(df)
    r = _range(df)
    is_doji = (_body(df) / r.replace(0, np.nan)) < body_pct
    gap_up_cond = df["Low"] > p["Low"]
    cond = _is_bearish(p) & is_doji & gap_up_cond
    result = pd.Series(0, index=df.index)
    result[cond] = 1
    return result


def bearish_doji_star(df, body_pct=0.05):
    """Bearish doji star — bullish candle followed by gapped doji."""
    p = _prev(df)
    r = _range(df)
    is_doji = (_body(df) / r.replace(0, np.nan)) < body_pct
    gap_down_cond = df["High"] < p["High"]
    cond = _is_bullish(p) & is_doji & gap_down_cond
    result = pd.Series(0, index=df.index)
    result[cond] = -1
    return result


def squeeze_alert(df):
    """Squeeze alert — doji between two candles of same color (indecision)."""
    p2 = _prev2(df)
    p1 = _prev(df)
    r1 = p1["High"] - p1["Low"]
    is_doji = (_body(p1) / r1.replace(0, np.nan)) < 0.05
    bull = _is_bearish(p2) & is_doji & _is_bullish(df)
    bear = _is_bullish(p2) & is_doji & _is_bearish(df)
    result = pd.Series(0, index=df.index)
    result[bull] = 1
    result[bear] = -1
    return result


def upside_tasuki_gap(df):
    """Upside tasuki gap — two bullish candles with gap, bearish that doesn't fill gap."""
    p2 = _prev2(df)
    p1 = _prev(df)
    gap = p1["Open"] > p2["Close"]
    cond = (
        _is_bullish(p2) & _is_bullish(p1) & _is_bearish(df)
        & gap
        & (df["Open"] < p1["Close"]) & (df["Open"] > p1["Open"])
        & (df["Close"] > p2["Close"])
    )
    result = pd.Series(0, index=df.index)
    result[cond] = 1
    return result


def downside_tasuki_gap(df):
    """Downside tasuki gap — two bearish candles with gap, bullish that doesn't fill gap."""
    p2 = _prev2(df)
    p1 = _prev(df)
    gap = p1["Open"] < p2["Close"]
    cond = (
        _is_bearish(p2) & _is_bearish(p1) & _is_bullish(df)
        & gap
        & (df["Open"] > p1["Close"]) & (df["Open"] < p1["Open"])
        & (df["Close"] < p2["Close"])
    )
    result = pd.Series(0, index=df.index)
    result[cond] = -1
    return result


# ─── registry ────────────────────────────────────────────────────────────────

ALL_PATTERNS = {
    # single-stick
    "Long Candle": long_candle,
    "Marubozu": marubozu,
    "Closing Marubozu": closing_marubozu,
    "Opening Marubozu": opening_marubozu,
    "Doji": doji,
    "Dragonfly Doji": dragonfly_doji,
    "Gravestone Doji": gravestone_doji,
    "Long-Legged Doji": long_legged_doji,
    "Spinning Top": spinning_top,
    "Hammer": hammer,
    "Hanging Man": hanging_man,
    "Belt Hold": belt_hold,
    # double-stick
    "Engulfing": engulfing,
    "Harami": harami,
    "Harami Cross": harami_cross,
    "Piercing / Dark Cloud": piercing_line,
    "Meeting Lines": meeting_lines,
    "Inverted Hammer": inverted_hammer_double,
    "Doji Star": doji_star,
    "Thrusting Lines": thrusting_lines,
    "Separating Lines": separating_lines,
    "On-Neck": on_neck,
    "In-Neck": in_neck,
    # three-stick
    "Three Inside Up": three_inside_up,
    "Three Inside Down": three_inside_down,
    "Three Outside Up": three_outside_up,
    "Three Outside Down": three_outside_down,
    "Three White Soldiers": three_white_soldiers,
    "Three Black Crows": three_black_crows,
    "Morning Star": morning_star,
    "Evening Star": evening_star,
    "Bullish Abandoned Baby": bullish_abandoned_baby,
    "Bearish Abandoned Baby": bearish_abandoned_baby,
    "Bullish Doji Star": bullish_doji_star,
    "Bearish Doji Star": bearish_doji_star,
    "Squeeze Alert": squeeze_alert,
    "Upside Tasuki Gap": upside_tasuki_gap,
    "Downside Tasuki Gap": downside_tasuki_gap,
}


def detect_all(df):
    """Run every pattern and return a DataFrame of signals."""
    results = {}
    for name, fn in ALL_PATTERNS.items():
        try:
            results[name] = fn(df)
        except Exception:
            results[name] = pd.Series(0, index=df.index)
    return pd.DataFrame(results)
