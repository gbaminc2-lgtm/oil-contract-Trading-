"""
Technical indicators: SMA, EMA, WMA, RSI, Bollinger Bands, Stochastic Oscillator.
All functions accept a DataFrame with OHLCV columns and return a new DataFrame
with the indicator columns added, leaving the original intact.
"""

import numpy as np
import pandas as pd


def sma(df, periods=(10, 20, 50, 200)):
    """Simple Moving Averages."""
    out = df.copy()
    for p in periods:
        out[f"SMA_{p}"] = df["Close"].rolling(p).mean()
    return out


def ema(df, periods=(12, 26, 50)):
    """Exponential Moving Averages."""
    out = df.copy()
    for p in periods:
        out[f"EMA_{p}"] = df["Close"].ewm(span=p, adjust=False).mean()
    return out


def wma(df, periods=(10, 20)):
    """Weighted Moving Averages (linearly weighted)."""
    out = df.copy()
    for p in periods:
        weights = np.arange(1, p + 1, dtype=float)
        out[f"WMA_{p}"] = (
            df["Close"]
            .rolling(p)
            .apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        )
    return out


def rsi(df, period=14):
    """Relative Strength Index. Overbought ≥ 70, oversold ≤ 30."""
    out = df.copy()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))
    return out


def bollinger_bands(df, period=20, num_std=2):
    """Bollinger Bands: middle (SMA), upper, lower, %B, bandwidth."""
    out = df.copy()
    mid = df["Close"].rolling(period).mean()
    std = df["Close"].rolling(period).std()
    out["BB_Mid"] = mid
    out["BB_Upper"] = mid + num_std * std
    out["BB_Lower"] = mid - num_std * std
    band_width = out["BB_Upper"] - out["BB_Lower"]
    out["BB_PctB"] = (df["Close"] - out["BB_Lower"]) / band_width.replace(0, np.nan)
    out["BB_Width"] = band_width / mid.replace(0, np.nan)
    return out


def stochastic(df, k_period=14, d_period=3, smooth_k=3):
    """
    Stochastic oscillator.
    Returns %K (fast), %D (slow signal line), overbought=80, oversold=20.
    Uses smoothed %K (default 3-period SMA of raw %K).
    """
    out = df.copy()
    low_min = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    raw_k = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    out["STOCH_K"] = raw_k.rolling(smooth_k).mean()
    out["STOCH_D"] = out["STOCH_K"].rolling(d_period).mean()
    return out


def macd(df, fast=12, slow=26, signal=9):
    """MACD line, signal line, and histogram."""
    out = df.copy()
    fast_ema = df["Close"].ewm(span=fast, adjust=False).mean()
    slow_ema = df["Close"].ewm(span=slow, adjust=False).mean()
    out["MACD"] = fast_ema - slow_ema
    out["MACD_Signal"] = out["MACD"].ewm(span=signal, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]
    return out


def add_all(df, sma_periods=(10, 20, 50, 200), ema_periods=(12, 26, 50),
            wma_periods=(10, 20), rsi_period=14, bb_period=20, bb_std=2,
            stoch_k=14, stoch_d=3, stoch_smooth=3):
    """Convenience: add every indicator to df and return result."""
    out = df.copy()
    for p in sma_periods:
        out[f"SMA_{p}"] = df["Close"].rolling(p).mean()
    for p in ema_periods:
        out[f"EMA_{p}"] = df["Close"].ewm(span=p, adjust=False).mean()
    weights_dict = {p: np.arange(1, p + 1, dtype=float) for p in wma_periods}
    for p, w in weights_dict.items():
        out[f"WMA_{p}"] = (
            df["Close"]
            .rolling(p)
            .apply(lambda x, wts=w: np.dot(x, wts) / wts.sum(), raw=True)
        )
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    loss = (-delta).clip(lower=0).ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))

    mid = df["Close"].rolling(bb_period).mean()
    std = df["Close"].rolling(bb_period).std()
    out["BB_Mid"] = mid
    out["BB_Upper"] = mid + bb_std * std
    out["BB_Lower"] = mid - bb_std * std
    bw = out["BB_Upper"] - out["BB_Lower"]
    out["BB_PctB"] = (df["Close"] - out["BB_Lower"]) / bw.replace(0, np.nan)
    out["BB_Width"] = bw / mid.replace(0, np.nan)

    low_min = df["Low"].rolling(stoch_k).min()
    high_max = df["High"].rolling(stoch_k).max()
    raw_k = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    out["STOCH_K"] = raw_k.rolling(stoch_smooth).mean()
    out["STOCH_D"] = out["STOCH_K"].rolling(stoch_d).mean()

    fast_ema = df["Close"].ewm(span=12, adjust=False).mean()
    slow_ema = df["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = fast_ema - slow_ema
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]

    return out
