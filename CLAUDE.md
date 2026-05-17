# Oil Contracts and Trading — Candlestick Chart App

## Project Overview
Full Python candlestick chart viewer with pattern detection and technical indicators.
Based on "Candlestick Charting for Dummies" by Russell Rhoads (Wiley, 2008).

## Files
- `candlestick_app.py` — Dash web app (main entry point)
- `patterns.py`        — 35+ candlestick pattern detectors (single / double / three-stick)
- `indicators.py`      — SMA, EMA, WMA, RSI, Bollinger Bands, Stochastics, MACD
- `visualization.py`   — Plotly chart builder with pattern markers and sub-panels
- `requirements.txt`   — Python dependencies

## How to Run
```
pip install -r requirements.txt
python candlestick_app.py
```
Then open http://127.0.0.1:8050 in your browser.

## Dependencies
yfinance, pandas, numpy, plotly, dash, dash-bootstrap-components

## Patterns Implemented
### Single-stick
Long Candle, Marubozu, Closing Marubozu, Opening Marubozu,
Doji, Dragonfly Doji, Gravestone Doji, Long-Legged Doji,
Spinning Top, Hammer, Hanging Man, Belt Hold

### Double-stick
Engulfing, Harami, Harami Cross, Piercing/Dark Cloud,
Meeting Lines, Inverted Hammer, Doji Star,
Thrusting Lines, Separating Lines, On-Neck, In-Neck

### Three-stick
Three Inside Up/Down, Three Outside Up/Down,
Three White Soldiers, Three Black Crows,
Morning Star, Evening Star,
Bullish/Bearish Abandoned Baby, Bullish/Bearish Doji Star,
Squeeze Alert, Upside/Downside Tasuki Gap

## Technical Indicators
- RSI (14-period, overbought=70, oversold=30)
- Simple / Exponential / Weighted Moving Averages
- Bollinger Bands (20-day, ±2σ) with %B and Width
- Stochastic Oscillator (%K, %D; overbought=80, oversold=20)
- MACD (12/26/9)
