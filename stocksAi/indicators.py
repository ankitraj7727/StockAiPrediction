"""
indicators.py
--------------
Turns raw OHLCV price data into technical-indicator features the model
can learn from. All indicators are computed WITHOUT looking into the
future (no lookahead bias) - each row only uses data up to and including
that day.
"""

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _macd(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    # position of price within the bands, 0 = lower band, 1 = upper band
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return mid, upper, lower, pct_b.fillna(0.5)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with Open, High, Low, Close, Volume,
    return a new DataFrame with added technical-indicator columns.
    """
    out = df.copy()
    close = out["Close"]
    volume = out["Volume"]

    out["return_1d"] = close.pct_change(1)
    out["return_5d"] = close.pct_change(5)
    out["return_10d"] = close.pct_change(10)

    out["sma_10"] = close.rolling(10).mean()
    out["sma_20"] = close.rolling(20).mean()
    out["sma_50"] = close.rolling(50).mean()
    out["ema_12"] = close.ewm(span=12, adjust=False).mean()
    out["ema_26"] = close.ewm(span=26, adjust=False).mean()

    # normalized so the model sees "distance from trend" rather than raw price
    out["price_vs_sma20"] = (close - out["sma_20"]) / out["sma_20"]
    out["price_vs_sma50"] = (close - out["sma_50"]) / out["sma_50"]
    out["sma10_vs_sma50"] = (out["sma_10"] - out["sma_50"]) / out["sma_50"]

    out["rsi_14"] = _rsi(close, 14)

    macd_line, signal_line, hist = _macd(close)
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist

    _, _, _, pct_b = _bollinger(close, 20, 2.0)
    out["bb_pct_b"] = pct_b

    out["volatility_10d"] = out["return_1d"].rolling(10).std()
    out["volatility_20d"] = out["return_1d"].rolling(20).std()

    vol_avg_20 = volume.rolling(20).mean()
    out["volume_ratio"] = volume / vol_avg_20.replace(0, np.nan)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


FEATURE_COLUMNS = [
    "return_1d", "return_5d", "return_10d",
    "price_vs_sma20", "price_vs_sma50", "sma10_vs_sma50",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_pct_b", "volatility_10d", "volatility_20d", "volume_ratio",
]
