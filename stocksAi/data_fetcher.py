"""
data_fetcher.py
----------------
Pulls historical OHLCV (Open/High/Low/Close/Volume) data for any symbol
Yahoo Finance supports:
    Stocks:  AAPL, TSLA, RELIANCE.NS, TCS.NS ...
    Crypto:  BTC-USD, ETH-USD ...
    Forex:   EURUSD=X, GBPUSD=X ...
    Indices: ^GSPC (S&P 500), ^NSEI (Nifty 50) ...
"""

import pandas as pd
import yfinance as yf


class DataFetchError(Exception):
    pass


def fetch_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for `symbol` between `start` and `end`
    (both 'YYYY-MM-DD' strings).

    Returns a DataFrame indexed by date with columns:
    Open, High, Low, Close, Volume
    """
    symbol = symbol.strip().upper()
    if not symbol:
        raise DataFetchError("No symbol provided.")

    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)

    if df is None or df.empty:
        raise DataFetchError(
            f"No data found for '{symbol}'. Check the symbol is correct "
            f"(e.g. AAPL, BTC-USD, EURUSD=X, RELIANCE.NS) and the date range "
            f"actually has trading days in it."
        )

    # yfinance sometimes returns MultiIndex columns for a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    if len(df) < 120:
        raise DataFetchError(
            f"Only {len(df)} trading days of data found for '{symbol}' in that "
            f"range. Need at least ~120 trading days (roughly 6 months) to train "
            f"a usable model. Widen the date range."
        )

    return df
