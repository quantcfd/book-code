"""
src/indicators.py — Technical indicators dùng nhiều lần trong các chiến lược.

Tất cả functions phải:
    1. Input: pd.Series giá close (hoặc DataFrame OHLCV nếu cần High/Low)
    2. Output: pd.Series cùng index với input
    3. Vectorized — không for-loop
    4. Có docstring + type hints
    5. Cẩn thận với NaN ở đầu (do rolling)

Đầy đủ implementation học dần qua các chương 4-9.
"""
import numpy as np
import pandas as pd


def sma(close: pd.Series, period: int = 20) -> pd.Series:
    """Simple Moving Average."""
    return close.rolling(period).mean()


def ema(close: pd.Series, period: int = 20) -> pd.Series:
    """Exponential Moving Average."""
    return close.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index — Wilder's original formula (EMA-based smoothing).

    Returns:
        pd.Series of RSI values [0, 100].
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Wilder's smoothing = EMA với α = 1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range — đo volatility từ OHLC data.

    Args:
        df: DataFrame phải có cột High, Low, Close.
        period: số kỳ tính ATR.

    Returns:
        pd.Series of ATR values cùng index.
    """
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands.

    Returns:
        DataFrame với 3 cột: 'middle', 'upper', 'lower'.
    """
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    return pd.DataFrame(
        {
            "middle": middle,
            "upper": middle + num_std * std,
            "lower": middle - num_std * std,
        },
        index=close.index,
    )


def realized_volatility(returns: pd.Series, period: int = 20) -> pd.Series:
    """Annualized realized volatility từ returns. Default 20-day rolling."""
    return returns.rolling(period).std() * np.sqrt(252)
