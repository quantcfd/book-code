"""
Bollinger Bands mean reversion strategy.
QuantCFD Chapter 8.3
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def bollinger_signals(
    df: pd.DataFrame,
    period: int = 20,
    k: float = 2.0,
    exit_at_middle: bool = True,
) -> pd.DataFrame:
    """
    Mean reversion strategy using Bollinger Bands.

    Entry: close < lower band (long) hoặc close > upper band (short)
    Exit: close cross middle band

    Args:
        df: DataFrame with 'close' column
        period: Lookback for SMA and std
        k: Standard deviation multiplier (typical 2.0)
        exit_at_middle: True = exit at middle band, False = exit at opposite band

    Returns:
        DataFrame with added columns: bb_middle, bb_upper, bb_lower, signal
    """
    df = df.copy()
    df["bb_middle"] = df["close"].rolling(period).mean()
    df["bb_std"] = df["close"].rolling(period).std()
    df["bb_upper"] = df["bb_middle"] + k * df["bb_std"]
    df["bb_lower"] = df["bb_middle"] - k * df["bb_std"]

    n = len(df)
    position = np.zeros(n, dtype=int)

    for i in range(period, n):
        prev_pos = position[i - 1]
        close_i = df["close"].iloc[i]

        if prev_pos == 0:
            if close_i < df["bb_lower"].iloc[i]:
                position[i] = 1
            elif close_i > df["bb_upper"].iloc[i]:
                position[i] = -1
            else:
                position[i] = 0
        elif prev_pos == 1:
            if exit_at_middle and close_i >= df["bb_middle"].iloc[i]:
                position[i] = 0
            elif not exit_at_middle and close_i >= df["bb_upper"].iloc[i]:
                position[i] = 0
            else:
                position[i] = 1
        elif prev_pos == -1:
            if exit_at_middle and close_i <= df["bb_middle"].iloc[i]:
                position[i] = 0
            elif not exit_at_middle and close_i <= df["bb_lower"].iloc[i]:
                position[i] = 0
            else:
                position[i] = -1

    df["signal"] = position
    df["signal"] = df["signal"].shift(1)
    return df


def bollinger_with_squeeze_filter(
    df: pd.DataFrame,
    period: int = 20,
    k: float = 2.0,
    squeeze_lookback: int = 100,
    squeeze_pct_max: float = 0.5,
) -> pd.DataFrame:
    """BB mean reversion với volatility squeeze filter.

    Trade only when BB width is in bottom squeeze_pct_max percentile
    (low volatility = stronger mean reversion regime).
    """
    df = bollinger_signals(df, period, k)
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_width_pct"] = df["bb_width"].rolling(squeeze_lookback).rank(pct=True)

    in_squeeze = df["bb_width_pct"] < squeeze_pct_max
    df["signal"] = np.where(in_squeeze, df["signal"], 0)
    return df


if __name__ == "__main__":
    np.random.seed(42)
    n = 1000
    prices = 1.10 + np.cumsum(np.random.randn(n) * 0.0008)
    df = pd.DataFrame({
        "close": prices,
        "high": prices + np.random.rand(n) * 0.0005,
        "low": prices - np.random.rand(n) * 0.0005,
    })
    df.index = pd.date_range("2024-01-01", periods=n, freq="h")

    result = bollinger_signals(df, period=20, k=2.0)
    print("Signal counts:")
    print(result["signal"].value_counts())
    print(f"\nFirst 5 trades signal changes:")
    changes = result[result["signal"].diff().abs() > 0.5]
    print(changes[["close", "bb_lower", "bb_upper", "signal"]].head())
