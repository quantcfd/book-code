"""
Connors RSI(2) mean reversion strategy.
QuantCFD Chapter 8.4
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI computation."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def connors_rsi_signals(
    df: pd.DataFrame,
    rsi_period: int = 2,
    oversold: float = 10,
    overbought: float = 90,
    exit_level: float = 50,
) -> pd.DataFrame:
    """
    Connors RSI(2) mean reversion strategy.

    Entry: RSI(2) < oversold (long) or > overbought (short)
    Exit: RSI(2) cross exit_level

    Args:
        df: DataFrame with 'close' column
        rsi_period: RSI period (default 2 = Connors)
        oversold: Long entry threshold (default 10)
        overbought: Short entry threshold (default 90)
        exit_level: Exit threshold (default 50)
    """
    df = df.copy()
    df["rsi"] = compute_rsi(df["close"], rsi_period)

    n = len(df)
    position = np.zeros(n, dtype=int)

    for i in range(rsi_period + 1, n):
        prev_pos = position[i - 1]
        rsi_i = df["rsi"].iloc[i]

        if prev_pos == 0:
            if rsi_i < oversold:
                position[i] = 1
            elif rsi_i > overbought:
                position[i] = -1
            else:
                position[i] = 0
        elif prev_pos == 1:
            position[i] = 0 if rsi_i >= exit_level else 1
        elif prev_pos == -1:
            position[i] = 0 if rsi_i <= exit_level else -1

    df["signal"] = position
    df["signal"] = df["signal"].shift(1)
    return df


def multi_tf_rsi(
    df_h1: pd.DataFrame,
    df_h4: pd.DataFrame,
    rsi_period_short: int = 2,
    rsi_period_long: int = 14,
) -> pd.DataFrame:
    """
    Multi-timeframe RSI confirmation.

    Trade RSI(2) on H1 only when RSI(14) on H4 has same bias.
    Long H1 only if RSI(14) H4 < 50 (room for revert up).
    Short H1 only if RSI(14) H4 > 50.
    """
    df_h1 = connors_rsi_signals(df_h1, rsi_period=rsi_period_short)
    df_h4 = df_h4.copy()
    df_h4["rsi_h4"] = compute_rsi(df_h4["close"], rsi_period_long)

    rsi_h4_resampled = df_h4["rsi_h4"].reindex(df_h1.index, method="ffill")

    long_ok = (df_h1["signal"] == 1) & (rsi_h4_resampled < 50)
    short_ok = (df_h1["signal"] == -1) & (rsi_h4_resampled > 50)

    df_h1["signal"] = np.where(long_ok, 1, np.where(short_ok, -1, 0))
    return df_h1


if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    prices = 1.10 + np.cumsum(np.random.randn(n) * 0.001)
    df = pd.DataFrame({"close": prices})
    df.index = pd.date_range("2024-01-01", periods=n, freq="h")

    result = connors_rsi_signals(df, rsi_period=2)
    print("RSI distribution:")
    print(result["rsi"].describe())
    print("\nSignal counts:")
    print(result["signal"].value_counts())
