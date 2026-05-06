"""
Z-score mean reversion strategy + half-life test.
QuantCFD Chapter 8.5 + 8.1 helper
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_half_life(series: pd.Series) -> float:
    """
    Compute mean reversion half-life via Ornstein-Uhlenbeck regression.

    Returns half-life in bars. np.inf means series is trending (no mean reversion).
    Half-life < 50 typically tradeable for mean reversion strategies.
    """
    spread_lag = series.shift(1)
    spread_diff = series - spread_lag

    df_reg = pd.concat([spread_diff, spread_lag], axis=1).dropna()
    df_reg.columns = ["diff", "lag"]

    if len(df_reg) < 30:
        return np.inf

    # Manual OLS: diff = alpha + beta * lag
    x = df_reg["lag"].values
    y = df_reg["diff"].values
    x_mean = x.mean()
    y_mean = y.mean()
    x_centered = x - x_mean
    y_centered = y - y_mean
    var_x = np.sum(x_centered ** 2)
    if var_x <= 0:
        return np.inf
    beta = np.sum(x_centered * y_centered) / var_x

    if beta >= 0:
        return np.inf

    half_life = -np.log(2) / beta
    return half_life


def hurst_exponent(series: pd.Series, lags=None) -> float:
    """
    Compute Hurst exponent via R/S analysis.
    H < 0.5: mean reverting
    H = 0.5: random walk
    H > 0.5: trending
    """
    if lags is None:
        lags = list(range(2, 20))

    series = pd.Series(series).dropna().values
    if len(series) < max(lags) + 10:
        return 0.5

    tau = []
    for lag in lags:
        diff = np.diff(series, n=lag)
        if len(diff) < 2 or np.std(diff) == 0:
            return 0.5
        tau.append(np.std(diff))

    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]


def zscore_signals(
    df: pd.DataFrame,
    lookback: int = 20,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
) -> pd.DataFrame:
    """
    Z-score mean reversion strategy.

    Args:
        df: DataFrame with 'close' column
        lookback: Rolling window for mean and std
        entry_z: Z-score threshold for entry (long below -entry_z, short above +entry_z)
        exit_z: Z-score threshold for exit (typical 0.0 or ±0.5)
    """
    df = df.copy()
    rolling_mean = df["close"].rolling(lookback).mean()
    rolling_std = df["close"].rolling(lookback).std()
    df["zscore"] = (df["close"] - rolling_mean) / rolling_std

    n = len(df)
    position = np.zeros(n, dtype=int)

    for i in range(lookback + 1, n):
        prev_pos = position[i - 1]
        z = df["zscore"].iloc[i]

        if pd.isna(z):
            position[i] = prev_pos
            continue

        if prev_pos == 0:
            if z < -entry_z:
                position[i] = 1
            elif z > entry_z:
                position[i] = -1
        elif prev_pos == 1:
            position[i] = 0 if z >= -exit_z else 1
        elif prev_pos == -1:
            position[i] = 0 if z <= exit_z else -1

    df["signal"] = position
    df["signal"] = df["signal"].shift(1)
    return df


def regime_classifier(df: pd.DataFrame, lookback: int = 100) -> pd.Series:
    """
    Classify market regime as 'trending', 'ranging', or 'transitional'.

    Uses Hurst exponent + ADX-style logic.
    """
    series = df["close"]
    n = len(df)
    regime = pd.Series(["transitional"] * n, index=df.index)

    for i in range(lookback, n):
        window = series.iloc[i - lookback:i]
        h = hurst_exponent(window)
        if h < 0.45:
            regime.iloc[i] = "ranging"
        elif h > 0.55:
            regime.iloc[i] = "trending"
        else:
            regime.iloc[i] = "transitional"
    return regime


if __name__ == "__main__":
    np.random.seed(42)
    n = 1000

    # Mean-reverting series (test mean reversion detection)
    mr_series = pd.Series(np.random.randn(n)).rolling(20).sum().fillna(0)
    print(f"Mean-reverting series:")
    print(f"  Half-life: {compute_half_life(mr_series):.1f} bars")
    print(f"  Hurst: {hurst_exponent(mr_series.values):.3f}")

    # Trending series
    trend_series = pd.Series(np.cumsum(np.random.randn(n) * 0.5 + 0.05))
    print(f"\nTrending series:")
    print(f"  Half-life: {compute_half_life(trend_series):.1f} bars")
    print(f"  Hurst: {hurst_exponent(trend_series.values):.3f}")
