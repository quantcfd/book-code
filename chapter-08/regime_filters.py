"""
Regime filters for mean reversion strategies.
QuantCFD Chapter 8.8
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR."""
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False).mean()


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ADX — average directional index. ADX > 25 = trending."""
    high = df["high"]
    low = df["low"]

    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=df.index)

    atr = compute_atr(df, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx.fillna(0)


def adx_filter(df: pd.DataFrame, max_adx: float = 25) -> pd.Series:
    """Boolean filter: True when ADX < max_adx (range-bound regime)."""
    df = df.copy()
    df["adx"] = compute_adx(df)
    return df["adx"] < max_adx


def vol_regime_filter(
    df: pd.DataFrame,
    vol_lookback_short: int = 20,
    vol_lookback_long: int = 252,
    max_pct: float = 70,
) -> pd.Series:
    """
    Filter: True when realized vol percentile <= max_pct (low vol regime).

    Mean reversion strategies work better in low vol regimes.
    """
    realized_vol = df["close"].pct_change().rolling(vol_lookback_short).std() * np.sqrt(252)
    vol_pct = realized_vol.rolling(vol_lookback_long).rank(pct=True) * 100
    return vol_pct <= max_pct


def apply_regime_filters(
    df: pd.DataFrame,
    base_strategy_signals: pd.Series,
    use_adx: bool = True,
    adx_max: float = 25,
    use_vol: bool = True,
    vol_max_pct: float = 70,
) -> pd.Series:
    """
    Combine multiple regime filters with base strategy signals.

    Returns filtered signals (zeros where filters disagree).
    """
    filtered = base_strategy_signals.copy()

    if use_adx:
        adx_ok = adx_filter(df, max_adx=adx_max)
        filtered = filtered.where(adx_ok, 0)

    if use_vol:
        vol_ok = vol_regime_filter(df, max_pct=vol_max_pct)
        filtered = filtered.where(vol_ok, 0)

    return filtered


def in_news_window(
    timestamp: pd.Timestamp,
    news_times_et: list[str],
    window_minutes: int = 30,
) -> bool:
    """
    Check if timestamp is within news window.
    news_times_et: list of HH:MM strings in ET timezone
    """
    timestamp_et = timestamp.tz_convert("America/New_York") if timestamp.tz else timestamp

    for news_time_str in news_times_et:
        hour, minute = map(int, news_time_str.split(":"))
        news_time = timestamp_et.replace(hour=hour, minute=minute, second=0)
        diff_minutes = abs((timestamp_et - news_time).total_seconds() / 60)
        if diff_minutes < window_minutes:
            return True
    return False


if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    prices = 1.10 + np.cumsum(np.random.randn(n) * 0.001)
    df = pd.DataFrame({
        "close": prices,
        "high": prices + np.random.rand(n) * 0.0005,
        "low": prices - np.random.rand(n) * 0.0005,
    })
    df.index = pd.date_range("2024-01-01", periods=n, freq="h")

    adx = compute_adx(df)
    print(f"ADX stats: mean={adx.mean():.1f}, max={adx.max():.1f}")
    print(f"% bars with ADX < 25: {(adx < 25).mean() * 100:.1f}%")

    vol_ok = vol_regime_filter(df)
    print(f"% bars in low vol regime: {vol_ok.mean() * 100:.1f}%")
