"""
QuantCFD — Chương 9.10
Regime Classifier cho Vol Breakout

Classify market regime as IDEAL / POSSIBLE / AVOID for vol breakout.

Metrics:
- ATR percentile (60-day rolling)
- ADX (trend strength)
- Bollinger Bandwidth percentile
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def compute_atr_pct(
    df: pd.DataFrame, atr_period: int = 14, lookback: int = 60,
) -> pd.Series:
    """ATR percentile over lookback window."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / atr_period, adjust=False).mean()
    return atr.rolling(lookback).rank(pct=True)


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Simplified ADX calculation."""
    high_diff = df["high"].diff()
    low_diff = -df["low"].diff()

    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx.fillna(0)


def compute_bb_width_pct(
    df: pd.DataFrame, bb_period: int = 20, lookback: int = 60,
) -> pd.Series:
    """Bollinger Bandwidth percentile."""
    sma = df["close"].rolling(bb_period).mean()
    std = df["close"].rolling(bb_period).std()
    bb_width = 4 * std / sma
    return bb_width.rolling(lookback).rank(pct=True)


def regime_classifier_breakout(
    df: pd.DataFrame,
    atr_period: int = 14,
    adx_period: int = 14,
    bb_period: int = 20,
    lookback: int = 60,
) -> pd.DataFrame:
    """
    Classify regime as IDEAL / POSSIBLE / AVOID for vol breakout.

    IDEAL:    ATR pct 30-50%, ADX < 25, BB width pct < 40%
              → contraction setup, ranging market
    POSSIBLE: ATR pct 50-70%, ADX < 30
              → moderate vol, trade with caution
    AVOID:    ATR pct > 70% OR ADX > 30 OR BB width pct > 60%
              → extreme vol or strong trend, vol BO unreliable

    Returns DataFrame with metrics and regime label.
    """
    df = df.copy()
    df["atr_pct"] = compute_atr_pct(df, atr_period, lookback).shift(1)
    df["adx"] = compute_adx(df, adx_period).shift(1)
    df["bb_width_pct"] = compute_bb_width_pct(df, bb_period, lookback).shift(1)

    ideal = (
        (df["atr_pct"] >= 0.30) & (df["atr_pct"] <= 0.50)
        & (df["adx"] < 25)
        & (df["bb_width_pct"] < 0.40)
    )
    avoid = (
        (df["atr_pct"] > 0.70)
        | (df["adx"] > 30)
        | (df["bb_width_pct"] > 0.60)
    )

    df["regime"] = "POSSIBLE"
    df.loc[ideal, "regime"] = "IDEAL"
    df.loc[avoid, "regime"] = "AVOID"
    return df


def filter_signals_by_regime(
    signals: pd.Series, df_with_regime: pd.DataFrame,
    allowed_regimes=("IDEAL", "POSSIBLE"),
) -> pd.Series:
    """Zero-out signals when regime not in allowed list."""
    common = signals.index.intersection(df_with_regime.index)
    filtered = signals.copy()
    bad_regime = ~df_with_regime.loc[common, "regime"].isin(allowed_regimes)
    filtered.loc[common[bad_regime]] = 0
    return filtered


if __name__ == "__main__":
    print("=" * 70)
    print("Regime Classifier for Vol Breakout — Demo")
    print("=" * 70)

    np.random.seed(42)
    n = 1500
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    # Synthetic with regime variations
    rets = np.zeros(n)
    for i in range(0, n, 200):
        end = min(i + 200, n)
        regime_type = np.random.choice(["calm", "trending", "volatile"])
        if regime_type == "calm":
            rets[i:end] = np.random.randn(end - i) * 0.005
        elif regime_type == "trending":
            rets[i:end] = np.random.randn(end - i) * 0.012 + 0.001
        else:
            rets[i:end] = np.random.randn(end - i) * 0.025

    closes = 1500 * np.exp(np.cumsum(rets))
    daily_vol = np.abs(rets) + 0.003
    highs = closes * (1 + daily_vol)
    lows = closes * (1 - daily_vol)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]

    df = pd.DataFrame({
        "open": opens,
        "high": np.maximum(np.maximum(opens, closes), highs),
        "low": np.minimum(np.minimum(opens, closes), lows),
        "close": closes,
    }, index=dates)

    df_regime = regime_classifier_breakout(df)

    # Distribution
    regime_counts = df_regime["regime"].value_counts()
    print(f"\nRegime distribution:")
    for regime, count in regime_counts.items():
        print(f"  {regime:<10}: {count} bars ({count/len(df)*100:.1f}%)")

    # Sample bars from each regime
    print(f"\nSample bars from each regime:")
    for regime in ["IDEAL", "POSSIBLE", "AVOID"]:
        bars = df_regime[df_regime["regime"] == regime]
        if len(bars) > 0:
            sample = bars.iloc[len(bars) // 2]
            print(f"\n  {regime} (e.g. {sample.name.date()}):")
            print(f"    ATR pct:       {sample['atr_pct']:.2f}")
            print(f"    ADX:           {sample['adx']:.1f}")
            print(f"    BB width pct:  {sample['bb_width_pct']:.2f}")
