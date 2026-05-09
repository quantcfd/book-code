"""
QuantCFD — Chương 9.4.5
Volatility Statistical Signals — Contraction Score

4 metrics combined into 0-100 score:
1. ATR ratio (current ATR / historical ATR)
2. Bollinger Bandwidth percentile
3. NR7 detection
4. IDnr7 detection (inside day + NR7)
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def atr_contraction_ratio(
    df: pd.DataFrame, atr_short: int = 14, atr_long: int = 60,
) -> pd.Series:
    """Tỷ số ATR ngắn / ATR dài. < 0.6 = strong contraction."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_s = tr.ewm(alpha=1 / atr_short, adjust=False).mean()
    atr_l = tr.ewm(alpha=1 / atr_long, adjust=False).mean()
    return (atr_s / atr_l).shift(1)


def bb_width_percentile(
    df: pd.DataFrame, bb_period: int = 20, lookback: int = 60,
) -> pd.Series:
    """Bollinger Bandwidth percentile rolling lookback."""
    sma = df["close"].rolling(bb_period).mean()
    std = df["close"].rolling(bb_period).std()
    bb_width = 4 * std / sma
    return bb_width.rolling(lookback).rank(pct=True).shift(1)


def detect_nr7(df: pd.DataFrame, lookback: int = 7) -> pd.Series:
    """NR7 = today range narrowest in lookback bars."""
    rng = df["high"] - df["low"]
    rolling_min = rng.rolling(lookback).min()
    rolling_count = rng.rolling(lookback).count()
    return (rng == rolling_min) & (rolling_count == lookback)


def detect_idnr7(df: pd.DataFrame, lookback: int = 7) -> pd.Series:
    """IDnr7 = inside day AND NR7."""
    nr7 = detect_nr7(df, lookback)
    inside = (
        (df["high"] < df["high"].shift(1))
        & (df["low"] > df["low"].shift(1))
    )
    return nr7 & inside


def contraction_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite 0-100 score for vol breakout setup quality.
    > 70 = strong setup, < 30 = no setup.

    Returns DataFrame with score components and total.
    """
    df = df.copy()
    df["atr_ratio"] = atr_contraction_ratio(df)
    df["bb_pct"] = bb_width_percentile(df)
    df["nr7"] = detect_nr7(df)
    df["idnr7"] = detect_idnr7(df)

    # Score components (0-25 each)
    df["score_atr"] = (1 - df["atr_ratio"]).clip(0, 1) * 25
    df["score_bbw"] = (1 - df["bb_pct"]).clip(0, 1) * 25
    df["score_nr7"] = df["nr7"].astype(int) * 25
    df["score_idnr7"] = df["idnr7"].astype(int) * 25

    df["contraction_score"] = (
        df["score_atr"].fillna(0)
        + df["score_bbw"].fillna(0)
        + df["score_nr7"].fillna(0)
        + df["score_idnr7"].fillna(0)
    )
    return df


def setup_quality_filter(
    df: pd.DataFrame, min_score: float = 50.0,
) -> pd.Series:
    """
    Return boolean series: True where setup quality is sufficient.

    Args:
        df: DataFrame with contraction_score column.
        min_score: Minimum score threshold (typically 50-70).

    Returns:
        Series of bool.
    """
    if "contraction_score" not in df.columns:
        df = contraction_score(df)
    return df["contraction_score"] >= min_score


if __name__ == "__main__":
    print("=" * 70)
    print("Contraction Score — Demo")
    print("=" * 70)

    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2022-01-01", periods=n, freq="D")

    # Synthetic with explicit contraction periods
    rets = np.random.randn(n) * 0.015
    # Inject 4 contraction periods
    contraction_starts = [150, 400, 650, 850]
    for cs in contraction_starts:
        rets[cs:cs + 25] *= 0.3

    closes = 1500 * np.exp(np.cumsum(rets))
    daily_vol = np.abs(rets) * 1.5 + 0.003
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

    df_score = contraction_score(df)

    # Distribution
    print(f"\nContraction score distribution:")
    print(f"  Min:    {df_score['contraction_score'].min():.1f}")
    print(f"  25th:   {df_score['contraction_score'].quantile(0.25):.1f}")
    print(f"  Median: {df_score['contraction_score'].median():.1f}")
    print(f"  75th:   {df_score['contraction_score'].quantile(0.75):.1f}")
    print(f"  Max:    {df_score['contraction_score'].max():.1f}")

    # Bars in different score ranges
    print(f"\nBars per score bucket:")
    buckets = [(0, 25), (25, 50), (50, 75), (75, 100)]
    for low, high in buckets:
        count = (
            (df_score["contraction_score"] >= low)
            & (df_score["contraction_score"] < high)
        ).sum()
        print(f"  [{low:>3}, {high:>3}): {count} bars ({count/len(df)*100:.1f}%)")

    # NR7 days detected
    print(f"\nNR7 days:        {df_score['nr7'].sum()} "
          f"({df_score['nr7'].sum()/len(df)*100:.1f}%)")
    print(f"IDnr7 days:      {df_score['idnr7'].sum()} "
          f"({df_score['idnr7'].sum()/len(df)*100:.1f}%)")
    print(f"High score (≥70): {(df_score['contraction_score'] >= 70).sum()}")

    # Sample top 5 setups
    print(f"\nTop 5 contraction setups:")
    top = df_score.nlargest(5, "contraction_score")[
        ["close", "atr_ratio", "bb_pct", "nr7", "idnr7", "contraction_score"]
    ]
    for date, row in top.iterrows():
        print(f"  {date.date()}: "
              f"score={row['contraction_score']:.0f}  "
              f"atr={row['atr_ratio']:.2f}  "
              f"bb_pct={row['bb_pct']:.2f}  "
              f"nr7={row['nr7']}  idnr7={row['idnr7']}")
