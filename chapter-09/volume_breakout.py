"""
QuantCFD — Chương 9.7
Volume + Price Breakout

Breakout của resistance/support với volume confirmation.
Range expansion alternative cho FX/XAU không có reliable volume.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def volume_breakout(
    df: pd.DataFrame,
    lookback: int = 20,
    volume_mult: float = 1.5,
    target_mult: float = 2.0,
    cost: float = 0.0008,
) -> pd.DataFrame:
    """
    Price breakout với volume confirmation.

    Long: close > rolling N-day high AND volume > volume_mult × avg.
    Short: close < rolling N-day low AND volume > volume_mult × avg.

    Args:
        df: OHLC + volume DataFrame.
        lookback: Resistance/support window.
        volume_mult: Volume multiplier threshold.
        target_mult: TP as multiple of pre-breakout range.

    Returns:
        DataFrame with signal and position.
    """
    df = df.copy()
    if "volume" not in df.columns:
        raise ValueError("DataFrame must have 'volume' column")

    df["resistance"] = df["high"].rolling(lookback).max().shift(1)
    df["support"] = df["low"].rolling(lookback).min().shift(1)
    df["avg_volume"] = df["volume"].rolling(lookback).mean().shift(1)
    df["range_size"] = df["resistance"] - df["support"]

    # Breakout signals
    long_breakout = (
        (df["close"] > df["resistance"])
        & (df["volume"] > volume_mult * df["avg_volume"])
    )
    short_breakout = (
        (df["close"] < df["support"])
        & (df["volume"] > volume_mult * df["avg_volume"])
    )

    df["signal"] = 0
    df.loc[long_breakout, "signal"] = 1
    df.loc[short_breakout, "signal"] = -1
    df["signal"] = df["signal"].shift(1)

    # Build position
    position = 0
    entry_price = None
    target_price = None
    stop_price = None
    positions = []
    for i in range(len(df)):
        row = df.iloc[i]
        sig = row["signal"] if not pd.isna(row["signal"]) else 0

        if position == 0:
            if sig == 1 and not pd.isna(row["range_size"]):
                position = 1
                entry_price = row["close"]
                stop_price = entry_price - row["range_size"] / 2
                target_price = entry_price + target_mult * row["range_size"]
            elif sig == -1 and not pd.isna(row["range_size"]):
                position = -1
                entry_price = row["close"]
                stop_price = entry_price + row["range_size"] / 2
                target_price = entry_price - target_mult * row["range_size"]
        elif position == 1:
            if row["low"] <= stop_price or row["high"] >= target_price:
                position = 0
                entry_price = stop_price = target_price = None
        elif position == -1:
            if row["high"] >= stop_price or row["low"] <= target_price:
                position = 0
                entry_price = stop_price = target_price = None

        positions.append(position)

    df["position"] = positions
    return df


def range_expansion_breakout(
    df: pd.DataFrame,
    lookback: int = 20,
    range_mult: float = 1.5,
    target_mult: float = 2.0,
) -> pd.DataFrame:
    """
    Alternative cho FX/XAU không có volume reliable.
    Filter: breakout candle range > range_mult × ATR.
    """
    df = df.copy()
    df["resistance"] = df["high"].rolling(lookback).max().shift(1)
    df["support"] = df["low"].rolling(lookback).min().shift(1)
    df["range_size"] = df["resistance"] - df["support"]

    # ATR
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["candle_range"] = df["high"] - df["low"]
    df["range_ratio"] = df["candle_range"] / df["atr"].shift(1)

    long_breakout = (
        (df["close"] > df["resistance"])
        & (df["range_ratio"] > range_mult)
    )
    short_breakout = (
        (df["close"] < df["support"])
        & (df["range_ratio"] > range_mult)
    )

    df["signal"] = 0
    df.loc[long_breakout, "signal"] = 1
    df.loc[short_breakout, "signal"] = -1
    df["signal"] = df["signal"].shift(1)

    # Position management with range_size targets
    position = 0
    entry_price = None
    target_price = None
    stop_price = None
    positions = []
    for i in range(len(df)):
        row = df.iloc[i]
        sig = row["signal"] if not pd.isna(row["signal"]) else 0

        if position == 0:
            if sig == 1 and not pd.isna(row["range_size"]):
                position = 1
                entry_price = row["close"]
                stop_price = entry_price - row["range_size"] / 2
                target_price = entry_price + target_mult * row["range_size"]
            elif sig == -1 and not pd.isna(row["range_size"]):
                position = -1
                entry_price = row["close"]
                stop_price = entry_price + row["range_size"] / 2
                target_price = entry_price - target_mult * row["range_size"]
        elif position == 1:
            if row["low"] <= stop_price or row["high"] >= target_price:
                position = 0
        elif position == -1:
            if row["high"] >= stop_price or row["low"] <= target_price:
                position = 0

        positions.append(position)

    df["position"] = positions
    return df


def breakout_metrics(
    df_with_pos: pd.DataFrame,
    cost: float = 0.0008,
    periods_per_year: int = 252,
) -> dict:
    df = df_with_pos.copy()
    df["ret"] = df["close"].pct_change()
    df["pos_change"] = df["position"].diff().abs().fillna(0)
    df["strat_ret"] = df["position"] * df["ret"] - df["pos_change"] * cost
    clean = df["strat_ret"].dropna()

    if len(clean) < 30 or clean.std() == 0:
        return {"error": "insufficient data"}

    sharpe = (clean.mean() / clean.std()) * np.sqrt(periods_per_year)
    cagr = (1 + clean.mean()) ** periods_per_year - 1
    eq = (1 + clean).cumprod()
    max_dd = (eq / eq.cummax() - 1).min()
    n_trades = int(df["pos_change"].sum() / 2)

    return {
        "sharpe": sharpe,
        "cagr": cagr,
        "max_dd": max_dd,
        "n_trades": n_trades,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Volume + Price / Range Expansion Breakout — Demo")
    print("=" * 70)

    np.random.seed(42)
    n = 1500
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    rets = np.random.randn(n) * 0.015
    closes = 100 * np.exp(np.cumsum(rets))
    daily_vol = np.abs(rets) + 0.005
    highs = closes * (1 + daily_vol)
    lows = closes * (1 - daily_vol)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    volumes = (1 + np.random.uniform(0, 1, n)) * 1e6
    volumes[np.abs(rets) > 0.025] *= 2

    df = pd.DataFrame({
        "open": opens,
        "high": np.maximum(np.maximum(opens, closes), highs),
        "low": np.minimum(np.minimum(opens, closes), lows),
        "close": closes, "volume": volumes,
    }, index=dates)

    print(f"\nData: {len(df)} days")

    # Volume breakout
    print(f"\n{'─' * 70}")
    print("Volume + price breakout:")
    df_v = volume_breakout(df, lookback=20, volume_mult=1.5)
    m_v = breakout_metrics(df_v)
    if "error" not in m_v:
        print(f"  Sharpe:    {m_v['sharpe']:.3f}")
        print(f"  CAGR:      {m_v['cagr']*100:.2f}%")
        print(f"  Max DD:    {m_v['max_dd']*100:.2f}%")
        print(f"  Trades:    {m_v['n_trades']}")

    # Range expansion (FX/XAU alternative)
    print(f"\n{'─' * 70}")
    print("Range expansion breakout (cho FX/XAU):")
    df_r = range_expansion_breakout(df, lookback=20, range_mult=1.5)
    m_r = breakout_metrics(df_r)
    if "error" not in m_r:
        print(f"  Sharpe:    {m_r['sharpe']:.3f}")
        print(f"  CAGR:      {m_r['cagr']*100:.2f}%")
        print(f"  Max DD:    {m_r['max_dd']*100:.2f}%")
        print(f"  Trades:    {m_r['n_trades']}")

    print(f"\nLessons:")
    print(f"  - Volume breakout ideal cho crypto (reliable volume)")
    print(f"  - Range expansion alternative cho FX (tick volume unreliable)")
    print(f"  - Higher volume_mult/range_mult → fewer trades, better quality")
