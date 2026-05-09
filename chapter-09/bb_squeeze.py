"""
QuantCFD — Chương 9.6
Bollinger Bands Squeeze Breakout

Squeeze = BB inside Keltner (low volatility regime).
When squeeze releases (BB exits Keltner), trade direction
based on momentum.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def compute_atr_local(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def detect_squeeze(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_mult: float = 2.0,
    kc_period: int = 20,
    kc_mult: float = 1.5,
) -> pd.DataFrame:
    """
    Detect Bollinger Bands inside Keltner Channels (squeeze).

    Returns DataFrame with columns: bb_upper, bb_lower, kc_upper, kc_lower,
    in_squeeze, momentum.
    """
    df = df.copy()
    sma = df["close"].rolling(bb_period).mean()
    std = df["close"].rolling(bb_period).std()
    df["bb_upper"] = sma + bb_mult * std
    df["bb_lower"] = sma - bb_mult * std

    atr = compute_atr_local(df, kc_period)
    ema = df["close"].ewm(span=kc_period, adjust=False).mean()
    df["kc_upper"] = ema + kc_mult * atr
    df["kc_lower"] = ema - kc_mult * atr

    df["in_squeeze"] = (
        (df["bb_upper"] < df["kc_upper"]) & (df["bb_lower"] > df["kc_lower"])
    )
    df["momentum"] = df["close"] - df["close"].shift(10)
    return df


def bb_squeeze_strategy(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_mult: float = 2.0,
    kc_period: int = 20,
    kc_mult: float = 1.5,
    min_squeeze_bars: int = 5,
    use_volume: bool = False,
    volume_mult: float = 1.5,
) -> pd.DataFrame:
    """
    BB Squeeze breakout strategy.

    Trigger: squeeze (BB inside KC) for ≥ min_squeeze_bars,
    then squeeze releases. Trade direction by momentum sign.

    Args:
        df: OHLC (and optional volume).
        min_squeeze_bars: Minimum bars in squeeze before signal.
        use_volume: Require volume > volume_mult × avg.

    Returns:
        DataFrame with signal and position columns.
    """
    df = detect_squeeze(df, bb_period, bb_mult, kc_period, kc_mult)

    # Squeeze duration counter
    df["squeeze_count"] = (
        df["in_squeeze"].groupby(
            (~df["in_squeeze"]).cumsum()
        ).cumcount() + 1
    ) * df["in_squeeze"].astype(int)

    # Squeeze release: was in squeeze previous bar, now not
    df["squeeze_release"] = (
        df["in_squeeze"].shift(1) & (~df["in_squeeze"])
        & (df["squeeze_count"].shift(1) >= min_squeeze_bars)
    )

    # Signal direction by momentum (use prior bar values to avoid look-ahead)
    df["signal"] = 0
    long_signal = df["squeeze_release"] & (df["momentum"].shift(1) > 0)
    short_signal = df["squeeze_release"] & (df["momentum"].shift(1) < 0)

    # Volume filter
    if use_volume and "volume" in df.columns:
        avg_vol = df["volume"].rolling(20).mean().shift(1)
        vol_ok = df["volume"] > volume_mult * avg_vol
        long_signal = long_signal & vol_ok
        short_signal = short_signal & vol_ok

    df.loc[long_signal, "signal"] = 1
    df.loc[short_signal, "signal"] = -1
    df["signal"] = df["signal"].shift(1)  # avoid look-ahead

    # Build position with hold time limit
    position = 0
    bars_held = 0
    max_holding = 20
    positions = []
    for i in range(len(df)):
        sig = df["signal"].iloc[i] if not pd.isna(df["signal"].iloc[i]) else 0

        if position == 0:
            if sig == 1:
                position = 1
                bars_held = 1
            elif sig == -1:
                position = -1
                bars_held = 1
        else:
            bars_held += 1
            # Exit: cross BB middle, max holding, or new opposite signal
            mid = (df["bb_upper"].iloc[i] + df["bb_lower"].iloc[i]) / 2
            if (
                bars_held >= max_holding
                or (position == 1 and df["close"].iloc[i] < mid)
                or (position == -1 and df["close"].iloc[i] > mid)
            ):
                position = 0
                bars_held = 0

        positions.append(position)

    df["position"] = positions
    return df


def squeeze_metrics(
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
        "n_squeeze_releases": int(df["squeeze_release"].sum()),
        "n_in_squeeze_bars": int(df["in_squeeze"].sum()),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("BB Squeeze Breakout — Demo (BTCUSD daily synthetic)")
    print("=" * 70)

    np.random.seed(42)
    n = 1500
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    # Synthetic crypto-like with squeezes built in
    rets = np.random.randn(n) * 0.025
    # Inject some low-vol periods
    for start in [200, 500, 800, 1100, 1300]:
        end = start + 30
        rets[start:end] *= 0.3  # contraction
        if start + 35 < n:
            rets[start + 30:start + 35] *= 3  # explosion

    closes = 30000 * np.exp(np.cumsum(rets))
    daily_vol = np.abs(rets) + 0.005
    highs = closes * (1 + daily_vol)
    lows = closes * (1 - daily_vol)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    volumes = (1 + np.random.uniform(0, 1, n)) * 1e6
    # Higher volume on big move days
    volumes[np.abs(rets) > 0.05] *= 2

    df = pd.DataFrame({
        "open": opens,
        "high": np.maximum(np.maximum(opens, closes), highs),
        "low": np.minimum(np.minimum(opens, closes), lows),
        "close": closes,
        "volume": volumes,
    }, index=dates)

    print(f"\nData: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Days: {len(df)}")

    # Run BB squeeze strategy
    df_result = bb_squeeze_strategy(
        df, bb_period=20, bb_mult=2.0,
        kc_period=20, kc_mult=1.5,
        min_squeeze_bars=5,
        use_volume=True, volume_mult=1.5,
    )

    m = squeeze_metrics(df_result, periods_per_year=252)

    if "error" in m:
        print(f"\n{m['error']}")
    else:
        print(f"\n{'─' * 70}")
        print(f"Bars in squeeze:    {m['n_in_squeeze_bars']} "
              f"({m['n_in_squeeze_bars']/len(df)*100:.1f}%)")
        print(f"Squeeze releases:   {m['n_squeeze_releases']}")
        print(f"Total trades:       {m['n_trades']}")
        print(f"Sharpe:             {m['sharpe']:.3f}")
        print(f"CAGR:               {m['cagr']*100:.2f}%")
        print(f"Max DD:             {m['max_dd']*100:.2f}%")

    print(f"\nLessons:")
    print(f"  - Squeeze releases ~5-10/năm typical")
    print(f"  - Volume filter reduces trades but improves quality")
    print(f"  - Crypto best market for BB squeeze (high vol baseline)")
    print(f"  - Direction filter (momentum) critical — squeeze itself ambiguous")
