"""
QuantCFD — Chương 9.5
Keltner Channel Breakout

Keltner = EMA ± atr_mult × ATR.
Smoother than Bollinger (uses ATR not std).
Breakout signals when close beyond bands.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — standard EMA-smoothed."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr


def keltner_channels(
    df: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 14,
    atr_mult: float = 2.0,
) -> pd.DataFrame:
    """Compute Keltner channel bands. Bands shifted to avoid look-ahead."""
    df = df.copy()
    df["ema"] = df["close"].ewm(span=ema_period, adjust=False).mean()
    df["atr"] = compute_atr(df, atr_period)
    df["upper"] = (df["ema"] + atr_mult * df["atr"]).shift(1)
    df["lower"] = (df["ema"] - atr_mult * df["atr"]).shift(1)
    df["mid"] = df["ema"].shift(1)
    return df


def keltner_breakout(
    df: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 14,
    atr_mult: float = 2.0,
    exit_method: str = "middle",
) -> pd.DataFrame:
    """
    Keltner breakout strategy.

    Long when close > upper band.
    Short when close < lower band.
    Exit: cross middle band (default) or opposite band.

    Args:
        df: OHLC DataFrame.
        exit_method: 'middle' or 'opposite'.

    Returns:
        DataFrame with position column added.
    """
    df = keltner_channels(df, ema_period, atr_period, atr_mult)

    position = 0
    positions = []
    for i in range(len(df)):
        row = df.iloc[i]
        if pd.isna(row["upper"]):
            positions.append(0)
            continue

        if position == 0:
            if row["close"] > row["upper"]:
                position = 1
            elif row["close"] < row["lower"]:
                position = -1
        elif position == 1:
            exit_level = row["lower"] if exit_method == "opposite" else row["mid"]
            if row["close"] < exit_level:
                position = 0
        elif position == -1:
            exit_level = row["upper"] if exit_method == "opposite" else row["mid"]
            if row["close"] > exit_level:
                position = 0

        positions.append(position)

    df["position"] = positions
    return df


def keltner_metrics(
    df_with_position: pd.DataFrame,
    cost: float = 0.0008,
    periods_per_year: int = 252,
) -> dict:
    """Compute strategy metrics from positions."""
    df = df_with_position.copy()
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

    # Trade-level stats
    df["trade_id"] = (df["pos_change"].cumsum() / 2).astype(int)
    trade_pnl = df.groupby("trade_id")["strat_ret"].sum()
    trade_pnl = trade_pnl[trade_pnl != 0]
    win_rate = (trade_pnl > 0).mean() if len(trade_pnl) > 0 else 0

    return {
        "sharpe": sharpe,
        "cagr": cagr,
        "max_dd": max_dd,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "equity_curve": eq,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Keltner Channel Breakout — Demo (XAUUSD H4 synthetic)")
    print("=" * 70)

    np.random.seed(42)
    n = 5000
    dates = pd.date_range("2020-01-01", periods=n, freq="4h")

    # Synthetic with regime changes
    rets = np.zeros(n)
    for i in range(0, n, 500):
        end = min(i + 500, n)
        regime = np.random.choice(["trend_up", "range", "trend_down"])
        if regime == "trend_up":
            rets[i:end] = np.random.randn(end - i) * 0.005 + 0.0003
        elif regime == "trend_down":
            rets[i:end] = np.random.randn(end - i) * 0.005 - 0.0003
        else:
            # Mean-reverting range
            for j in range(i, end):
                rets[j] = np.random.randn() * 0.005 - 0.1 * (rets[j-1] if j > 0 else 0)

    closes = 2000 * np.exp(np.cumsum(rets))
    daily_vol = np.random.uniform(0.003, 0.015, n)
    highs = closes * (1 + daily_vol)
    lows = closes * (1 - daily_vol * np.random.uniform(0.8, 1.2, n))
    opens = np.roll(closes, 1) * (1 + np.random.randn(n) * 0.003)
    opens[0] = closes[0] * 0.999

    df = pd.DataFrame({
        "open": opens,
        "high": np.maximum(np.maximum(opens, closes), highs),
        "low": np.minimum(np.minimum(opens, closes), lows),
        "close": closes,
    }, index=dates)

    print(f"\nData: {df.index[0]} → {df.index[-1]}")
    print(f"Bars: {len(df)}")

    # Run Keltner breakout
    df_result = keltner_breakout(df, ema_period=20, atr_period=14, atr_mult=2.0)
    m = keltner_metrics(df_result, periods_per_year=252 * 6)  # H4 = 6 bars/day

    print(f"\n{'─' * 70}")
    print("Keltner Breakout (20, 14, 2.0):")
    print(f"  Sharpe:      {m['sharpe']:.3f}")
    print(f"  CAGR:        {m['cagr']*100:.2f}%")
    print(f"  Max DD:      {m['max_dd']*100:.2f}%")
    print(f"  Trades:      {m['n_trades']}")
    print(f"  Win rate:    {m['win_rate']*100:.1f}%")

    # Parameter sensitivity
    print(f"\n{'─' * 70}")
    print("Parameter sensitivity:")
    print(f"{'─' * 70}")
    print(f"{'Params':<25} {'Sharpe':>8} {'Trades':>8} {'WinRate':>8}")
    for ema_p in [10, 20, 30]:
        for mult in [1.5, 2.0, 2.5]:
            df_r = keltner_breakout(df, ema_period=ema_p, atr_mult=mult)
            m_r = keltner_metrics(df_r, periods_per_year=252 * 6)
            if "error" not in m_r:
                print(f"  ({ema_p}, 14, {mult}){'':<10} "
                      f"{m_r['sharpe']:>8.3f} {m_r['n_trades']:>8} "
                      f"{m_r['win_rate']*100:>7.1f}%")

    print(f"\nLessons:")
    print(f"  - (20, 14, 2.0) sweet spot for most CFD instruments")
    print(f"  - Lower mult = more trades, lower quality")
    print(f"  - Higher mult = fewer trades, higher quality")
    print(f"  - Tune via WFA, không curve-fit on single backtest")
