"""
QuantCFD — Chương 9.4
NR7 / NR4 / IDnr7 — Toby Crabel narrow range breakout

Insight: today range narrowest in N days predicts expansion next day.
Trade direction of breakout.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def detect_narrow_range(df: pd.DataFrame, lookback: int = 7) -> pd.Series:
    """
    Detect narrow range days (today range narrowest in lookback bars).

    Args:
        df: Daily OHLC DataFrame.
        lookback: Window (7 for NR7, 4 for NR4).

    Returns:
        Series of bool, True if today is NR-N day.
    """
    df = df.copy()
    df["range"] = df["high"] - df["low"]
    rolling_min = df["range"].rolling(lookback).min()
    rolling_count = df["range"].rolling(lookback).count()
    is_nr = (df["range"] == rolling_min) & (rolling_count == lookback)
    return is_nr


def detect_inside_day(df: pd.DataFrame) -> pd.Series:
    """
    Detect inside day (today high < yesterday high AND today low > yesterday low).
    """
    df = df.copy()
    return (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))


def detect_idnr(df: pd.DataFrame, lookback: int = 7) -> pd.Series:
    """
    Detect IDnr (Inside Day + NR) — strongest signal in Crabel framework.
    """
    nr = detect_narrow_range(df, lookback)
    inside = detect_inside_day(df)
    return nr & inside


def nr_breakout_strategy(
    df: pd.DataFrame,
    lookback: int = 7,
    target_mult: float = 1.5,
    use_idnr: bool = False,
) -> pd.DataFrame:
    """
    Trade breakout direction next day after NR setup.

    Long when next day high > setup day high.
    Short when next day low < setup day low.
    Stop: opposite side of setup range.
    Target: target_mult × setup range size.

    Args:
        df: Daily OHLC.
        lookback: NR lookback (7 = NR7, 4 = NR4).
        target_mult: Take-profit multiple.
        use_idnr: If True, only trade IDnr (stronger but rare signal).

    Returns:
        DataFrame of trades.
    """
    df = df.copy()
    df["range"] = df["high"] - df["low"]

    if use_idnr:
        df["setup"] = detect_idnr(df, lookback)
    else:
        df["setup"] = detect_narrow_range(df, lookback)

    # Setup info from PREVIOUS day
    df["setup_high"] = df["high"].shift(1).where(df["setup"].shift(1))
    df["setup_low"] = df["low"].shift(1).where(df["setup"].shift(1))
    df["setup_range"] = df["range"].shift(1).where(df["setup"].shift(1))

    trades = []
    for idx, row in df.iterrows():
        if pd.isna(row["setup_high"]):
            continue

        sh, sl, sr = row["setup_high"], row["setup_low"], row["setup_range"]

        # Long breakout
        if row["high"] > sh:
            entry = sh
            stop = sl
            target = entry + target_mult * sr
            if row["low"] <= stop:  # stop hit
                exit_price = stop
                result = "stop"
            elif row["high"] >= target:
                exit_price = target
                result = "target"
            else:
                exit_price = row["close"]
                result = "eod"
            trades.append({
                "date": idx, "side": "long",
                "entry": entry, "exit": exit_price,
                "pnl": exit_price - entry,
                "result": result, "setup_range": sr,
            })

        # Short breakout
        elif row["low"] < sl:
            entry = sl
            stop = sh
            target = entry - target_mult * sr
            if row["high"] >= stop:
                exit_price = stop
                result = "stop"
            elif row["low"] <= target:
                exit_price = target
                result = "target"
            else:
                exit_price = row["close"]
                result = "eod"
            trades.append({
                "date": idx, "side": "short",
                "entry": entry, "exit": exit_price,
                "pnl": entry - exit_price,
                "result": result, "setup_range": sr,
            })

    return pd.DataFrame(trades)


def nr_metrics(trades_df: pd.DataFrame) -> dict:
    """Summary metrics for NR strategy."""
    if len(trades_df) == 0:
        return {"error": "no trades"}

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]

    return {
        "total_trades": len(trades_df),
        "wins": len(wins),
        "win_rate": len(wins) / len(trades_df),
        "avg_win": wins["pnl"].mean() if len(wins) > 0 else 0,
        "avg_loss": losses["pnl"].mean() if len(losses) > 0 else 0,
        "total_pnl": trades_df["pnl"].sum(),
        "results": trades_df["result"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("NR7 / NR4 / IDnr7 — Demo (XAUUSD daily synthetic)")
    print("=" * 70)

    np.random.seed(42)
    n = 1500
    dates = pd.date_range("2018-01-01", periods=n, freq="D")

    # Synthetic daily OHLC with mixed regimes
    rets = np.random.randn(n) * 0.012
    closes = 1500 * np.exp(np.cumsum(rets))

    # Realistic OHLC: high/low with some volatility around close
    daily_vol = np.random.uniform(0.005, 0.025, n)
    highs = closes * (1 + daily_vol)
    lows = closes * (1 - daily_vol * np.random.uniform(0.8, 1.2, n))
    opens = closes * (1 + np.random.randn(n) * 0.005)

    df = pd.DataFrame({
        "open": opens,
        "high": np.maximum(np.maximum(opens, closes), highs),
        "low": np.minimum(np.minimum(opens, closes), lows),
        "close": closes,
    }, index=dates)

    print(f"\nData: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Days: {len(df)}")

    # Detect NR7 frequency
    nr7_days = detect_narrow_range(df, 7)
    nr4_days = detect_narrow_range(df, 4)
    idnr7_days = detect_idnr(df, 7)
    print(f"\nNR7 days:    {nr7_days.sum()} ({nr7_days.sum()/len(df)*100:.1f}%)")
    print(f"NR4 days:    {nr4_days.sum()} ({nr4_days.sum()/len(df)*100:.1f}%)")
    print(f"IDnr7 days:  {idnr7_days.sum()} ({idnr7_days.sum()/len(df)*100:.1f}%)")

    # Run NR7 breakout strategy
    print(f"\n{'─' * 70}")
    print("NR7 Breakout Strategy:")
    print(f"{'─' * 70}")
    trades_nr7 = nr_breakout_strategy(df, lookback=7, target_mult=1.5)
    m = nr_metrics(trades_nr7)
    if "error" not in m:
        print(f"  Total trades:  {m['total_trades']}")
        print(f"  Win rate:      {m['win_rate']*100:.1f}%")
        print(f"  Avg win:       {m['avg_win']:.2f}")
        print(f"  Avg loss:      {m['avg_loss']:.2f}")
        print(f"  Total P&L:     {m['total_pnl']:.2f}")
        print(f"  Results:       {m['results']}")

    # Compare NR4
    print(f"\n{'─' * 70}")
    print("NR4 Breakout Strategy (more frequent):")
    print(f"{'─' * 70}")
    trades_nr4 = nr_breakout_strategy(df, lookback=4, target_mult=1.5)
    m4 = nr_metrics(trades_nr4)
    if "error" not in m4:
        print(f"  Total trades:  {m4['total_trades']}")
        print(f"  Win rate:      {m4['win_rate']*100:.1f}%")
        print(f"  Total P&L:     {m4['total_pnl']:.2f}")

    print(f"\nLessons:")
    print(f"  - NR7 frequency ~5% of days = 13-15 setups/năm typical")
    print(f"  - NR4 frequency ~10% = 25-30 setups/năm, but weaker per signal")
    print(f"  - IDnr7 rare (~1-2%) but high conviction")
