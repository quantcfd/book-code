"""
QuantCFD — Chương 9, Bài tập 2
NR7 / NR4 / IDnr7 detection + trading (90 phút)

Yêu cầu:
- Implement detect_nr7, detect_idnr7
- Build full strategy with entry/stop/target
- Compare NR7 vs NR4 frequency
- Test trên synthetic XAU daily 2018-2024
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def detect_nr_n(df: pd.DataFrame, n: int) -> pd.Series:
    """Detect NR-N day (range narrowest in N bars)."""
    rng = df["high"] - df["low"]
    rolling_min = rng.rolling(n).min()
    rolling_count = rng.rolling(n).count()
    return (rng == rolling_min) & (rolling_count == n)


def detect_inside_day(df: pd.DataFrame) -> pd.Series:
    """Today high < yesterday high AND today low > yesterday low."""
    return (
        (df["high"] < df["high"].shift(1))
        & (df["low"] > df["low"].shift(1))
    )


def detect_idnr(df: pd.DataFrame, n: int = 7) -> pd.Series:
    """Inside Day + NR-N (strongest signal)."""
    return detect_nr_n(df, n) & detect_inside_day(df)


def nr_strategy_full(
    df: pd.DataFrame,
    n: int = 7,
    target_mult: float = 1.5,
    use_idnr: bool = False,
) -> dict:
    """
    Full NR-N breakout strategy with metrics.
    """
    df = df.copy()
    df["range"] = df["high"] - df["low"]

    if use_idnr:
        setup = detect_idnr(df, n)
    else:
        setup = detect_nr_n(df, n)

    df["setup_high"] = df["high"].shift(1).where(setup.shift(1))
    df["setup_low"] = df["low"].shift(1).where(setup.shift(1))
    df["setup_range"] = df["range"].shift(1).where(setup.shift(1))

    trades = []
    for idx, row in df.iterrows():
        if pd.isna(row["setup_high"]):
            continue

        sh, sl, sr = row["setup_high"], row["setup_low"], row["setup_range"]

        # Long
        if row["high"] > sh:
            entry = sh
            stop = sl
            target = entry + target_mult * sr
            if row["low"] <= stop:
                exit_p, result = stop, "stop"
            elif row["high"] >= target:
                exit_p, result = target, "target"
            else:
                exit_p, result = row["close"], "eod"
            trades.append({
                "date": idx, "side": "long", "entry": entry,
                "exit": exit_p, "pnl": exit_p - entry, "result": result,
            })

        # Short
        elif row["low"] < sl:
            entry = sl
            stop = sh
            target = entry - target_mult * sr
            if row["high"] >= stop:
                exit_p, result = stop, "stop"
            elif row["low"] <= target:
                exit_p, result = target, "target"
            else:
                exit_p, result = row["close"], "eod"
            trades.append({
                "date": idx, "side": "short", "entry": entry,
                "exit": exit_p, "pnl": entry - exit_p, "result": result,
            })

    if len(trades) == 0:
        return {"error": "no trades"}

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]

    n_setup = setup.sum()

    return {
        "n_setups": int(n_setup),
        "n_trades": len(trades_df),
        "win_rate": len(wins) / len(trades_df),
        "avg_win": wins["pnl"].mean() if len(wins) > 0 else 0,
        "avg_loss": losses["pnl"].mean() if len(losses) > 0 else 0,
        "total_pnl": trades_df["pnl"].sum(),
        "results": trades_df["result"].value_counts().to_dict(),
        "trades": trades_df,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Bài tập 2 — NR7 / NR4 / IDnr7")
    print("=" * 60)

    np.random.seed(42)
    n = 7 * 252
    dates = pd.date_range("2018-01-01", periods=n, freq="D")

    rets = np.random.randn(n) * 0.012
    closes = 1500 * np.exp(np.cumsum(rets))
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

    # Compare NR7 vs NR4 vs IDnr7
    print(f"\n{'─' * 60}")
    print(f"{'Strategy':<20} {'Setups':>8} {'Trades':>8} {'WR':>8} {'Total':>10}")
    print(f"{'─' * 60}")

    configs = [
        ("NR7", 7, False),
        ("NR4", 4, False),
        ("IDnr7", 7, True),
    ]

    for name, n_lookback, use_id in configs:
        r = nr_strategy_full(df, n=n_lookback, use_idnr=use_id)
        if "error" not in r:
            print(f"  {name:<18} {r['n_setups']:>8} {r['n_trades']:>8} "
                  f"{r['win_rate']*100:>7.1f}% {r['total_pnl']:>10.2f}")

    # Detailed view of NR7
    print(f"\n{'─' * 60}")
    print("Detailed NR7 results:")
    print(f"{'─' * 60}")
    r = nr_strategy_full(df, n=7)
    print(f"  Setups:        {r['n_setups']}")
    print(f"  Trades:        {r['n_trades']}")
    print(f"  Win rate:      {r['win_rate']*100:.1f}%")
    print(f"  Avg win:       {r['avg_win']:.2f}")
    print(f"  Avg loss:      {r['avg_loss']:.2f}")
    if r['avg_loss'] != 0:
        rr = -r['avg_win'] / r['avg_loss']
        print(f"  R:R ratio:     {rr:.2f}")
    print(f"  Total P&L:     {r['total_pnl']:.2f}")
    print(f"  Results:       {r['results']}")

    print(f"\nLessons:")
    print(f"  - NR7 detection requires checking ALL 7 prior bars (rolling.min)")
    print(f"  - Setup day = previous day, Trade day = today")
    print(f"  - Inside Day filter (IDnr7) reduces frequency 50% but better quality")
    print(f"  - target_mult tunable: 1.5 default, 2.0 aggressive, 1.0 conservative")
