"""
QuantCFD — Chương 9, Bài tập 1
Opening Range Breakout từ scratch (60 phút)

Yêu cầu:
- Implement ORB function từ đầu
- Test trên synthetic intraday data
- Compute Sharpe, win rate, R:R
- Verify zero look-ahead bias
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def orb_from_scratch(
    df_intraday: pd.DataFrame,
    range_minutes: int = 30,
    target_mult: float = 2.0,
) -> dict:
    """
    Full ORB implementation from scratch.

    Returns trades list and metrics.
    """
    df = df_intraday.copy()
    df["date"] = df.index.date
    trades = []

    for date, day_data in df.groupby("date"):
        if len(day_data) < 5:
            continue

        market_open = day_data.index[0]
        range_end = market_open + pd.Timedelta(minutes=range_minutes)

        opening_range = day_data[day_data.index <= range_end]
        if len(opening_range) < 2:
            continue

        or_high = opening_range["high"].max()
        or_low = opening_range["low"].min()
        or_size = or_high - or_low

        if or_size <= 0:
            continue

        post_or = day_data[day_data.index > range_end]

        position = 0
        entry = stop = target = None
        for ts, row in post_or.iterrows():
            if position == 0:
                if row["close"] > or_high:
                    position = 1
                    entry = row["close"]
                    stop = or_low
                    target = entry + target_mult * or_size
                elif row["close"] < or_low:
                    position = -1
                    entry = row["close"]
                    stop = or_high
                    target = entry - target_mult * or_size
            elif position == 1:
                if row["low"] <= stop:
                    trades.append({
                        "date": date, "side": "long", "pnl": stop - entry,
                        "result": "stop",
                    })
                    position = 0
                elif row["high"] >= target:
                    trades.append({
                        "date": date, "side": "long", "pnl": target - entry,
                        "result": "target",
                    })
                    position = 0
            elif position == -1:
                if row["high"] >= stop:
                    trades.append({
                        "date": date, "side": "short", "pnl": entry - stop,
                        "result": "stop",
                    })
                    position = 0
                elif row["low"] <= target:
                    trades.append({
                        "date": date, "side": "short", "pnl": entry - target,
                        "result": "target",
                    })
                    position = 0

        # End-of-day exit
        if position != 0:
            last = post_or.iloc[-1]["close"]
            pnl = (last - entry) * position
            trades.append({
                "date": date, "side": "long" if position == 1 else "short",
                "pnl": pnl, "result": "eod",
            })

    if len(trades) == 0:
        return {"error": "no trades"}

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]

    return {
        "n_trades": len(trades_df),
        "win_rate": len(wins) / len(trades_df),
        "avg_win": wins["pnl"].mean() if len(wins) > 0 else 0,
        "avg_loss": losses["pnl"].mean() if len(losses) > 0 else 0,
        "total_pnl": trades_df["pnl"].sum(),
        "trades": trades_df,
    }


def verify_no_lookahead(df: pd.DataFrame) -> bool:
    """
    Verify ORB doesn't use future data:
    Run on full dataset, then on truncated. Compare overlapping period.
    """
    full = orb_from_scratch(df)
    if "error" in full:
        return False

    mid_idx = len(df) * 2 // 3
    truncated = df.iloc[: mid_idx + 1]
    trunc = orb_from_scratch(truncated)

    if "error" in trunc:
        return False

    # Compare trades up to truncation point
    full_trades = full["trades"]
    trunc_trades = trunc["trades"]
    truncation_date = df.index[mid_idx].date()

    full_before = full_trades[full_trades["date"] <= truncation_date]
    trunc_match = trunc_trades[trunc_trades["date"] <= truncation_date]

    if len(full_before) == len(trunc_match):
        # Same number of trades up to truncation → no look-ahead
        return True
    else:
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Bài tập 1 — ORB từ scratch")
    print("=" * 60)

    np.random.seed(42)

    # Generate synthetic 5-min intraday data
    bars_per_day = 78  # 9:30-16:00 ET = 78 bars
    n_days = 100

    all_data = []
    for day in range(n_days):
        date_start = pd.Timestamp("2024-01-02") + pd.Timedelta(days=day)
        if date_start.weekday() >= 5:  # skip weekends
            continue
        bars = pd.date_range(
            date_start.replace(hour=9, minute=30),
            periods=bars_per_day, freq="5min",
        )
        base = 4500 + day * 1.5
        intraday_drift = np.random.choice([-0.002, 0, 0.002])
        rets = np.random.randn(bars_per_day) * 0.0008 + intraday_drift / bars_per_day
        prices = base * np.exp(np.cumsum(rets))
        opens = prices * (1 + np.random.randn(bars_per_day) * 0.0003)
        highs = np.maximum(opens, prices) * (
            1 + np.abs(np.random.randn(bars_per_day)) * 0.0006
        )
        lows = np.minimum(opens, prices) * (
            1 - np.abs(np.random.randn(bars_per_day)) * 0.0006
        )
        all_data.append(pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": prices,
        }, index=bars))

    df = pd.concat(all_data)
    print(f"\nData: {df.index[0]} → {df.index[-1]} ({len(df)} bars)")

    result = orb_from_scratch(df, range_minutes=30, target_mult=2.0)
    if "error" in result:
        print(result["error"])
    else:
        print(f"\nResults:")
        print(f"  Total trades: {result['n_trades']}")
        print(f"  Win rate:     {result['win_rate']*100:.1f}%")
        print(f"  Avg win:      ${result['avg_win']:.2f}")
        print(f"  Avg loss:     ${result['avg_loss']:.2f}")
        if result['avg_loss'] != 0:
            rr = -result['avg_win'] / result['avg_loss']
            print(f"  R:R ratio:    {rr:.2f}")
        print(f"  Total P&L:    ${result['total_pnl']:.2f}")

    print(f"\n--- Look-ahead verification ---")
    no_lookahead = verify_no_lookahead(df)
    print(f"  {'✓ PASS' if no_lookahead else '✗ FAIL'} — no look-ahead bias")

    print(f"\nLessons:")
    print(f"  - .iloc / shift(1) prevent look-ahead")
    print(f"  - End-of-day exit critical (don't carry overnight)")
    print(f"  - Stop loss at opposite OR boundary (full or partial)")
    print(f"  - Realistic backtest needs slippage model (xem live_execution_breakout)")
