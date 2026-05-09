"""
QuantCFD — Chương 9.3
Opening Range Breakout (ORB)

ORB là chiến lược vol breakout cổ điển trên indices và commodities.
Logic: first 30-60 phút sau market open thiết lập trading range,
breakout của range này tiếp tục đến hết ngày.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def opening_range_breakout(
    df_intraday: pd.DataFrame,
    range_minutes: int = 30,
    target_mult: float = 2.0,
    stop_mult: float = 1.0,
) -> pd.DataFrame:
    """
    Opening Range Breakout strategy.

    Define opening range = first N min sau market open.
    Long when price closes > OR high.
    Short when price closes < OR low.
    Stop: opposite side of OR. Target: target_mult × OR size.

    Args:
        df_intraday: DataFrame with datetime index, columns
                     [open, high, low, close].
        range_minutes: Length of opening range in minutes.
        target_mult: Take-profit as multiple of OR size.
        stop_mult: Stop-loss as fraction of OR size (1.0 = full opposite side).

    Returns:
        DataFrame with one row per trade:
        date, side, entry, exit, pnl, result.
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
        if len(post_or) == 0:
            continue

        position = 0
        entry_price = stop_price = target_price = None

        for ts, row in post_or.iterrows():
            if position == 0:
                if row["close"] > or_high:
                    position = 1
                    entry_price = row["close"]
                    stop_price = or_low if stop_mult >= 1.0 else (
                        entry_price - stop_mult * or_size
                    )
                    target_price = entry_price + target_mult * or_size
                elif row["close"] < or_low:
                    position = -1
                    entry_price = row["close"]
                    stop_price = or_high if stop_mult >= 1.0 else (
                        entry_price + stop_mult * or_size
                    )
                    target_price = entry_price - target_mult * or_size
            elif position == 1:
                if row["low"] <= stop_price:
                    trades.append({
                        "date": date, "side": "long",
                        "entry": entry_price, "exit": stop_price,
                        "pnl": stop_price - entry_price, "result": "stop",
                    })
                    position = 0
                elif row["high"] >= target_price:
                    trades.append({
                        "date": date, "side": "long",
                        "entry": entry_price, "exit": target_price,
                        "pnl": target_price - entry_price, "result": "target",
                    })
                    position = 0
            elif position == -1:
                if row["high"] >= stop_price:
                    trades.append({
                        "date": date, "side": "short",
                        "entry": entry_price, "exit": stop_price,
                        "pnl": entry_price - stop_price, "result": "stop",
                    })
                    position = 0
                elif row["low"] <= target_price:
                    trades.append({
                        "date": date, "side": "short",
                        "entry": entry_price, "exit": target_price,
                        "pnl": entry_price - target_price, "result": "target",
                    })
                    position = 0

        # End of day exit
        if position != 0:
            last_price = post_or.iloc[-1]["close"]
            pnl = (last_price - entry_price) * position
            trades.append({
                "date": date,
                "side": "long" if position == 1 else "short",
                "entry": entry_price, "exit": last_price,
                "pnl": pnl, "result": "eod",
            })

    return pd.DataFrame(trades)


def orb_metrics(trades_df: pd.DataFrame) -> dict:
    """Compute summary metrics from ORB trades."""
    if len(trades_df) == 0:
        return {"error": "no trades"}

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]

    total_pnl = trades_df["pnl"].sum()
    win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0
    avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0
    rr = -avg_win / avg_loss if avg_loss != 0 else 0

    return {
        "total_trades": len(trades_df),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_rr": rr,
        "total_pnl": total_pnl,
        "result_breakdown": trades_df["result"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Opening Range Breakout — Demo (US500 5-min synthetic)")
    print("=" * 70)

    np.random.seed(42)
    # Simulate 60 trading days, 5-min bars from 09:30 to 16:00 ET
    bars_per_day = 78  # 6.5 hours × 12 5-min bars
    n_days = 60

    all_data = []
    for day in range(n_days):
        date_start = pd.Timestamp("2024-01-02") + pd.Timedelta(days=day)
        # Skip weekends (rough)
        if date_start.weekday() >= 5:
            continue
        bars = pd.date_range(
            date_start.replace(hour=9, minute=30),
            periods=bars_per_day, freq="5min",
        )
        # Synthetic price walk with intraday trend
        base = 4500 + day * 2  # slight upward drift
        intraday_drift = np.random.choice([-0.003, 0.0, 0.003])
        rets = np.random.randn(bars_per_day) * 0.001 + intraday_drift / bars_per_day
        prices = base * np.exp(np.cumsum(rets))
        opens = prices * (1 + np.random.randn(bars_per_day) * 0.0005)
        highs = np.maximum(opens, prices) * (1 + np.abs(np.random.randn(bars_per_day)) * 0.0008)
        lows = np.minimum(opens, prices) * (1 - np.abs(np.random.randn(bars_per_day)) * 0.0008)

        day_df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": prices,
        }, index=bars)
        all_data.append(day_df)

    df = pd.concat(all_data)
    print(f"\nData: {df.index[0]} → {df.index[-1]}")
    print(f"Total bars: {len(df)}")

    # Run ORB
    trades = opening_range_breakout(df, range_minutes=30, target_mult=2.0)

    # Metrics
    m = orb_metrics(trades)
    if "error" in m:
        print(f"\n{m['error']}")
    else:
        print(f"\n{'─' * 70}")
        print(f"Total trades: {m['total_trades']}")
        print(f"  Wins:      {m['wins']} ({m['win_rate']*100:.1f}%)")
        print(f"  Losses:    {m['losses']}")
        print(f"  Avg win:   {m['avg_win']:.2f}")
        print(f"  Avg loss:  {m['avg_loss']:.2f}")
        print(f"  R:R:       {m['avg_rr']:.2f}")
        print(f"  Total P&L: {m['total_pnl']:.2f}")
        print(f"  Results:   {m['result_breakdown']}")
