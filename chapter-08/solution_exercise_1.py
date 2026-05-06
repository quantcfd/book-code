"""
QuantCFD — Chương 8, Bài tập 1
Bollinger Bands MR từ scratch (90 phút)

Yêu cầu:
- Implement BB(20, 2) MR strategy
- Test trên synthetic XAUUSD H4
- Compute Sharpe, CAGR, max DD
- Verify anti-look-ahead
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def bb_mr_solution(
    df: pd.DataFrame,
    period: int = 20,
    std_mult: float = 2.0,
    cost: float = 0.0008,
) -> dict:
    """Full BB MR implementation từ scratch — không dùng helper modules."""
    out = df.copy()

    # 1. Compute BB
    sma = out["close"].rolling(period).mean()
    std = out["close"].rolling(period).std()

    # 2. Bands SHIFTED to avoid look-ahead
    out["upper"] = (sma + std_mult * std).shift(1)
    out["lower"] = (sma - std_mult * std).shift(1)
    out["mid"] = sma.shift(1)

    # 3. Generate signals (state machine)
    position = 0
    positions = []
    for i in range(len(out)):
        row = out.iloc[i]
        if pd.isna(row["lower"]):
            positions.append(0)
            continue
        price = row["close"]

        if position == 0:
            if price < row["lower"]:
                position = 1   # oversold → long
            elif price > row["upper"]:
                position = -1  # overbought → short
        elif position == 1:
            if price >= row["mid"]:
                position = 0   # mean → exit
        elif position == -1:
            if price <= row["mid"]:
                position = 0

        positions.append(position)

    out["position"] = positions

    # 4. Compute returns
    out["asset_return"] = out["close"].pct_change()
    out["strat_return"] = out["position"] * out["asset_return"]

    # 5. Apply cost
    out["pos_change"] = pd.Series(positions, index=out.index).diff().abs().fillna(0)
    out["strat_return_net"] = (
        out["strat_return"] - out["pos_change"] * cost
    )

    out_clean = out.dropna()
    if len(out_clean) < 30:
        return {"error": "insufficient data"}

    # 6. Metrics
    bars_per_year = 252 * 6  # H4 = 6 bars/day
    sharpe = (
        out_clean["strat_return_net"].mean() / out_clean["strat_return_net"].std()
        * np.sqrt(bars_per_year)
        if out_clean["strat_return_net"].std() > 0 else 0
    )
    cagr = (1 + out_clean["strat_return_net"].mean()) ** bars_per_year - 1

    equity = (1 + out_clean["strat_return_net"]).cumprod()
    max_dd = (equity / equity.cummax() - 1).min()

    total_trades = int(out_clean["pos_change"].sum() / 2)

    # Trade-level stats
    out_clean = out_clean.copy()
    out_clean["trade_id"] = (out_clean["pos_change"].cumsum() / 2).astype(int)
    trade_pnl = out_clean.groupby("trade_id")["strat_return_net"].sum()
    win_rate = (trade_pnl > 0).mean() if len(trade_pnl) > 0 else 0

    return {
        "sharpe": sharpe,
        "cagr": cagr,
        "max_dd": max_dd,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "equity_curve": equity,
        "positions": out_clean["position"],
    }


def verify_no_lookahead(df: pd.DataFrame, period: int = 20, std_mult: float = 2.0):
    """Verify that signal at bar T doesn't peek into future."""
    full = bb_mr_solution(df, period, std_mult)
    full_pos = full["positions"]

    mid_idx = len(df) * 2 // 3
    truncated = df.iloc[: mid_idx + 1]
    trunc = bb_mr_solution(truncated, period, std_mult)

    if "error" in trunc:
        return False
    trunc_pos = trunc["positions"]

    last_common = trunc_pos.index[-1]
    if trunc_pos.loc[last_common] == full_pos.loc[last_common]:
        print(f"✓ No look-ahead: position at bar {last_common} consistent")
        return True
    else:
        print(f"✗ LOOK-AHEAD DETECTED: full={full_pos.loc[last_common]}, "
              f"truncated={trunc_pos.loc[last_common]}")
        return False


if __name__ == "__main__":
    np.random.seed(42)
    n = 5000
    dates = pd.date_range("2018-01-01", periods=n, freq="4h")

    # Synthetic mean-reverting prices
    drift = 0.0001
    returns = np.random.randn(n) * 0.005 + drift
    # Add MR component (autocorrelation -0.1)
    for i in range(1, n):
        returns[i] -= 0.1 * returns[i-1]
    prices = 1500 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({"close": prices}, index=dates)

    print("=" * 60)
    print("Bài tập 1 — Bollinger Bands MR (XAUUSD synthetic H4)")
    print("=" * 60)

    result = bb_mr_solution(df, period=20, std_mult=2.0)

    print(f"\nSharpe:        {result['sharpe']:.3f}")
    print(f"CAGR:          {result['cagr']*100:.2f}%")
    print(f"Max DD:        {result['max_dd']*100:.2f}%")
    print(f"Total trades:  {result['total_trades']}")
    print(f"Win rate:      {result['win_rate']*100:.1f}%")

    print("\n" + "-" * 60)
    print("Verify no look-ahead bias:")
    verify_no_lookahead(df)

    print("\nLessons:")
    print("  - .shift(1) trên BB bands là CRITICAL")
    print("  - Verify bằng truncation test")
    print("  - MR strategy có win rate cao (60-70%) nhưng tail risk")
    print("  - BB(20,2) là baseline — tune params với WFA (xem solution_exercise_4)")
