"""
QuantCFD - Chapter 5 - Bài 1 Solution
=======================================
Lời giải Bài tập 1: Implement engine + cross-check với vectorbt.

3 việc:
    1. Backtest XAUUSD MA(20,50) trên data 2020-2024 với CFD costs đầy đủ
    2. Cross-check Sharpe với vectorbt khi cost = 0 (sai số phải <5%)
    3. In bảng so sánh idealized vs realistic

Yêu cầu:
    pip install yfinance pandas numpy
    pip install vectorbt  # optional, cho cross-check

Chạy:
    python chapter-05/solution_exercise_1.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from .backtest_engine import run_backtest
    from .cost_models import XAUUSD_SPREAD_BY_HOUR
    from .demo_xauusd_macross import ma_crossover_signal, print_stats
except ImportError:
    from backtest_engine import run_backtest
    from cost_models import XAUUSD_SPREAD_BY_HOUR
    from demo_xauusd_macross import ma_crossover_signal, print_stats


def cross_check_with_vectorbt(df: pd.DataFrame) -> dict:
    """
    Cross-check engine với vectorbt khi cost = 0.

    Sharpe 2 engines phải khớp ±5%. Nếu khác > 5% → có bug trong engine.
    """
    try:
        import vectorbt as vbt
    except ImportError:
        return {"vectorbt_available": False}

    # vectorbt cần 1 series Close + entries/exits
    fast = df["Close"].rolling(20).mean()
    slow = df["Close"].rolling(50).mean()
    entries = (fast > slow) & (fast.shift() <= slow.shift())
    exits = (fast < slow) & (fast.shift() >= slow.shift())

    pf = vbt.Portfolio.from_signals(
        df["Close"],
        entries=entries,
        exits=exits,
        freq="D",
        fees=0.0,
        slippage=0.0,
        init_cash=10_000.0,
    )

    return {
        "vectorbt_available":  True,
        "vbt_total_return":    float(pf.total_return()),
        "vbt_sharpe":          float(pf.sharpe_ratio()),
        "vbt_max_drawdown":    float(pf.max_drawdown()),
        "vbt_n_trades":        int(pf.trades.count()),
    }


def main() -> None:
    print("=" * 65)
    print(" BÀI 1: VALIDATE ENGINE — XAUUSD MA(20,50) BACKTEST")
    print("=" * 65)

    df = yf.download(
        "GC=F",
        start="2020-01-01",
        end="2024-12-31",
        progress=False,
        auto_adjust=True,
    )
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    signal_fn = ma_crossover_signal(20, 50)

    # ---- Idealized run ----
    r_ideal = run_backtest(
        df,
        signal_fn,
        spread_profile=None,
        swap_long_pct=0,
        swap_short_pct=0,
        slippage_base_pips=0,
        slippage_atr_mult=0,
        pip_size=0.01,
    )

    # ---- Realistic run ----
    r_real = run_backtest(
        df,
        signal_fn,
        spread_profile=XAUUSD_SPREAD_BY_HOUR,
        swap_long_pct=-3.0,
        swap_short_pct=+1.5,
        slippage_base_pips=2.0,
        slippage_atr_mult=0.05,
        pip_size=0.01,
    )

    print_stats("IDEALIZED (no costs)", r_ideal.stats())
    print_stats("REALISTIC (full CFD costs)", r_real.stats())

    # ---- Cross-check với vectorbt ----
    print("\n" + "=" * 65)
    print(" CROSS-CHECK với vectorbt (cost = 0, expect ±5% match)")
    print("=" * 65)
    vbt_check = cross_check_with_vectorbt(df)
    if not vbt_check.get("vectorbt_available"):
        print("vectorbt không cài. Skip cross-check.")
        print("Cài: pip install vectorbt")
    else:
        my_sharpe = r_ideal.stats().get("sharpe", 0)
        vbt_sharpe = vbt_check["vbt_sharpe"]
        diff_pct = abs(my_sharpe - vbt_sharpe) / max(abs(vbt_sharpe), 0.01) * 100

        print(f"  My engine Sharpe:    {my_sharpe:+.3f}")
        print(f"  vectorbt Sharpe:     {vbt_sharpe:+.3f}")
        print(f"  Difference:          {diff_pct:.1f}%")

        if diff_pct < 5:
            print(f"  ✓ PASSED (< 5% — engine của anh em consistent với vectorbt)")
        else:
            print(f"  ✗ FAILED ({diff_pct:.1f}% > 5%)")
            print(
                f"    Một trong 2 engine có bug. Re-check:\n"
                f"    - shift(1) cho signal đúng chỗ chưa\n"
                f"    - Tính returns đúng (Close.pct_change())\n"
                f"    - vectorbt có dùng entries/exits đúng cách"
            )


if __name__ == "__main__":
    main()
