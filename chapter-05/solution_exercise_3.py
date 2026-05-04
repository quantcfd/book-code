"""
QuantCFD - Chapter 5 - Bài 3 Solution
=======================================
Lời giải Bài tập 3: Walk-forward analysis trên XAUUSD MA crossover.

Param grid:
    fast in [10, 20, 30, 50]
    slow in [30, 50, 100, 150]

Train 2 years anchored, test 1 year. Output bảng:
    window | best_params | in_sample_sharpe | oos_sharpe | oos_return | oos_dd

Verdict: strategy có robust không?

Chạy:
    python chapter-05/solution_exercise_3.py
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

try:
    from .cost_models import XAUUSD_SPREAD_BY_HOUR
    from .demo_xauusd_macross import ma_crossover_signal
    from .walk_forward import walk_forward_analysis, summarize_wfa
except ImportError:
    from cost_models import XAUUSD_SPREAD_BY_HOUR
    from demo_xauusd_macross import ma_crossover_signal
    from walk_forward import walk_forward_analysis, summarize_wfa


def main() -> None:
    print("=" * 70)
    print(" BÀI 3: WALK-FORWARD ANALYSIS — XAUUSD MA CROSSOVER")
    print("=" * 70)

    # Load 5 years data — đủ cho ~3 windows of (train=2y, test=1y) anchored
    df = yf.download(
        "GC=F",
        start="2019-01-01",
        end="2024-12-31",
        progress=False,
        auto_adjust=True,
    )
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    print(f"Loaded {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}\n")

    # Param grid: 16 combinations
    param_grid = []
    for fast in [10, 20, 30, 50]:
        for slow in [30, 50, 100, 150]:
            if fast < slow:  # fast phải nhỏ hơn slow
                param_grid.append({"fast": fast, "slow": slow})

    print(f"Param grid: {len(param_grid)} combinations")
    print(f"  fast in [10, 20, 30, 50]")
    print(f"  slow in [30, 50, 100, 150]")
    print(f"  constraint: fast < slow\n")

    # Run walk-forward
    print("Running anchored walk-forward (train=2y, test=1y)...")
    print("(takes 1-2 minutes for 16 params × 3 windows...)\n")

    results = walk_forward_analysis(
        df,
        signal_fn_factory=ma_crossover_signal,
        param_grid=param_grid,
        train_years=2,
        test_years=1,
        anchored=True,
        # Engine kwargs - apply CFD costs
        spread_profile=XAUUSD_SPREAD_BY_HOUR,
        swap_long_pct=-3.0,
        swap_short_pct=+1.5,
        slippage_base_pips=2.0,
        slippage_atr_mult=0.05,
        pip_size=0.01,
    )

    # ---- Print bảng ----
    print("\n" + "=" * 90)
    print(" WALK-FORWARD RESULTS")
    print("=" * 90)
    print(
        f"{'Window':12s} {'Best params':22s} "
        f"{'IS Sharpe':>10s} {'OOS Sharpe':>11s} "
        f"{'OOS Ret':>10s} {'OOS DD':>10s}"
    )
    print("-" * 90)

    for _, row in results.iterrows():
        win_str = (
            f"{row['window_start'].strftime('%Y-%m')} → "
            f"{row['window_end'].strftime('%Y-%m')}"
        )
        params_str = str(row["best_params"])
        print(
            f"{win_str:12s} {params_str:22s} "
            f"{row['in_sample_sharpe']:>+10.2f} "
            f"{row['out_of_sample_sharpe']:>+11.2f} "
            f"{row['oos_total_return']:>+10.2%} "
            f"{row['oos_max_dd']:>+10.2%}"
        )

    # ---- Verdict ----
    print("\n" + "=" * 90)
    summary = summarize_wfa(results)
    print(" SUMMARY")
    print("=" * 90)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:25s} = {v:+.3f}")
        else:
            print(f"  {k:25s} = {v}")

    # ---- Discussion ----
    print(
        "\nDiscussion (anh em viết phần này trong bài nộp):\n"
        "\n"
        "  1. Mean OOS Sharpe = ?\n"
        "     - ≥ 0.5 → strategy có edge thực\n"
        "     - 0.3 đến 0.5 → marginal, cần thêm filter\n"
        "     - < 0.3 → có thể là noise hoặc overfit\n"
        "\n"
        "  2. Số window có OOS âm = ?\n"
        "     - 0 → consistent, robust\n"
        "     - 1-2 → có period bad nhưng overall OK\n"
        "     - >2 → strategy không robust theo regime\n"
        "\n"
        "  3. In-sample to out-of-sample decay = ?\n"
        "     - < 30% → params chọn tốt, generalize được\n"
        "     - > 50% → overfitting nặng → đừng live trade\n"
        "\n"
        "  4. Quyết định live trade?\n"
        "     - Verdict ROBUST + decay < 30% + max DD chấp nhận được → YES\n"
        "     - Còn lại → tinh chỉnh hoặc bỏ"
    )


if __name__ == "__main__":
    main()
