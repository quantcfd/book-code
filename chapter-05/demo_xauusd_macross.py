"""
QuantCFD - Chapter 5 - Demo: XAUUSD MA Crossover
==================================================
Section 5.5: Backtest với engine, so sánh idealized vs realistic CFD costs.

Cùng chiến lược, cùng data, kết quả khác xa khi tính đầy đủ costs.

Yêu cầu:
    pip install yfinance pandas numpy

Chạy:
    python chapter-05/demo_xauusd_macross.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from .backtest_engine import run_backtest
    from .cost_models import XAUUSD_SPREAD_BY_HOUR
except ImportError:
    from backtest_engine import run_backtest
    from cost_models import XAUUSD_SPREAD_BY_HOUR


def ma_crossover_signal(fast: int = 20, slow: int = 50):
    """Factory: trả về signal_fn cho MA(fast,slow) crossover."""

    def signal_fn(df: pd.DataFrame) -> pd.Series:
        ma_fast = df["Close"].rolling(fast).mean()
        ma_slow = df["Close"].rolling(slow).mean()
        return pd.Series(
            np.where(ma_fast > ma_slow, 1, -1),
            index=df.index,
        )

    return signal_fn


def print_stats(label: str, stats: dict) -> None:
    print(f"\n{label}:")
    if not stats:
        print("  (no data)")
        return
    for k, v in stats.items():
        if isinstance(v, float):
            if k in ("win_rate", "cost_drag"):
                print(f"  {k:18s} = {v:>10.2%}")
            elif "return" in k or "drawdown" in k:
                print(f"  {k:18s} = {v:>10.2%}")
            else:
                print(f"  {k:18s} = {v:>10.4f}")
        else:
            print(f"  {k:18s} = {v:>10}")


def main() -> None:
    print("Loading XAUUSD (GC=F) data 2020-2024...")
    df = yf.download(
        "GC=F",
        start="2020-01-01",
        end="2024-12-31",
        progress=False,
        auto_adjust=True,
    )

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    print(f"Loaded {len(df)} daily bars")

    signal_fn = ma_crossover_signal(fast=20, slow=50)

    # ---- Run 1: NO costs (idealized — như 99% YouTube tutorial) ----
    print("\n" + "=" * 60)
    print(" RUN 1: IDEALIZED (no costs)")
    print("=" * 60)
    r1 = run_backtest(
        df,
        signal_fn,
        spread_profile=None,
        swap_long_pct=0,
        swap_short_pct=0,
        slippage_base_pips=0,
        slippage_atr_mult=0,
        pip_size=0.01,
    )
    print_stats("Idealized stats", r1.stats(periods_per_year=252))

    # ---- Run 2: WITH realistic CFD costs ----
    print("\n" + "=" * 60)
    print(" RUN 2: REALISTIC (full CFD costs)")
    print("=" * 60)
    r2 = run_backtest(
        df,
        signal_fn,
        spread_profile=XAUUSD_SPREAD_BY_HOUR,
        swap_long_pct=-3.0,
        swap_short_pct=+1.5,
        slippage_base_pips=2.0,  # 2 cents base for XAUUSD
        slippage_atr_mult=0.05,
        pip_size=0.01,
    )
    print_stats("Realistic stats", r2.stats(periods_per_year=252))

    # ---- So sánh ----
    s1 = r1.stats(periods_per_year=252)
    s2 = r2.stats(periods_per_year=252)
    print("\n" + "=" * 60)
    print(" COST IMPACT — đây là điểm quan trọng nhất của chương")
    print("=" * 60)
    if s1 and s2:
        ret_drop = (s1["total_return"] - s2["total_return"]) / abs(
            s1["total_return"]
        )
        sharpe_drop = (s1["sharpe"] - s2["sharpe"]) / abs(s1["sharpe"])

        print(
            f"  Total return idealized:  {s1['total_return']:+.2%}\n"
            f"  Total return realistic:  {s2['total_return']:+.2%}\n"
            f"  Return drop:             {ret_drop:+.1%}\n"
            f"\n"
            f"  Sharpe idealized:        {s1['sharpe']:+.2f}\n"
            f"  Sharpe realistic:        {s2['sharpe']:+.2f}\n"
            f"  Sharpe drop:             {sharpe_drop:+.1%}\n"
            f"\n"
            f"  Cost drag on initial $10k: {s2['cost_drag']:.1%}"
        )

    print(
        "\nKết luận: Cùng strategy, cùng data, nhưng Sharpe thực giảm 40-60% "
        "khi tính đầy đủ CFD costs. Đây là vì sao 90% backtest 'đẹp' không "
        "reproduce được trong live trade."
    )


if __name__ == "__main__":
    main()
