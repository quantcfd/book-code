"""
QuantCFD — Chương 10.8
Drawdown Control

DD-based size scaling + equity curve filter.
Reduce position size as DD deepens. Stop trading at hard limits.
"""

from __future__ import annotations
import pandas as pd
import numpy as np


def dd_size_multiplier(
    current_dd: float, dd_thresholds: list = None,
) -> float:
    """
    Scale position size based on current drawdown.

    Args:
        current_dd: Current drawdown (negative number, e.g. -0.10 = -10%).
        dd_thresholds: List of (threshold, multiplier) tuples.

    Returns:
        Size multiplier (0.0 to 1.0).
    """
    if dd_thresholds is None:
        dd_thresholds = [
            (-0.05, 1.00),   # DD < 5%: full size
            (-0.10, 0.75),   # DD 5-10%: 75% size
            (-0.15, 0.50),   # DD 10-15%: 50% size
            (-0.20, 0.25),   # DD 15-20%: 25% size
            (-1.00, 0.00),   # DD > 20%: halt
        ]

    for threshold, multiplier in dd_thresholds:
        if current_dd > threshold:
            return multiplier
    return 0


def equity_curve_filter(
    strategy_equity: pd.Series, ma_period: int = 50,
) -> bool:
    """
    Trade only when equity above moving average.

    Args:
        strategy_equity: Equity time series.
        ma_period: MA window length.

    Returns:
        True if equity > MA (trade allowed).
    """
    if len(strategy_equity) < ma_period:
        return True  # Not enough data, allow trading
    ma = strategy_equity.rolling(ma_period).mean().iloc[-1]
    current = strategy_equity.iloc[-1]
    return current > ma


def streak_loss_multiplier(consecutive_losses: int) -> float:
    """
    Scale size based on consecutive loss streak.

    - 0-4 losses: 1.0x
    - 5-7 losses: 0.5x
    - 8-9 losses: 0.25x
    - 10+ losses: 0.0x (halt)
    """
    if consecutive_losses < 5:
        return 1.0
    elif consecutive_losses < 8:
        return 0.5
    elif consecutive_losses < 10:
        return 0.25
    else:
        return 0.0


def compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Compute drawdown series from equity curve."""
    peak = equity_curve.cummax()
    return (equity_curve - peak) / peak


def max_drawdown(equity_curve: pd.Series) -> dict:
    """
    Compute maximum drawdown statistics.

    Returns:
        Dict with max_dd, peak_date, trough_date, recovery_date, days_to_recovery.
    """
    if len(equity_curve) < 2:
        return {"max_dd": 0}

    dd = compute_drawdown_series(equity_curve)
    max_dd = dd.min()
    trough_date = dd.idxmin()
    peak_value = equity_curve[:trough_date].max()
    peak_date = equity_curve[:trough_date].idxmax()

    # Recovery: when equity reaches peak again
    after_trough = equity_curve[trough_date:]
    recovery_dates = after_trough[after_trough >= peak_value]
    if len(recovery_dates) > 0:
        recovery_date = recovery_dates.index[0]
        days_to_recovery = (recovery_date - trough_date).days
    else:
        recovery_date = None
        days_to_recovery = None

    return {
        "max_dd": max_dd,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "days_to_recovery": days_to_recovery,
    }


def underwater_curve(equity_curve: pd.Series) -> pd.Series:
    """
    Underwater curve = drawdown over time.
    Visualization tool for DD profile.
    """
    return compute_drawdown_series(equity_curve) * 100


if __name__ == "__main__":
    print("=" * 70)
    print("DD Control — Demo")
    print("=" * 70)

    # Demo 1: DD size multiplier
    print("\n--- DD-based size scaling ---")
    test_dds = [0.0, -0.03, -0.07, -0.12, -0.18, -0.25]
    for dd in test_dds:
        mult = dd_size_multiplier(dd)
        print(f"  DD {dd*100:>6.1f}% → size multiplier {mult:.2f}x")

    # Demo 2: Equity curve filter
    print("\n--- Equity curve filter ---")
    np.random.seed(42)
    n = 200
    eq = pd.Series(
        10000 * np.cumprod(1 + np.random.randn(n) * 0.01),
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )
    above_ma = equity_curve_filter(eq, ma_period=50)
    print(f"  Current equity: ${eq.iloc[-1]:,.2f}")
    print(f"  50-day MA: ${eq.rolling(50).mean().iloc[-1]:,.2f}")
    print(f"  Trade allowed: {above_ma}")

    # Demo 3: Streak loss scaling
    print("\n--- Streak loss multiplier ---")
    for streak in [0, 3, 5, 7, 9, 12]:
        mult = streak_loss_multiplier(streak)
        print(f"  {streak} consecutive losses → {mult:.2f}x")

    # Demo 4: Max DD analysis on synthetic equity curve
    print("\n--- Max DD analysis ---")
    np.random.seed(42)
    n = 1000
    rets = np.random.randn(n) * 0.012
    rets[300:400] -= 0.005  # injected DD period
    rets[700:750] -= 0.008  # second DD period
    eq = pd.Series(
        10000 * np.cumprod(1 + rets),
        index=pd.date_range("2022-01-01", periods=n, freq="D"),
    )
    dd_stats = max_drawdown(eq)
    print(f"  Max DD:          {dd_stats['max_dd']*100:.2f}%")
    print(f"  Peak date:       {dd_stats['peak_date'].date()}")
    print(f"  Trough date:     {dd_stats['trough_date'].date()}")
    if dd_stats["recovery_date"]:
        print(f"  Recovery date:   {dd_stats['recovery_date'].date()}")
        print(f"  Days to recover: {dd_stats['days_to_recovery']}")
    else:
        print(f"  Recovery:        not yet recovered")

    # Demo 5: Underwater curve sample
    print("\n--- Underwater curve (last 10 bars) ---")
    uw = underwater_curve(eq)
    for date, val in uw.tail(10).items():
        print(f"  {date.date()}: {val:>6.2f}%")
