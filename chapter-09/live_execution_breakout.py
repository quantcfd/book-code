"""
QuantCFD — Chương 9.11
Live Execution Helpers cho Vol Breakout

5 issues addressed:
1. News event filter (NFP, FOMC, CPI ±30 min)
2. Slippage adjustment for vol spikes
3. Gap-through stop simulation
4. Daily loss limit
5. Streak loss size scaling
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import time as dtime


def is_news_window(
    timestamp: pd.Timestamp,
    window_minutes: int = 30,
) -> bool:
    """
    Check if timestamp is within ±window_minutes of major news event.

    Simplified version covering:
    - NFP: First Friday of month, 13:30 UTC (8:30 ET)
    - CPI: ~10-15th of month, 13:30 UTC
    - FOMC: ~Wednesday afternoons, 18:00-19:00 UTC
    """
    ts = pd.Timestamp(timestamp)

    # NFP: first Friday, 13:30 UTC
    if ts.weekday() == 4 and ts.day <= 7:
        nfp_time = ts.replace(hour=13, minute=30, second=0, microsecond=0)
        if abs((ts - nfp_time).total_seconds()) <= window_minutes * 60:
            return True

    # CPI: roughly day 10-15, 13:30 UTC
    if 10 <= ts.day <= 15:
        cpi_time = ts.replace(hour=13, minute=30, second=0, microsecond=0)
        if abs((ts - cpi_time).total_seconds()) <= window_minutes * 60:
            return True

    # FOMC: Wednesday 18:00 UTC
    if ts.weekday() == 2:
        fomc_time = ts.replace(hour=18, minute=0, second=0, microsecond=0)
        if abs((ts - fomc_time).total_seconds()) <= window_minutes * 60:
            return True

    return False


def filter_news_signals(
    signals: pd.Series, window_minutes: int = 30,
) -> pd.Series:
    """Zero-out signals during news windows."""
    filtered = signals.copy()
    for ts in signals.index:
        if is_news_window(ts, window_minutes):
            filtered.loc[ts] = 0
    return filtered


def simulate_slippage(
    backtest_fill: float,
    direction: int,
    typical_spread: float,
    is_news_window_active: bool = False,
    is_session_active: bool = True,
) -> float:
    """
    Adjust backtest fill price for realistic slippage.

    Args:
        backtest_fill: Original fill price.
        direction: +1 (buy) or -1 (sell).
        typical_spread: Typical bid-ask spread.
        is_news_window_active: If True, apply 3-5x slippage.
        is_session_active: If False, apply 2x slippage (low liquidity).

    Returns:
        Adjusted fill price.
    """
    base_slippage = typical_spread * 1.5  # 1.5x spread default

    if is_news_window_active:
        base_slippage *= 4  # 6x spread total
    elif not is_session_active:
        base_slippage *= 2  # 3x spread total

    return backtest_fill + direction * base_slippage


def simulate_gap_through_stop(
    entry_price: float,
    stop_price: float,
    actual_open_price: float,
    direction: int = 1,
) -> dict:
    """
    Simulate gap-through stop execution.

    When market opens with gap beyond stop level, fill at open
    (not stop). Calculates excess loss vs intended.

    Args:
        entry_price: Position entry.
        stop_price: Intended stop level.
        actual_open_price: Actual market open after gap.
        direction: +1 (long) or -1 (short).

    Returns:
        Dict with fill_price, actual_loss, expected_loss, excess_loss_pct.
    """
    if direction == 1:
        # Long: gap down through stop
        if actual_open_price < stop_price:
            fill_price = actual_open_price
        else:
            fill_price = stop_price
        actual_loss = entry_price - fill_price
        expected_loss = entry_price - stop_price
    else:
        # Short: gap up through stop
        if actual_open_price > stop_price:
            fill_price = actual_open_price
        else:
            fill_price = stop_price
        actual_loss = fill_price - entry_price
        expected_loss = stop_price - entry_price

    excess_loss = actual_loss - expected_loss
    excess_pct = excess_loss / expected_loss if expected_loss > 0 else 0

    return {
        "fill_price": fill_price,
        "expected_loss": expected_loss,
        "actual_loss": actual_loss,
        "excess_loss": excess_loss,
        "excess_loss_pct": excess_pct,
    }


def daily_loss_limit_check(
    today_pnl_pct: float,
    daily_loss_threshold: float = -0.025,
) -> dict:
    """
    Halt new entries nếu daily loss exceeds threshold.

    Args:
        today_pnl_pct: Today's P&L as fraction of equity.
        daily_loss_threshold: Threshold (default -2.5%).

    Returns:
        Dict with allow_new_entries and message.
    """
    if today_pnl_pct < daily_loss_threshold:
        return {
            "allow_new_entries": False,
            "message": (
                f"Daily loss limit hit ({today_pnl_pct*100:.2f}%). "
                f"Halt entries 24h."
            ),
        }
    return {"allow_new_entries": True, "message": "Normal trading"}


def streak_loss_scaling(consecutive_losses: int) -> float:
    """
    Position size multiplier based on consecutive loss streak.

    - 0-4 losses: full size (1.0x)
    - 5-7 losses: half size (0.5x)
    - 8-9 losses: quarter size (0.25x)
    - 10+ losses: halt (0x)
    """
    if consecutive_losses < 5:
        return 1.0
    elif consecutive_losses < 8:
        return 0.5
    elif consecutive_losses < 10:
        return 0.25
    else:
        return 0.0


def vol_breakout_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    contract_value_per_point: float = 1.0,
    breakout_strength: float = 1.5,
    base_risk_pct: float = 0.008,
) -> float:
    """
    Position sizing cho vol breakout với strength scaling.

    Args:
        equity: Account equity.
        entry_price: Entry price.
        stop_price: Stop loss price.
        contract_value_per_point: Dollar per 1 point per 1 lot.
        breakout_strength: Range multiple of breakout candle (≥1.0).
        base_risk_pct: Base risk per trade (default 0.8%).

    Returns:
        Position size in lots.
    """
    if breakout_strength < 1.0:
        risk_pct = base_risk_pct * 0.5
    elif breakout_strength < 1.5:
        risk_pct = base_risk_pct * 0.75
    else:
        risk_pct = base_risk_pct

    risk_amount = equity * risk_pct
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return 0
    return risk_amount / (stop_distance * contract_value_per_point)


if __name__ == "__main__":
    print("=" * 70)
    print("Live Execution Helpers — Vol Breakout Demo")
    print("=" * 70)

    # Demo 1: News window detection
    print("\n--- News Window Detection ---")
    test_times = [
        pd.Timestamp("2024-03-08 13:30:00"),  # NFP first Friday March
        pd.Timestamp("2024-03-08 14:30:00"),  # 1h after NFP, still in window if 60min
        pd.Timestamp("2024-03-13 18:00:00"),  # FOMC Wednesday
        pd.Timestamp("2024-03-12 10:00:00"),  # Tuesday morning, no news
    ]
    for t in test_times:
        in_w = is_news_window(t, window_minutes=30)
        print(f"  {t}: in news window = {in_w}")

    # Demo 2: Gap-through stop
    print("\n--- Gap-Through Stop ---")
    scenarios = [
        ("Normal stop hit", 100, 95, 95, 1),
        ("1% gap down through stop", 100, 95, 94, 1),
        ("3% gap down (overnight crash)", 100, 95, 92, 1),
        ("Extreme gap (Black Monday)", 100, 95, 85, 1),
    ]
    for name, entry, stop, open_p, dir in scenarios:
        r = simulate_gap_through_stop(entry, stop, open_p, dir)
        print(f"  {name:<35}: fill ${r['fill_price']:.2f}, "
              f"actual loss ${r['actual_loss']:.2f} "
              f"(expected ${r['expected_loss']:.2f}, "
              f"excess {r['excess_loss_pct']*100:.0f}%)")

    # Demo 3: Daily loss limit
    print("\n--- Daily Loss Limit ---")
    for pnl in [0.005, -0.01, -0.025, -0.05]:
        r = daily_loss_limit_check(pnl, daily_loss_threshold=-0.025)
        print(f"  PnL {pnl*100:+.1f}% → allow_new={r['allow_new_entries']}  "
              f"| {r['message']}")

    # Demo 4: Streak loss scaling
    print("\n--- Streak Loss Scaling ---")
    for streak in [3, 5, 7, 9, 12]:
        mult = streak_loss_scaling(streak)
        print(f"  {streak} consecutive losses → size multiplier {mult}x")

    # Demo 5: Position sizing
    print("\n--- Vol Breakout Position Sizing ---")
    sizes = [
        ("Weak breakout (0.8 ATR)", 0.8),
        ("Medium (1.2 ATR)", 1.2),
        ("Strong (2.0 ATR)", 2.0),
    ]
    for name, strength in sizes:
        size = vol_breakout_size(
            equity=10000, entry_price=2030, stop_price=2010,
            contract_value_per_point=100, breakout_strength=strength,
        )
        print(f"  {name:<25}: {size:.4f} lots")

    # Demo 6: Slippage adjustment
    print("\n--- Slippage Adjustment ---")
    base = 100
    typical_spread = 0.05
    print(f"  Backtest fill: ${base}")
    print(f"  Normal session: "
          f"${simulate_slippage(base, 1, typical_spread):.3f}")
    print(f"  News window:    "
          f"${simulate_slippage(base, 1, typical_spread, is_news_window_active=True):.3f}")
    print(f"  Off-session:    "
          f"${simulate_slippage(base, 1, typical_spread, is_session_active=False):.3f}")
