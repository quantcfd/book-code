"""
QuantCFD — Chương 8.10.5
Live Execution helpers for MR strategies

5 issues addressed:
1. Limit order fill rate simulation
2. News event filter (avoid NFP, FOMC)
3. Spread expansion adjustment
4. Pairs trading fail-safe execution
5. Gap-through stop simulation
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, time


def simulate_limit_fill(
    df: pd.DataFrame,
    limit_price_col: str,
    direction: str = "buy",
    fill_rate: float = 0.70,
    seed: int = 42,
) -> pd.Series:
    """
    Simulate limit order fill rate.

    Backtest assumes 100% fill at limit price. Reality: ~70% fill rate
    when price oscillates around extreme without breaking through.

    Args:
        df: Price data with 'high', 'low' columns.
        limit_price_col: Column with limit prices.
        direction: 'buy' (limit below) or 'sell' (limit above).
        fill_rate: Probability of fill when price touches limit.
        seed: Random seed for reproducibility.

    Returns:
        Series of 1 (filled) or 0 (not filled).
    """
    rng = np.random.RandomState(seed)
    fills = pd.Series(0, index=df.index, dtype=int)

    for i in range(len(df)):
        limit = df[limit_price_col].iloc[i]
        if pd.isna(limit):
            continue

        # Check if price touched limit
        if direction == "buy":
            touched = df["low"].iloc[i] <= limit
        elif direction == "sell":
            touched = df["high"].iloc[i] >= limit
        else:
            raise ValueError(f"Unknown direction: {direction}")

        if touched and rng.random() < fill_rate:
            fills.iloc[i] = 1

    return fills


# Major news events that historically disrupt MR strategies
MAJOR_NEWS_EVENTS = {
    "NFP": "first_friday_of_month_8:30_ET",
    "CPI": "monthly_8:30_ET",
    "FOMC": "8_per_year_14:00_ET",
    "ECB": "monthly_7:45_ET",
    "BOJ": "8_per_year_variable",
}


def is_news_window(timestamp: pd.Timestamp, window_minutes: int = 15) -> bool:
    """
    Check if timestamp is within ±window_minutes of major news event.

    Simplified version — checks for typical news times:
    - NFP: First Friday of month, 8:30 ET (12:30 UTC, 19:30 VN)
    - CPI: ~12-15th of month, 8:30 ET
    - FOMC: ~8 times/year, 14:00 ET (18:00 UTC, 01:00 VN next day)

    Args:
        timestamp: Pandas timestamp (assumed UTC).
        window_minutes: Window size around news (default ±15 min).

    Returns:
        True if within news window.
    """
    # Convert to ET (UTC-5 standard, ignoring DST for simplicity)
    dt = timestamp - pd.Timedelta(hours=5)

    # NFP: First Friday of month, 8:30 ET
    if dt.weekday() == 4 and dt.day <= 7:
        nfp_time = dt.replace(hour=8, minute=30, second=0)
        if abs((dt - nfp_time).total_seconds()) <= window_minutes * 60:
            return True

    # CPI: typically 10-15th of month, 8:30 ET (approximate window)
    if 10 <= dt.day <= 15:
        cpi_time = dt.replace(hour=8, minute=30, second=0)
        if abs((dt - cpi_time).total_seconds()) <= window_minutes * 60:
            return True

    # FOMC: typically Wednesday 14:00 ET (rough — actual dates vary)
    if dt.weekday() == 2:  # Wednesday
        fomc_time = dt.replace(hour=14, minute=0, second=0)
        if abs((dt - fomc_time).total_seconds()) <= window_minutes * 60:
            return True

    return False


def filter_news_periods(
    signals: pd.Series, window_minutes: int = 15,
) -> pd.Series:
    """
    Zero-out signals during news windows.

    Args:
        signals: Trading signals (any non-zero = signal).
        window_minutes: News window size.

    Returns:
        Filtered signals (0 during news, original elsewhere).
    """
    filtered = signals.copy()
    for ts in signals.index:
        if is_news_window(ts, window_minutes):
            filtered.loc[ts] = 0
    return filtered


def adjust_spread_for_volatility(
    base_spread: float,
    realized_vol_ratio: float,
    cap: float = 5.0,
) -> float:
    """
    Adjust spread cost when vol is elevated.

    Backtest typically uses average spread. Reality: spread expands 2-5x
    during vol spikes. This function returns multiplier.

    Args:
        base_spread: Average spread cost.
        realized_vol_ratio: Current vol / average vol.
        cap: Max multiplier (default 5x).

    Returns:
        Adjusted spread.
    """
    multiplier = min(max(1.0, realized_vol_ratio), cap)
    return base_spread * multiplier


def simulate_gap_through_stop(
    entry_price: float,
    stop_price: float,
    actual_open_price: float,
    direction: str = "long",
) -> dict:
    """
    Simulate gap-through stop execution.

    Backtest assumes fill at stop_price. Reality: gap might fill far
    beyond stop. Models MR catastrophic loss (Black Monday, COVID, crypto crash).

    Args:
        entry_price: Position entry price.
        stop_price: Intended stop loss price.
        actual_open_price: Market open price (may gap).
        direction: 'long' or 'short'.

    Returns:
        Dict with fill_price, slippage, loss_vs_expected.
    """
    if direction == "long":
        if actual_open_price < stop_price:
            # Gap down through stop — fill at open
            fill_price = actual_open_price
            slippage = stop_price - actual_open_price
        else:
            fill_price = stop_price
            slippage = 0
        actual_loss = entry_price - fill_price
        expected_loss = entry_price - stop_price
    elif direction == "short":
        if actual_open_price > stop_price:
            fill_price = actual_open_price
            slippage = actual_open_price - stop_price
        else:
            fill_price = stop_price
            slippage = 0
        actual_loss = fill_price - entry_price
        expected_loss = stop_price - entry_price
    else:
        raise ValueError(f"Unknown direction: {direction}")

    excess_loss = actual_loss - expected_loss
    excess_loss_pct = (excess_loss / expected_loss) if expected_loss > 0 else 0

    return {
        "fill_price": fill_price,
        "slippage": slippage,
        "expected_loss": expected_loss,
        "actual_loss": actual_loss,
        "excess_loss": excess_loss,
        "excess_loss_pct": excess_loss_pct,
    }


def safe_pairs_entry(
    long_filled: bool, short_filled: bool, max_skew_seconds: int = 10,
) -> dict:
    """
    Pseudo-code for pairs entry fail-safe logic.

    For pairs trading, both legs must fill within seconds.
    If only 1 leg fills, abort to avoid naked single-asset exposure.

    Returns:
        Dict with action: 'proceed', 'close_long', 'close_short', 'retry'.
    """
    if long_filled and short_filled:
        return {"action": "proceed", "message": "Both legs filled, position safe"}
    elif long_filled and not short_filled:
        return {
            "action": "close_long",
            "message": (
                "Short leg failed to fill. Close long to avoid naked exposure."
            ),
        }
    elif short_filled and not long_filled:
        return {
            "action": "close_short",
            "message": (
                "Long leg failed to fill. Close short to avoid naked exposure."
            ),
        }
    else:
        return {"action": "retry", "message": "Neither leg filled, retry entry"}


if __name__ == "__main__":
    print("=" * 70)
    print("Live Execution Simulators — MR Strategies")
    print("=" * 70)

    # Demo 1: Limit fill rate
    print("\n--- Limit Order Fill Simulation ---")
    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.005))
    df = pd.DataFrame({
        "close": prices,
        "high": prices * (1 + np.abs(np.random.randn(n)) * 0.003),
        "low": prices * (1 - np.abs(np.random.randn(n)) * 0.003),
    }, index=dates)
    # Limit at 1% below close (BB lower band approximation)
    df["limit_buy"] = df["close"].rolling(20).mean() * 0.98

    fills = simulate_limit_fill(df, "limit_buy", direction="buy", fill_rate=0.70)
    print(f"Bars where limit touched: {((df['low'] <= df['limit_buy']) & ~df['limit_buy'].isna()).sum()}")
    print(f"Fills (70% rate):         {fills.sum()}")

    # Demo 2: News window detection
    print("\n--- News Window Detection ---")
    test_times = [
        pd.Timestamp("2024-03-08 13:30:00"),  # NFP first Friday March 8:30 ET = 13:30 UTC
        pd.Timestamp("2024-03-08 12:00:00"),  # 1.5h before NFP — should be False
        pd.Timestamp("2024-04-10 13:30:00"),  # CPI day around 13:30 UTC
        pd.Timestamp("2024-03-13 18:00:00"),  # FOMC Wednesday 14:00 ET = 18:00 UTC
    ]
    for t in test_times:
        in_window = is_news_window(t, window_minutes=30)
        print(f"  {t}: in news window = {in_window}")

    # Demo 3: Spread adjustment
    print("\n--- Spread Expansion ---")
    base = 0.0003  # 3 pips XAUUSD
    for vol_ratio in [1.0, 2.0, 5.0, 10.0]:
        adj = adjust_spread_for_volatility(base, vol_ratio, cap=5.0)
        print(f"  Vol ratio {vol_ratio:4.1f}x → adjusted spread = {adj:.5f} ({adj/base:.1f}x base)")

    # Demo 4: Gap-through stop
    print("\n--- Gap-Through Stop Simulation ---")
    scenarios = [
        ("Normal stop", 100, 95, 95),
        ("1% gap through", 100, 95, 94),
        ("3% gap through", 100, 95, 92),
        ("Black Monday gap", 100, 95, 85),
    ]
    for name, entry, stop, actual_open in scenarios:
        r = simulate_gap_through_stop(entry, stop, actual_open, "long")
        print(f"  {name:20s}: fill={r['fill_price']:.0f}  "
              f"actual loss=${r['actual_loss']:.0f}  "
              f"expected=${r['expected_loss']:.0f}  "
              f"excess={r['excess_loss_pct']*100:.0f}%")

    # Demo 5: Pairs fail-safe
    print("\n--- Pairs Trading Fail-Safe ---")
    for long_f, short_f in [(True, True), (True, False), (False, True), (False, False)]:
        r = safe_pairs_entry(long_f, short_f)
        print(f"  long={long_f}, short={short_f}: action={r['action']:20s} | {r['message']}")
