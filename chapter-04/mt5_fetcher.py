"""
QuantCFD - Chapter 4 - MetaTrader 5 Data Fetcher
==================================================
Section 4.3: Lấy historical bars + tick data từ MT5.

Yêu cầu:
    - Windows OS (MT5 Python API không support Mac/Linux native)
    - MT5 desktop app cài + login broker
    - pip install MetaTrader5

Chạy (chỉ trên Windows):
    python chapter-04/mt5_fetcher.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone

import pandas as pd

try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False


def init_mt5(
    login: int | None = None,
    password: str | None = None,
    server: str | None = None,
) -> bool:
    """Initialize MT5 connection. Trả về True nếu thành công."""
    if not HAS_MT5:
        print("MetaTrader5 chưa cài. Run: pip install MetaTrader5")
        return False

    if login and password and server:
        ok = mt5.initialize(login=login, password=password, server=server)
    else:
        # Dùng credentials đã có sẵn trong MT5 desktop app
        ok = mt5.initialize()

    if not ok:
        print(f"MT5 init failed: {mt5.last_error()}")
        return False

    return True


def fetch_mt5_bars(
    symbol: str = "EURUSD",
    timeframe: str = "M1",
    count: int = 50000,
) -> pd.DataFrame:
    """
    Lấy N bars gần nhất từ MT5.

    Args:
        symbol: tên symbol broker dùng (vd "EURUSD", "XAUUSD", "US500").
        timeframe: "M1", "M5", "M15", "H1", "H4", "D1".
        count: số bars (max ~100k tuỳ broker).

    Returns:
        DataFrame với index=time UTC, columns=open/high/low/close/tick_volume/spread.
    """
    if not HAS_MT5:
        raise ImportError("MT5 chưa cài")

    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    if timeframe not in tf_map:
        raise ValueError(f"Timeframe phải là: {list(tf_map.keys())}")

    rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"Không lấy được data cho {symbol}. Error: {mt5.last_error()}"
        )

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    return df


def fetch_mt5_ticks(
    symbol: str = "EURUSD",
    from_date: datetime = None,
    count: int = 100000,
) -> pd.DataFrame:
    """
    Lấy tick data với bid/ask thực.

    Args:
        symbol: tên symbol.
        from_date: datetime UTC bắt đầu lấy (default: 7 ngày trước).
        count: số ticks.

    Returns:
        DataFrame với index=time UTC, columns=bid/ask/last/volume/spread_pips.
    """
    if not HAS_MT5:
        raise ImportError("MT5 chưa cài")

    if from_date is None:
        from datetime import timedelta
        from_date = datetime.now(tz=timezone.utc) - timedelta(days=7)

    ticks = mt5.copy_ticks_from(
        symbol, from_date, count, mt5.COPY_TICKS_ALL
    )
    if ticks is None or len(ticks) == 0:
        raise RuntimeError(f"Không có tick data cho {symbol}")

    df = pd.DataFrame(ticks)
    df["time"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
    df.set_index("time", inplace=True)

    # Tính spread (giả định EURUSD-style: pip = 0.0001)
    # Cho XAUUSD: pip = 0.01
    # Caller tự adjust nếu cần
    df["spread_raw"] = df["ask"] - df["bid"]

    return df


def main() -> None:
    if not HAS_MT5:
        print("CHƯA CÀI MetaTrader5. Trên Windows: pip install MetaTrader5")
        sys.exit(1)

    if not init_mt5():
        sys.exit(1)

    try:
        # Bars
        print("Fetching 50,000 bars EURUSD M1...")
        bars = fetch_mt5_bars("EURUSD", "M1", 50000)
        print(f"  Loaded {len(bars):,} bars from {bars.index.min()} to {bars.index.max()}")

        # Ticks (bid/ask thực)
        print("\nFetching 100,000 EURUSD ticks (last 7 days)...")
        ticks = fetch_mt5_ticks("EURUSD", count=100000)
        spread_pips = ticks["spread_raw"] * 1e4  # convert to pips
        print(f"  Loaded {len(ticks):,} ticks")
        print(f"  Median spread: {spread_pips.median():.2f} pips")
        print(f"  Mean spread:   {spread_pips.mean():.2f} pips")
        print(f"  95th pct:      {spread_pips.quantile(0.95):.2f} pips")
        print(f"  Max spread:    {spread_pips.max():.2f} pips")

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
