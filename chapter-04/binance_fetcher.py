"""
QuantCFD - Chapter 4 - Binance OHLCV Fetcher
==============================================
Section 4.2: Lấy data crypto từ Binance qua ccxt.

Function fetch_binance_ohlcv() tự động paginate để lấy toàn bộ data
từ since_date đến hiện tại, không bị giới hạn 1000 candle/call.

Yêu cầu:
    pip install ccxt pyarrow

Chạy:
    python chapter-04/binance_fetcher.py
"""
from __future__ import annotations

import ccxt
import pandas as pd


def fetch_binance_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    since_date: str = "2023-01-01",
    limit_per_call: int = 1000,
) -> pd.DataFrame:
    """
    Fetch toàn bộ OHLCV từ Binance, paginate tự động.

    Args:
        symbol: ccxt symbol format, vd "BTC/USDT", "ETH/USDT".
        timeframe: "1m", "5m", "15m", "1h", "4h", "1d", v.v.
        since_date: ngày bắt đầu (YYYY-MM-DD).
        limit_per_call: số candle mỗi API call (max 1000 cho Binance).

    Returns:
        DataFrame với index=datetime UTC, columns=open/high/low/close/volume.
    """
    exchange = ccxt.binance({"enableRateLimit": True})
    since = exchange.parse8601(f"{since_date}T00:00:00Z")

    all_candles: list[list] = []
    while True:
        candles = exchange.fetch_ohlcv(
            symbol, timeframe, since=since, limit=limit_per_call
        )
        if not candles:
            break
        all_candles.extend(candles)
        since = candles[-1][0] + 1  # next millisecond
        if len(candles) < limit_per_call:
            break  # đã hết data

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    df.drop(columns=["timestamp"], inplace=True)
    return df


def main() -> None:
    print("Fetching BTC/USDT 1h từ Binance...")
    btc = fetch_binance_ohlcv("BTC/USDT", "1h", "2023-01-01")
    print(f"Loaded {len(btc):,} bars")
    print(f"\nFirst 3 rows:")
    print(btc.head(3))
    print(f"\nLast 3 rows:")
    print(btc.tail(3))

    # Save lại để các script sau dùng
    out_path = "btc_1h.parquet"
    btc.to_parquet(out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
