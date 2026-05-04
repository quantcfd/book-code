"""
QuantCFD - Chapter 4 - Bài 1 Solution: Unified Data Module
=============================================================
Lời giải Bài tập 1 / Chương 4: Build module data.py cho 3 thị trường.

3 functions chuẩn hoá:
    fetch_crypto(symbol, timeframe, since)
    fetch_forex(symbol, timeframe, since)
    fetch_index_or_commodity(symbol, timeframe, since)

Tất cả:
    - Trả DataFrame index=datetime UTC, cột open/high/low/close/volume
    - Cache vào data/raw/{symbol}_{timeframe}.parquet
    - Skip nếu file đã có (cache locally)

Yêu cầu:
    pip install ccxt requests pyarrow
    (Optional cho forex MT5: pip install MetaTrader5 — chỉ Windows)

Chạy:
    python chapter-04/solution_exercise_1.py
"""
from __future__ import annotations

import os
from pathlib import Path

import ccxt
import pandas as pd
import requests


CACHE_DIR = Path("data/raw")
TWELVEDATA_KEY = os.environ.get("TWELVEDATA_API_KEY", "")


def _cache_path(symbol: str, timeframe: str, source: str) -> Path:
    """Build cache file path. Sanitize / trong symbol thành _."""
    safe_symbol = symbol.replace("/", "_").replace(":", "_")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{source}_{safe_symbol}_{timeframe}.parquet"


def fetch_crypto(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    since_date: str = "2023-01-01",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch crypto OHLCV từ Binance qua ccxt.

    Args:
        symbol: ccxt format ("BTC/USDT", "ETH/USDT").
        timeframe: "1m", "5m", "15m", "1h", "4h", "1d".
        since_date: "YYYY-MM-DD".
        use_cache: True = load từ cache nếu đã tải; False = tải lại.

    Returns:
        DataFrame index=datetime UTC, columns=open/high/low/close/volume.
    """
    cache = _cache_path(symbol, timeframe, "binance")

    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    exchange = ccxt.binance({"enableRateLimit": True})
    since = exchange.parse8601(f"{since_date}T00:00:00Z")

    all_candles: list[list] = []
    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not candles:
            break
        all_candles.extend(candles)
        since = candles[-1][0] + 1
        if len(candles) < 1000:
            break

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    df.drop(columns=["timestamp"], inplace=True)

    df.to_parquet(cache)
    print(f"[CRYPTO] Fetched {len(df):,} bars of {symbol} → {cache}")
    return df


def fetch_forex(
    symbol: str = "EUR/USD",
    timeframe: str = "1h",
    since_date: str = "2023-01-01",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch Forex từ TwelveData (cách dễ nhất, cross-platform).

    Cho lịch sử dài + tick precision: dùng dukascopy_fetcher hoặc mt5_fetcher.

    Args:
        symbol: "EUR/USD", "GBP/USD" (TwelveData format có "/").
        timeframe: "1min", "5min", "15min", "1h", "4h", "1day".
        since_date: "YYYY-MM-DD".
        use_cache: True = dùng cache.

    Returns:
        DataFrame chuẩn hoá.
    """
    cache = _cache_path(symbol, timeframe, "twelvedata")

    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    if not TWELVEDATA_KEY:
        raise ValueError(
            "Set TWELVEDATA_API_KEY env variable. "
            "Free signup tại twelvedata.com."
        )

    # TwelveData dùng "1h" thay vì "1h"; map nếu cần
    interval = timeframe if timeframe != "1h" else "1h"

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "start_date": since_date,
        "apikey": TWELVEDATA_KEY,
        "format": "JSON",
        "timezone": "UTC",
        "outputsize": 5000,
    }
    r = requests.get(url, params=params, timeout=30)
    data = r.json()

    if "values" not in data:
        raise ValueError(f"TwelveData error: {data.get('message', data)}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df.set_index("datetime", inplace=True)
    df = df.sort_index()
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)

    # TwelveData không trả volume cho Forex → thêm cột rỗng
    if "volume" not in df.columns:
        df["volume"] = pd.NA

    df.to_parquet(cache)
    print(f"[FOREX]  Fetched {len(df):,} bars of {symbol} → {cache}")
    return df


def fetch_index_or_commodity(
    symbol: str = "SPX",
    timeframe: str = "1day",
    since_date: str = "2023-01-01",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch indices hoặc commodity CFD từ TwelveData.

    Args:
        symbol: "SPX", "NDX", "DJI" (indices); "XAU/USD", "CL", "NG" (commodity).
        timeframe: "1min", "5min", "15min", "1h", "4h", "1day".
        since_date: "YYYY-MM-DD".
        use_cache: True = dùng cache.

    Returns:
        DataFrame chuẩn hoá.
    """
    cache = _cache_path(symbol, timeframe, "twelvedata_idx")

    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    if not TWELVEDATA_KEY:
        raise ValueError(
            "Set TWELVEDATA_API_KEY env variable. "
            "Free signup tại twelvedata.com."
        )

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": timeframe,
        "start_date": since_date,
        "apikey": TWELVEDATA_KEY,
        "format": "JSON",
        "timezone": "UTC",
        "outputsize": 5000,
    }
    r = requests.get(url, params=params, timeout=30)
    data = r.json()

    if "values" not in data:
        raise ValueError(f"TwelveData error: {data.get('message', data)}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df.set_index("datetime", inplace=True)
    df = df.sort_index()
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    else:
        df["volume"] = pd.NA

    df.to_parquet(cache)
    print(f"[INDEX]  Fetched {len(df):,} bars of {symbol} → {cache}")
    return df


def main() -> None:
    print("=" * 60)
    print(" UNIFIED DATA MODULE — TEST 3 MARKETS")
    print("=" * 60)

    # Test crypto
    try:
        btc = fetch_crypto("BTC/USDT", "1h", "2024-01-01")
        print(f"BTC: {len(btc):,} bars, latest close = {btc['close'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"BTC error: {e}")

    # Test forex (cần TWELVEDATA_API_KEY)
    if TWELVEDATA_KEY:
        try:
            eur = fetch_forex("EUR/USD", "1h", "2024-01-01")
            print(f"EUR/USD: {len(eur):,} bars")
        except Exception as e:
            print(f"EUR/USD error: {e}")

        # Test index
        try:
            spx = fetch_index_or_commodity("SPX", "1day", "2024-01-01")
            print(f"SPX: {len(spx):,} bars")
        except Exception as e:
            print(f"SPX error: {e}")
    else:
        print("\nSkipping Forex/Index test — set TWELVEDATA_API_KEY env var.")

    print("\nCache files:")
    if CACHE_DIR.exists():
        for f in sorted(CACHE_DIR.iterdir()):
            print(f"  {f.name}  ({f.stat().st_size / 1e3:.1f} KB)")

    print(
        "\nModule này có thể import từ chương sau:\n"
        "    from chapter_04.solution_exercise_1 import fetch_crypto\n"
        "    btc = fetch_crypto('BTC/USDT', '1h', '2023-01-01')\n"
        "Hoặc copy 3 functions vào src/data.py của repo cá nhân anh em."
    )


if __name__ == "__main__":
    main()
