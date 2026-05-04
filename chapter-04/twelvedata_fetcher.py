"""
QuantCFD - Chapter 4 - TwelveData REST API Fetcher
====================================================
Section 4.4: Lấy data Indices/Commodities/Forex từ TwelveData.

Đăng ký free tier tại: https://twelvedata.com/
Free: 800 calls/day. Paid: từ $9/tháng cho 5000 calls/day.

Yêu cầu:
    pip install requests

Setup:
    Đặt API key vào biến môi trường TWELVEDATA_API_KEY, hoặc edit
    constant API_KEY trong file này.

Chạy:
    export TWELVEDATA_API_KEY=your_key
    python chapter-04/twelvedata_fetcher.py
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import requests


API_KEY = os.environ.get("TWELVEDATA_API_KEY", "YOUR_API_KEY_HERE")


def fetch_twelvedata(
    symbol: str,
    interval: str = "1h",
    start_date: str = "2023-01-01",
    end_date: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch OHLCV từ TwelveData REST API.

    Args:
        symbol: vd "SPX", "XAU/USD", "CL" (crude oil), "EUR/USD", "AAPL".
        interval: "1min", "5min", "15min", "30min", "1h", "4h", "1day".
        start_date: "YYYY-MM-DD".
        end_date: optional. None = đến hiện tại.
        api_key: optional override; default đọc từ env.
        timeout: HTTP timeout (giây).

    Returns:
        DataFrame với index=datetime UTC, columns=open/high/low/close/volume.
    """
    key = api_key or API_KEY
    if key == "YOUR_API_KEY_HERE":
        raise ValueError(
            "Set TWELVEDATA_API_KEY env variable hoặc edit API_KEY trong file."
        )

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "start_date": start_date,
        "apikey": key,
        "format": "JSON",
        "timezone": "UTC",
        "outputsize": 5000,  # max per call cho free/cheap plans
    }
    if end_date:
        params["end_date"] = end_date

    r = requests.get(url, params=params, timeout=timeout)
    data = r.json()

    if "values" not in data:
        msg = data.get("message", str(data))
        raise ValueError(f"TwelveData API error: {msg}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df.set_index("datetime", inplace=True)
    df = df.sort_index()

    # Convert numeric columns
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    return df


def main() -> None:
    if API_KEY == "YOUR_API_KEY_HERE":
        print("Set TWELVEDATA_API_KEY env variable trước khi chạy.")
        return

    # Demo: 3 instruments khác nhau
    instruments = [
        ("SPX", "1day", "S&P 500 Index"),
        ("XAU/USD", "1h", "Gold spot"),
        ("CL", "1day", "Crude Oil futures"),
    ]

    for symbol, interval, desc in instruments:
        print(f"\n{'='*60}")
        print(f"Fetching {symbol} ({desc}) at {interval}...")
        try:
            df = fetch_twelvedata(symbol, interval=interval, start_date="2024-01-01")
            print(f"  Loaded {len(df):,} bars")
            print(f"  Period: {df.index.min()} → {df.index.max()}")
            print(f"  Last close: {df['close'].iloc[-1]:.2f}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()
