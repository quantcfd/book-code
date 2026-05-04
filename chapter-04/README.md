# Chương 4 — Dữ liệu: máu của quant

Code cho Chương 4 của sách **QuantCFD**.

## Files

| File | Mô tả | Section |
|---|---|---|
| `binance_fetcher.py` | OHLCV crypto từ Binance via ccxt (auto-paginate) | 4.2 |
| `binance_funding.py` | Funding rate history + analysis cho perpetual | 4.2 |
| `dukascopy_fetcher.py` | Forex tick data lịch sử miễn phí | 4.3 |
| `mt5_fetcher.py` | MT5 bars + tick data với bid/ask thực | 4.3 |
| `twelvedata_fetcher.py` | Indices/Commodity/Forex REST API | 4.4 |
| `spread_session_analyzer.py` | Spread theo giờ UTC + heatmap (Bài 2) | 4.3, Bài 2 |
| `rollover_detector.py` | Phát hiện gap rollover trong futures-based CFD | 4.4 |
| `storage_benchmark.py` | CSV vs Parquet vs SQLite benchmark (Bài 3) | 4.7, Bài 3 |
| `solution_exercise_1.py` | **Lời giải Bài 1** — unified data module | Bài 1 |

## Yêu cầu cài đặt

```bash
pip install ccxt pandas numpy matplotlib pyarrow requests
pip install duka                 # cho Dukascopy (optional)
pip install MetaTrader5          # cho MT5 (Windows only, optional)
```

## API Keys cần có

- **Binance**: free, không cần key cho public OHLCV (đã include trong `binance_fetcher.py`)
- **TwelveData**: free signup tại [twelvedata.com](https://twelvedata.com/), 800 calls/day. Set env var:
  ```bash
  export TWELVEDATA_API_KEY=your_key_here    # Mac/Linux
  set TWELVEDATA_API_KEY=your_key_here       # Windows CMD
  $env:TWELVEDATA_API_KEY="your_key_here"    # PowerShell
  ```

## Chạy

```bash
# Crypto (không cần API key)
python chapter-04/binance_fetcher.py
python chapter-04/binance_funding.py

# Forex (cần Dukascopy hoặc MT5)
python chapter-04/dukascopy_fetcher.py
python chapter-04/mt5_fetcher.py             # Windows only

# Indices/Commodity (cần TwelveData key)
python chapter-04/twelvedata_fetcher.py

# Analysis tools
python chapter-04/spread_session_analyzer.py # demo với synthetic data
python chapter-04/rollover_detector.py       # dùng yfinance ES=F

# Storage benchmark
python chapter-04/storage_benchmark.py

# Bài 1 solution: unified module
python chapter-04/solution_exercise_1.py
```

## Module quan trọng dùng lại nhiều chương sau

`solution_exercise_1.py` chứa 3 functions chuẩn hoá:

```python
from chapter_04.solution_exercise_1 import (
    fetch_crypto,
    fetch_forex,
    fetch_index_or_commodity,
)

btc = fetch_crypto("BTC/USDT", "1h", "2023-01-01")
eur = fetch_forex("EUR/USD", "1h", "2023-01-01")
spx = fetch_index_or_commodity("SPX", "1day", "2020-01-01")
```

Tất cả tự cache vào `data/raw/*.parquet` — gọi lại lần 2 không tốn API call.

## Land mines cần tránh

1. **yfinance daily close là 00:00 UTC** — nếu chiến lược thực thi giờ broker (giờ NY hoặc giờ London), backtest sẽ lệch 7-8 tiếng.
2. **Spread CFD biến động theo session** — backtest spread cố định cho EURUSD scalping = sai có hệ thống.
3. **Crypto top 10 hôm nay ≠ top 10 hai năm trước** — cherry-pick winners.
4. **Indices/commodity CFD dựa trên futures** — gap rollover có thể giả lệnh giả vờ thắng.

Đọc kỹ Chương 4 sách để hiểu chi tiết mỗi vấn đề.
