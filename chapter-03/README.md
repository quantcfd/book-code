# Chương 3 — Python cho trader (crash course)

Code cho Chương 3 của sách **QuantCFD**.

## Files

| File | Mô tả | Section trong sách |
|---|---|---|
| `speed_test.py` | For-loop vs numpy — demo 200x+ speedup | 3.2 |
| `max_drawdown_numpy.py` | Max drawdown bằng 4 dòng numpy | 3.2 |
| `pandas_patterns.py` | 10 thao tác pandas chính trên Gold data | 3.3 |
| `matplotlib_5_charts.py` | 5 chart types tạo file PNG | 3.4 |
| `ten_lines_pandas.py` | 6 task quant: pandas vs VBA | 3.5 |
| `solution_exercise_2.py` | **Lời giải Bài tập 2** — 10 pandas drills trên Gold | Bài tập |

## Chạy

```bash
python chapter-03/speed_test.py
python chapter-03/max_drawdown_numpy.py
python chapter-03/pandas_patterns.py
python chapter-03/matplotlib_5_charts.py        # Tạo 5 file PNG
python chapter-03/ten_lines_pandas.py
python chapter-03/solution_exercise_2.py
```

## Output từ matplotlib_5_charts.py

Sẽ tạo 5 file PNG trong thư mục chạy:

- `chart1_equity.png` — equity curve (line chart)
- `chart2_histogram.png` — distribution of returns
- `chart3_scatter.png` — BTC vs ETH correlation scatter
- `chart4_heatmap.png` — correlation matrix 5 instruments
- `chart5_subplots.png` — 4-panel strategy report

`.gitignore` đã loại trừ `*.png` ngoại trừ trong `docs/` — không cần lo commit nhầm.

## Bài tập tự làm thêm

- **Bài 1:** Vectorization Speed Test — chạy `speed_test.py`, sau đó viết bài toán tương tự (max drawdown) bằng 2 cách.
- **Bài 3:** Setup repo cá nhân theo cấu trúc trong `template/` ở root repo này.

## Pattern dùng lại nhiều lần

Các functions trong `pandas_patterns.py` và `ten_lines_pandas.py` được tái sử dụng ở các chương sau (đặc biệt Chương 5 backtest engine + Chương 6 metrics). Học thuộc patterns này, anh em sẽ đỡ phải Google nhiều khi code chương sau.
