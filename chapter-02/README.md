# Chương 2 — Tư duy thống kê cho trader

Code cho Chương 2 của sách **QuantCFD**.

## Files

| File | Mô tả | Section trong sách |
|---|---|---|
| `expectancy_calculator.py` | Hàm `expectancy_report()` + demo 3 chiến lược | 2.1 |
| `fat_tails_visualizer.py` | Phân tích kurtosis + plot fat tails BTC | 2.2 |
| `stationarity_tester.py` | ADF test trên price/returns/spread + rolling check | 2.3 |
| `rolling_correlation.py` | Rolling 90-day correlation cho 3 cặp asset | 2.4 |
| `sharpe_significance.py` | Hàm `sharpe_significance()` + bảng năm cần thiết | 2.5 |
| `lookahead_bug_detector.py` | Demo Sharpe inflation do look-ahead bug | 2.6 |
| `solution_exercise_3.py` | **Lời giải Bài tập 3** — tìm 2 bug trong RSI strategy | Bài tập |

## Chạy

```bash
# Từ root của repo
python chapter-02/expectancy_calculator.py
python chapter-02/fat_tails_visualizer.py
python chapter-02/stationarity_tester.py
python chapter-02/rolling_correlation.py
python chapter-02/sharpe_significance.py
python chapter-02/lookahead_bug_detector.py
python chapter-02/solution_exercise_3.py
```

## Modules quan trọng

Hai functions sau được dùng lại nhiều lần ở các chương sau:

```python
from chapter_02.expectancy_calculator import expectancy_report
from chapter_02.sharpe_significance   import sharpe_significance
```

## Output của mỗi script

- `fat_tails_visualizer.py` → `btc_fat_tails.png`
- `rolling_correlation.py` → `rolling_correlation.png`

Các file PNG sẽ được tạo trong thư mục hiện tại — `.gitignore` đã loại trừ.

## Bài tập tự làm thêm

- **Bài 1:** Audit trade log cá nhân bằng `expectancy_calculator.py`. Tính expectancy của hệ thống đang chạy.
- **Bài 2:** Chạy `fat_tails_visualizer.py` cho cả 4 instruments (BTC, ETH, GC=F, ES=F). So sánh kurtosis.

Lời giải sẽ được up dần khi anh em hỏi trên Discord [discord.gg/CC6xsZ8tcf](https://discord.gg/CC6xsZ8tcf).
