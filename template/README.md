# Template — Cấu trúc dự án quant chuẩn

Đây là template được giảng ở **Chương 3, Section 3.6** của sách **QuantCFD**.

Anh em fork/copy thư mục này về làm seed cho dự án nghiên cứu quant cá nhân của mình. Đến **Chương 12**, repo cá nhân sẽ có 4 chiến lược hoàn chỉnh + một backtest engine tự build — đủ làm portfolio để apply việc.

## Cấu trúc

```
template/
├── data/
│   ├── raw/           # CSV/parquet gốc — KHÔNG BAO GIỜ EDIT
│   └── processed/     # đã clean, đã align timezone
├── notebooks/         # research, exploration
├── src/               # code reusable, có thể import
│   ├── __init__.py
│   ├── data.py        # data loading, cleaning
│   ├── indicators.py  # MA, RSI, ATR, Bollinger
│   ├── backtest.py    # engine (sẽ build ở Chương 5)
│   ├── metrics.py     # Sharpe, drawdown, expectancy
│   └── risk.py        # position sizing (Chương 10)
├── strategies/        # mỗi strategy 1 file
├── results/           # backtest output, không commit
├── tests/             # unit tests cho src/
├── requirements.txt   # danh sách package
├── .gitignore         # bỏ qua data/, results/
└── README.md          # mô tả dự án
```

## Quy tắc cốt lõi

1. **`data/raw/` không bao giờ edit.** Đây là source of truth. Cần clean → output ra `data/processed/`.
2. **`notebooks/` là nơi explore.** Code đẹp, reusable thì refactor sang `src/`.
3. **`src/` chứa logic dùng nhiều lần.** Function gọi từ ≥2 notebooks → thuộc về `src/`.
4. **Mỗi backtest run output ra `results/<date>_<name>/`** với timestamp. Không bao giờ ghi đè.
5. **`.gitignore` phải bỏ qua** `data/raw/`, `results/`, `.ipynb_checkpoints/`, `__pycache__/`.

## Cách dùng template

```bash
# Copy template ra thư mục riêng của anh em
cp -r template/ ~/projects/my-quantcfd-research/
cd ~/projects/my-quantcfd-research/

# Init git
git init
git add .
git commit -m "Initial commit from QuantCFD template"

# Tạo repo trên GitHub rồi:
git remote add origin https://github.com/YOUR_USERNAME/my-quantcfd-research.git
git push -u origin main
```

## Naming conventions

- `snake_case` cho variable + function: `rolling_sharpe`, `compute_drawdown`
- `PascalCase` cho class: `BacktestEngine`, `PortfolioManager`
- `UPPER_CASE` cho constant: `TRADING_DAYS_YEAR = 252`
- Tên file có nghĩa: `donchian_breakout.py`, không phải `untitled.ipynb`

## Module quan trọng (sẽ implement ở các chương sau)

- `src/indicators.py` — Chương 4 (Data + Indicators)
- `src/backtest.py` — Chương 5 (Backtest engine — chương quan trọng nhất)
- `src/metrics.py` — Chương 6 (Đo lường chiến lược)
- `src/risk.py` — Chương 10 (Risk management)

Hiện tại các module này còn empty (chỉ có docstring placeholder) — anh em sẽ fill khi đi qua các chương.
