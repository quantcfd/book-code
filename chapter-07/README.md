# Chương 7 — Trend Following

Source code đầy đủ kèm Chương 7 của QuantCFD: 4 chiến lược trend following + position sizing + trailing stops + regime filters + multi-instrument portfolio + crypto-specific + walk-forward analysis.

## Files

### Core strategies

| File | Mô tả |
|------|-------|
| `ma_crossover.py` | Strategy 1 — Moving Average crossover với anti-look-ahead, param sensitivity |
| `donchian.py` | Strategy 2 — Donchian channel breakout (Turtle System 1+2) |
| `triple_ma.py` | Strategy 3 — Triple MA alignment + MACD confirmation |
| `momentum_ranking.py` | Strategy 4 — Cross-sectional momentum ranking (long top-N, short bottom-N) |
| `crypto_trend.py` | Crypto-specific trend với vol targeting + funding rate awareness |

### Position sizing & risk

| File | Mô tả |
|------|-------|
| `atr_sizing.py` | ATR computation + Turtle unit sizing + vol targeting + asset class table |
| `pyramiding.py` | Turtle add-on rules (max 4 units, raise stop sync, 0.5N intervals) |
| `trailing_stops.py` | 5 phương pháp: fixed %, ATR, Donchian, Chandelier, Parabolic SAR |

### Filters & validation

| File | Mô tả |
|------|-------|
| `regime_filters.py` | ADX + SMA200 + vol regime + multi-TF, combine modes |
| `walk_forward.py` | Rolling IS/OOS analysis với WFE metric, GO/NO-GO verdict |
| `multi_instrument.py` | Run strategy trên 6 instruments, vol-parity weighting, correlation matrix |

### Solutions

| File | Bài tập |
|------|---------|
| `solution_exercise_1.py` | MA crossover từ scratch + verify no look-ahead |
| `solution_exercise_2.py` | Donchian System 1 + System 2 trên BTCUSD |
| `solution_exercise_6.py` | Walk-forward MA params trên 3 instruments (BONUS) |

## Quick start

```bash
# Run any module standalone (có demo built-in)
python ma_crossover.py
python donchian.py
python atr_sizing.py
python regime_filters.py
python pyramiding.py
python walk_forward.py
python multi_instrument.py
python crypto_trend.py
```

## Dependencies

Cùng requirements.txt với toàn bộ project:

```
pandas>=2.0
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
```

Optional:
```
seaborn  # cho heatmap visualization
```

## Workflow recommended

1. **Đọc Ch7.3** → chạy `python ma_crossover.py` → hiểu MA crossover output
2. **Đọc Ch7.4** → chạy `python donchian.py` → so sánh System 1 vs 2
3. **Đọc Ch7.7** → chạy `python atr_sizing.py` → hiểu Turtle position sizing
4. **Đọc Ch7.7.5** → chạy `python pyramiding.py` → so sánh 1/2/4/6 units
5. **Đọc Ch7.8** → chạy `python trailing_stops.py` → benchmark 4 methods
6. **Đọc Ch7.9** → chạy `python regime_filters.py` → xem filter pass rate
7. **Đọc Ch7.4.5** → chạy `python walk_forward.py` → validate strategy
8. **Đọc Ch7.10** → chạy `python multi_instrument.py` → portfolio benefit
9. **Đọc Ch7.11.3** → chạy `python crypto_trend.py` → crypto specifics

## Important notes

### Anti look-ahead bias
Tất cả modules dùng `.shift(1)` trên signal. Đừng remove shift — sẽ tạo Sharpe ảo.

### Synthetic data
Demo trong `if __name__ == "__main__"` dùng synthetic random walk + drift, KHÔNG phải data thực. Numbers chỉ minh họa cấu trúc — đừng deploy với hyperparams này.

### Cost models
Default cost = 0.05% (5 bps) round-trip. Realistic cho XAU H4 nhưng:
- HIGH cho EURUSD daily (overestimate)
- LOW cho BTC scalping (underestimate)
- Adjust theo broker spec của anh em

### Position sizing
`turtle_unit_size()` returns **lots**, không phải dollar amount. Validate với broker contract specs trước khi deploy.

## License

MIT — code kèm sách QuantCFD by Anthony Nguyen / Monta Capital.
