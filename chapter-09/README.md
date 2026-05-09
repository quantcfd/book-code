# Chương 9 — Volatility & Breakout

Source code đầy đủ kèm Chương 9 của QuantCFD: 5 vol breakout strategies + filters + regime classifier + WFA + multi-strategy portfolio + 3-strategy combined system + 5 exercises.

## Files

### Core strategies (5 files)

| File | Mô tả |
|------|-------|
| `opening_range_breakout.py` | Strategy 1 — ORB intraday (indices, commodities) |
| `nr7_breakout.py` | Strategy 2 — NR7 / NR4 / IDnr7 daily breakout |
| `keltner_breakout.py` | Strategy 3 — Keltner channel breakout (FX, crypto, all markets) |
| `bb_squeeze.py` | Strategy 4 — Bollinger squeeze with Keltner inside detection |
| `volume_breakout.py` | Strategy 5 — Volume + price breakout (with range expansion fallback for FX) |

### Filters & risk (3 files)

| File | Mô tả |
|------|-------|
| `contraction_score.py` | 4-metric contraction score 0-100 (ATR ratio, BB width pct, NR7, IDnr7) |
| `regime_classifier_breakout.py` | IDEAL / POSSIBLE / AVOID classification (ATR pct, ADX, BB width) |
| `live_execution_breakout.py` | News filter, gap-through stops, daily loss limit, streak loss scaling, position sizing |

### Validation & combination (3 files)

| File | Mô tả |
|------|-------|
| `walk_forward_breakout.py` | WFA cho Keltner breakout — IS=24mo, OOS=6mo, step=3mo |
| `multi_strategy_breakout.py` | Combine 3 vol breakout strategies (Keltner + NR7 + BB Squeeze) |
| `combined_3_strategies.py` | Master 3-strategy portfolio: Trend (Ch7) + MR (Ch8) + Vol BO (Ch9) |

### Solutions (5 files)

| File | Bài tập |
|------|---------|
| `solution_exercise_1.py` | ORB từ scratch + verify no look-ahead |
| `solution_exercise_2.py` | NR7/NR4/IDnr7 detection + trading + comparison |
| `solution_exercise_3.py` | Keltner breakout 3-instrument + parameter sensitivity |
| `solution_exercise_6.py` | (BONUS) Complete vol breakout system với 5 filters |
| `solution_exercise_7.py` | (BONUS) 3-strategy portfolio (trend + MR + vol BO) |

## Quick start

```bash
# Test core strategies
python opening_range_breakout.py
python nr7_breakout.py
python keltner_breakout.py
python bb_squeeze.py
python volume_breakout.py

# Filters and regime
python contraction_score.py
python regime_classifier_breakout.py
python live_execution_breakout.py

# Combination
python multi_strategy_breakout.py
python combined_3_strategies.py

# Validation
python walk_forward_breakout.py

# Exercises
python solution_exercise_1.py
python solution_exercise_2.py
python solution_exercise_3.py
python solution_exercise_6.py
python solution_exercise_7.py
```

## Recommended workflow

1. **Đọc Ch9.3** → chạy `python opening_range_breakout.py` → ORB intraday
2. **Đọc Ch9.4** → chạy `python nr7_breakout.py` → narrow range patterns
3. **Đọc Ch9.4.5** → chạy `python contraction_score.py` → setup quality scoring
4. **Đọc Ch9.5** → chạy `python keltner_breakout.py` → channel breakout
5. **Đọc Ch9.6** → chạy `python bb_squeeze.py` → BB squeeze (best cho crypto)
6. **Đọc Ch9.7** → chạy `python volume_breakout.py` → volume confirmation
7. **Đọc Ch9.10** → chạy `python regime_classifier_breakout.py` → regime filter
8. **Đọc Ch9.11** → chạy `python live_execution_breakout.py` → execution helpers
9. **Đọc Ch9.12** → chạy `python multi_strategy_breakout.py` → portfolio
10. **Đọc Ch9.14** → chạy `python combined_3_strategies.py` → 3-class portfolio
11. **Đọc Ch9.17** → chạy `python walk_forward_breakout.py` → WFA validation

## Workflow validation 1 vol breakout strategy (sau khi đọc Ch9)

```bash
# Step 1: Pick strategy + market, run baseline
python keltner_breakout.py    # hoặc strategy khác

# Step 2: Add filters
python contraction_score.py
python regime_classifier_breakout.py

# Step 3: WFA validation
python walk_forward_breakout.py     # check OOS Sharpe ≥ 0.5

# Step 4: Production-ready system
python solution_exercise_6.py       # all 5 filters + risk controls

# Step 5: Combine với trend (Ch7) + MR (Ch8)
python combined_3_strategies.py
```

## Important notes

### Vol breakout characteristics
- **Win rate moderate**: 40-55% (vs trend 35-45%, MR 60-75%)
- **R:R high**: 2-3 (vs trend 2-4, MR 0.5-1)
- **Hold time medium**: hours đến days (vs trend weeks, MR days)
- **Tail risk**: gap-through stops trong news events

### 5 filters reduce fakeout 60% → 30%
1. **Vol contraction prerequisite** — `contraction_score >= 50`
2. **Volume / range expansion** — confirm institutional flow
3. **Time-of-day** — active sessions only
4. **News event blocker** — NFP, FOMC, CPI ±30 min
5. **Confirmation candle** — wait 1 bar before entry

### Cost matters for vol breakout
- Slippage model 1.5-2x typical spread
- 3-5x slippage during news events
- Strategy must be profitable với realistic costs

### Crypto best market for vol breakout
- High baseline volatility
- Clear regime changes (consolidation → expansion)
- Reliable volume from exchanges
- 24/7 trading (no session-based fakeouts)

### 3-strategy portfolio
Combining trend (Ch7) + MR (Ch8) + vol BO (Ch9):
- Single best strategy: Sharpe 1.0-1.4
- Combined portfolio: Sharpe 1.5-1.7
- Diversification due to low correlation across strategies
- Recommended allocation: balanced 45/30/25

### Synthetic data caveat
Demo trong `if __name__ == "__main__"` dùng synthetic data với regime variations. Numbers minh họa logic — đừng deploy với hyperparams này. Chạy trên real data từ broker historical.

## Dependencies

```
pandas>=2.0
numpy>=1.24
```

Tất cả files work without optional dependencies. Statsmodels không required (manual implementations đầy đủ).

## License

MIT — code kèm sách QuantCFD by Anthony Nguyễn / Monta Capital.
