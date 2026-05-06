# Chương 8 — Mean Reversion

Source code đầy đủ kèm Chương 8 của QuantCFD: 4 chiến lược mean reversion + statistical validation + risk management + walk-forward + stat arb extensions + crypto MR + multi-strategy portfolio + live execution helpers.

## Files

### Core strategies (5 files)

| File | Mô tả |
|------|-------|
| `bollinger_reversion.py` | Strategy 1 — Bollinger Bands MR (volatility-based) |
| `connors_rsi.py` | Strategy 2 — Connors RSI extreme (sentiment-based) |
| `zscore_strategy.py` | Strategy 3 — Z-score MR (statistical-based) |
| `pairs_trading.py` | Strategy 4 — Pairs trading với cointegration |
| `regime_filters.py` | Filter để chỉ trade khi market mean revert (ADX, vol, multi-TF) |

### Statistical validation & risk (3 files)

| File | Mô tả |
|------|-------|
| `statistical_tests.py` | Hurst exponent + ADF test + half-life calc + validation report |
| `risk_management_mr.py` | MR-specific position sizing, time stops, correlation alerts, DD scale-down |
| `walk_forward_mr.py` | Walk-forward analysis cho MR (IS/OOS calibrated) |

### Advanced (3 files)

| File | Mô tả |
|------|-------|
| `stat_arb_extension.py` | Johansen multi-asset cointegration + Kalman dynamic hedge ratio |
| `crypto_mr.py` | Funding rate strategy, BTC/ETH ratio MR, basis trade |
| `multi_strategy_mr.py` | Combine 4 MR strategies vào portfolio (diversification) |
| `live_execution_mr.py` | Limit order fills, news filter, gap-through stops, pairs fail-safe |

### Solutions (5 files)

| File | Bài tập |
|------|---------|
| `solution_exercise_1.py` | BB MR từ scratch + verify no look-ahead |
| `solution_exercise_2.py` | Pairs trading XAU-XAG với cointegration test |
| `solution_exercise_3.py` | Cross-sectional Z-score (6 FX pairs) |
| `solution_exercise_6.py` | Statistical tests trên 5 instruments (BONUS) |
| `solution_exercise_7.py` | Yin-yang portfolio (trend + MR) (BONUS) |

## Quick start

```bash
# Test core MR strategies
python bollinger_reversion.py
python connors_rsi.py
python zscore_strategy.py
python pairs_trading.py

# Validation BEFORE deploying any strategy
python statistical_tests.py     # Hurst + ADF + half-life

# Risk management
python risk_management_mr.py    # Position sizing + correlation alerts
python walk_forward_mr.py       # WFA validation

# Advanced
python stat_arb_extension.py    # Johansen + Kalman
python crypto_mr.py             # Funding rate strategies
python multi_strategy_mr.py     # 4-strategy portfolio
python live_execution_mr.py     # Live trading helpers
```

## Recommended workflow

1. **Đọc Ch8.5.5** → chạy `python statistical_tests.py` → hiểu Hurst/ADF/half-life
2. **Đọc Ch8.3** → chạy `python bollinger_reversion.py` → BB strategy
3. **Đọc Ch8.4** → chạy `python connors_rsi.py` → RSI strategy
4. **Đọc Ch8.5** → chạy `python zscore_strategy.py` → Z-score strategy
5. **Đọc Ch8.6** → chạy `python pairs_trading.py` → pairs trading
6. **Đọc Ch8.7.5** → chạy `python risk_management_mr.py` → risk controls
7. **Đọc Ch8.13.5** → chạy `python walk_forward_mr.py` → validate strategy
8. **Đọc Ch8.13.7** → chạy `python stat_arb_extension.py` → advanced techniques
9. **Đọc Ch8.10.7** → chạy `python crypto_mr.py` → crypto-specific
10. **Đọc Ch8.10.5** → chạy `python live_execution_mr.py` → execution issues

## Workflow validation 1 MR strategy mới (sau khi đọc Ch8)

```bash
# Step 1: Hurst + ADF + half-life
python statistical_tests.py

# Step 2: Backtest trên 5 năm
python bollinger_reversion.py    # hoặc strategy khác

# Step 3: Walk-forward validation
python walk_forward_mr.py        # check OOS Sharpe ≥ 0.5

# Step 4: Risk management
python risk_management_mr.py     # verify position sizing logic

# Step 5: Live execution simulation
python live_execution_mr.py      # check cost-to-edge ratio

# Step 6: Combine với strategies khác
python multi_strategy_mr.py      # diversification benefit
```

## Important notes

### MR ≠ Trend
Mean reversion có rủi ro DIFFERENT từ trend:
- Win rate cao (60-75%) NHƯNG tail risk lớn
- 1 trade tệ có thể wipe 5-8 trades thắng
- Position sizing PHẢI conservative hơn (0.3-0.7% per trade)
- Time-based stops critical (max 2× half-life)

### Statistical validation BẮT BUỘC
KHÔNG bao giờ deploy MR strategy mà không pass:
- Hurst < 0.5 trên series được trade
- ADF p-value < 0.05
- Half-life finite và reasonable (5-50 bars)

90% MR strategies retail fail chính do skip 3 tests này.

### Cost matters more for MR
MR turnover 2-5x trend → cost ratio cao hơn nhiều:
- Trend cost/profit ratio: 3-5%
- MR cost/profit ratio: 10-30%

Nếu cost-to-edge ratio > 30%, abandon strategy hoặc tăng timeframe.

### Synthetic data caveat
Demo trong `if __name__ == "__main__"` dùng synthetic data. Numbers minh họa logic — đừng deploy với hyperparams này. Chạy trên real data từ broker historical.

## Dependencies

```
pandas>=2.0
numpy>=1.24
```

Optional (cho advanced features):
```
statsmodels  # ADF test, Johansen cointegration
pykalman     # Kalman filter dynamic hedge ratio
scipy        # statistical tests
```

Hầu hết files work without optional deps (fallback implementations).

## License

MIT — code kèm sách QuantCFD by Anthony Nguyễn / Monta Capital.
