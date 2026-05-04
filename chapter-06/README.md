# Chương 6 — Đo lường chiến lược

Source code đầy đủ kèm Chương 6 của QuantCFD: 40+ metrics + tear sheet generator + portfolio analytics + pitfall detection + live monitoring.

## Files

### Core modules

| File | Mô tả |
|------|-------|
| `metrics.py` | Đầy đủ 40+ metrics: returns, risk, risk-adjusted ratios, trade-level, tail risk, stability |
| `tear_sheet.py` | Generate 1-page PDF tear sheet với 6 panels (equity, drawdown, rolling Sharpe, monthly heatmap, trades, yearly) |
| `portfolio_metrics.py` | Multi-strategy: correlation matrix, MCR, risk parity, Herfindahl, leverage |
| `live_monitor.py` | Live trading: tracking error, alerts, StopTradingRules class, daily summary |
| `pitfall_detection.py` | Detect 7 bad-faith tricks: cherry-picking, overfit, survivorship, costs, leverage, start date, CAGR-only |

### Solutions

| File | Bài tập |
|------|---------|
| `solution_exercise_1.py` | Implement metrics module + 8 unit tests |
| `solution_exercise_2.py` | Tear sheet trên 3 strategies |
| `solution_exercise_3.py` | Portfolio analysis (correlation, RP, MCR) |
| `solution_exercise_4.py` | Pitfall audit 3 simulated vendors |
| `solution_exercise_5.py` | Stress test 5 scenarios |
| `solution_exercise_6.py` | Strategy ranking system (BONUS) |
| `solution_exercise_7.py` | Live monitoring dashboard (BONUS, Streamlit) |

## Quick start

```bash
# Chạy demo của metrics
python metrics.py

# Generate tear sheet PDF
python tear_sheet.py

# Portfolio analysis demo
python portfolio_metrics.py

# Live monitoring demo
python live_monitor.py

# Pitfall detection demo
python pitfall_detection.py

# Run all exercise solutions
python solution_exercise_1.py
python solution_exercise_2.py
python solution_exercise_3.py
python solution_exercise_4.py
python solution_exercise_5.py
python solution_exercise_6.py
python solution_exercise_7.py    # CLI mode
streamlit run solution_exercise_7.py    # Web dashboard
```

## Dependencies

```bash
pip install numpy pandas scipy matplotlib seaborn pytest streamlit
```

Tối thiểu cho metrics + tear_sheet: `numpy pandas scipy matplotlib`. Seaborn cải thiện heatmap. Streamlit cho dashboard. Pytest cho exercise 1.

## Sample usage

### Compute metrics cho backtest output

```python
import pandas as pd
from metrics import returns_summary, sharpe_ratio, max_drawdown

# returns: pd.Series với DatetimeIndex
returns = pd.read_csv('my_backtest.csv', parse_dates=['date'], index_col='date')['return']

print(returns_summary(returns))
print(f"Sharpe: {sharpe_ratio(returns):.2f}")
print(f"Max DD: {max_drawdown(returns)['max_drawdown']*100:.1f}%")
```

### Generate tear sheet

```python
from tear_sheet import create_tear_sheet

create_tear_sheet(
    returns=returns,
    benchmark_returns=benchmark,
    strategy_name='My Strategy',
    instrument='XAUUSD',
    output_path='my_tear_sheet.pdf',
)
```

### Portfolio analysis

```python
from portfolio_metrics import (
    strategy_correlation_matrix, risk_parity_weights,
    portfolio_diversification_test,
)

returns_dict = {'xau': xau_ret, 'eur': eur_ret, 'btc': btc_ret}

corr = strategy_correlation_matrix(returns_dict)
print(corr)

rp_weights = risk_parity_weights(returns_dict)
result = portfolio_diversification_test(returns_dict, weights=rp_weights)
print(f"Portfolio Sharpe: {result['portfolio_sharpe']:.2f}")
print(f"Diversification lift: {result['diversification_lift']:+.1f}%")
```

### Detect pitfalls

```python
from pitfall_detection import full_audit_report

report = full_audit_report(
    returns,
    metrics_claimed={'cagr': 0.45, 'sharpe': 4.2, 'max_dd': -0.05},
    period_claimed=('2022-01-01', '2024-06-30'),
)
print(report)
```

## Mapping với book

| Section sách | File code |
|--------------|-----------|
| 6.2 Returns metrics | `metrics.py` (returns_summary, cagr, geometric_mean) |
| 6.3 Risk metrics | `metrics.py` (max_drawdown, ulcer_index, pain_index, lake_ratio) |
| 6.4 Risk-adjusted | `metrics.py` (sharpe, sortino, calmar, mar, omega, burke, sterling, martin) |
| 6.5 Trade-level | `metrics.py` (win_rate, profit_factor, expectancy, kelly, MAE/MFE) |
| 6.6 Tail risk | `metrics.py` (var, cvar, tail_ratio, distribution_moments) |
| 6.7 Stability | `metrics.py` (yearly_breakdown, rolling_sharpe, edge_significance, stability_index) |
| 6.8 Benchmark | `metrics.py` (alpha_beta, capture_ratios) |
| 6.9 Portfolio | `portfolio_metrics.py` |
| 6.10 Tear sheet | `tear_sheet.py` |
| 6.11 Case studies | `solution_exercise_4.py` |
| 6.12 Pitfalls | `pitfall_detection.py` |
| 6.13 Live monitoring | `live_monitor.py` |

## Discord & support

- **Discord**: https://discord.gg/CC6xsZ8tcf
- **Channel**: `#chapter-06-metrics` cho hỏi đáp về Chương 6
- **Bug reports**: [GitHub Issues](https://github.com/quantcfd/book-code/issues)

---

QuantCFD by Anthony Nguyễn — Monta Capital Investment Company Limited
