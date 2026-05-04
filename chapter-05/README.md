# Chapter 5 — Backtest Engine (50-page version)

Code đi kèm Chương 5 mở rộng. Engine vectorized self-built, multi-strategy support, 4 demos thực tế, Numba speedup, walk-forward + CPCV.

## Files

### Core engine
- `backtest_engine.py` — Vectorized engine với cost integration
- `cost_models.py` — Spread, commission, slippage, swap, funding cho 8 instruments
- `position_sizing.py` — Fixed fractional, vol target, Kelly
- `optimal_f.py` — Ralph Vince's optimal f
- `walk_forward.py` — Anchored & rolling walk-forward
- `combinatorial_cv.py` — Lopez de Prado's Combinatorial Purged CV
- `multi_strategy.py` — Portfolio engine combining N strategies
- `numba_speedup.py` — JIT compilation cho hot loops
- `common_bugs.py` — 5 common engine bugs với fix

### Demos
- `demo_xauusd_macross.py` — XAUUSD H4 MA cross (baseline)
- `demo_eurusd_breakout.py` — EURUSD H4 Donchian breakout
- `demo_btc_perpetual.py` — BTC perpetual với funding rate
- `demo_us500_rsi.py` — US500 RSI(2) mean reversion với swap

### Solutions
- `solution_exercise_1.py` — Cost realistic vs idealized
- `solution_exercise_2.py` — Custom slippage model
- `solution_exercise_3.py` — Walk-forward implementation
- `solution_exercise_4.py` — Multi-strategy portfolio
- `solution_exercise_5.py` — Numba profiling + speedup

## Quick start

```bash
# Demo XAUUSD baseline (cần data CSV)
python demo_xauusd_macross.py --csv data/XAUUSD_H4.csv

# Optimal f demo (no data needed)
python optimal_f.py

# Bug walkthrough (no data needed)
python common_bugs.py

# Numba speedup demo (no data needed)
python numba_speedup.py

# CPCV demo (no data needed)
python combinatorial_cv.py

# Multi-strategy demo (synthetic data)
python multi_strategy.py
```

## Dependencies

```
pip install numpy pandas scipy numba joblib matplotlib
```
