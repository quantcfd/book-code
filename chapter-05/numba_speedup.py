"""
QuantCFD Chương 5 — Numba JIT speedup cho backtest engine

Vectorized pandas/numpy thường đủ nhanh cho daily/H4 data.
Khi dataset > 1M bars (M1 multi-year), Python loops trong stop-loss / trailing logic
trở thành bottleneck (10-30x slower than nó cần).

Numba JIT compile Python loop → near-C speed mà không phải viết Cython.

Use case Numba shine:
    - Sequential state machine (stop loss, trailing stop, partial exits)
    - Tick-level position tracking
    - Monte Carlo with many iterations
    - Custom indicators không có numpy equivalent

Use case Numba KHÔNG shine:
    - Code đã vectorized (numpy operations)
    - Pandas operations
    - I/O bound code

Run:
    pip install numba
    python numba_speedup.py
"""
import time
import numpy as np
from numba import njit


# ============================================================
# Pure Python: sequential trailing stop logic
# ============================================================
def trailing_stop_python(prices: np.ndarray, atr: np.ndarray, k: float = 3.0) -> np.ndarray:
    """
    Trailing stop k × ATR. Long-only.
    Return: array equity multiplier (1.0 = even).
    """
    n = len(prices)
    in_position = False
    entry_price = 0.0
    trailing_high = 0.0
    equity = np.ones(n)

    for i in range(1, n):
        equity[i] = equity[i-1]
        if not in_position:
            # Simple entry: sau 5 bars
            if i > 5 and prices[i] > prices[i-5]:
                in_position = True
                entry_price = prices[i]
                trailing_high = prices[i]
        else:
            if prices[i] > trailing_high:
                trailing_high = prices[i]
            stop_level = trailing_high - k * atr[i]
            if prices[i] < stop_level:
                # Exit
                pnl = (prices[i] - entry_price) / entry_price
                equity[i] = equity[i-1] * (1 + pnl)
                in_position = False

    return equity


# ============================================================
# Numba: cùng logic, JIT compiled
# ============================================================
@njit(cache=True)
def trailing_stop_numba(prices: np.ndarray, atr: np.ndarray, k: float = 3.0) -> np.ndarray:
    n = len(prices)
    in_position = False
    entry_price = 0.0
    trailing_high = 0.0
    equity = np.ones(n)

    for i in range(1, n):
        equity[i] = equity[i-1]
        if not in_position:
            if i > 5 and prices[i] > prices[i-5]:
                in_position = True
                entry_price = prices[i]
                trailing_high = prices[i]
        else:
            if prices[i] > trailing_high:
                trailing_high = prices[i]
            stop_level = trailing_high - k * atr[i]
            if prices[i] < stop_level:
                pnl = (prices[i] - entry_price) / entry_price
                equity[i] = equity[i-1] * (1 + pnl)
                in_position = False

    return equity


def benchmark():
    """So sánh Python vs Numba trên 1M bars."""
    np.random.seed(0)
    n = 1_000_000
    prices = 100 + np.cumsum(np.random.normal(0, 0.05, n))
    atr = np.full(n, 1.0)

    # Warm up Numba JIT
    _ = trailing_stop_numba(prices[:100], atr[:100])

    print(f"Backtest {n:,} bars\n")

    # Python
    t0 = time.time()
    eq_py = trailing_stop_python(prices, atr)
    t_py = time.time() - t0
    print(f"Pure Python:  {t_py:.2f}s  → final equity {eq_py[-1]:.4f}")

    # Numba
    t0 = time.time()
    eq_nb = trailing_stop_numba(prices, atr)
    t_nb = time.time() - t0
    print(f"Numba JIT:    {t_nb:.4f}s → final equity {eq_nb[-1]:.4f}")

    print(f"\nSpeedup:      {t_py/t_nb:.0f}×")
    print(f"Match check:  {np.allclose(eq_py, eq_nb)}")


# ============================================================
# Parallel walk-forward
# ============================================================
def parallel_walk_forward_demo():
    """Sử dụng joblib để parallel walk-forward windows."""
    from joblib import Parallel, delayed

    def evaluate_window(window_idx: int, fast: int, slow: int) -> dict:
        # Synthetic: simulate 1 evaluation
        time.sleep(0.05)
        return {
            'window': window_idx,
            'fast': fast,
            'slow': slow,
            'sharpe': np.random.normal(1.0, 0.5),
        }

    n_windows = 10
    param_grid = [
        {'fast': 10, 'slow': 30},
        {'fast': 20, 'slow': 50},
        {'fast': 50, 'slow': 200},
    ]

    tasks = [(w, p['fast'], p['slow'])
             for w in range(n_windows) for p in param_grid]

    # Sequential
    t0 = time.time()
    results_seq = [evaluate_window(*t) for t in tasks]
    t_seq = time.time() - t0
    print(f"\nWalk-forward {len(tasks)} tasks:")
    print(f"Sequential: {t_seq:.2f}s")

    # Parallel với 4 workers
    t0 = time.time()
    results_par = Parallel(n_jobs=4)(
        delayed(evaluate_window)(*t) for t in tasks
    )
    t_par = time.time() - t0
    print(f"Parallel (4 workers): {t_par:.2f}s")
    print(f"Speedup: {t_seq/t_par:.1f}×")


if __name__ == '__main__':
    benchmark()
    print("\n" + "="*60)
    parallel_walk_forward_demo()
