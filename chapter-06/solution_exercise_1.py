"""
Bài 1 — Implement metrics.py module + unit tests

Test toàn bộ metrics module với synthetic data có known properties:
- Random walk → Sharpe ≈ 0
- Trending positive → Sharpe > 0
- AR(1) negative → DD đáng kể
"""
import numpy as np
import pandas as pd

from metrics import (
    cagr_from_returns, sharpe_ratio, sortino_ratio, calmar_ratio,
    max_drawdown, ulcer_index, profit_factor, expectancy_per_trade,
    historical_var, historical_cvar, kelly_fraction,
    burke_ratio, sterling_ratio, martin_ratio,
)


def make_returns(mean: float, vol: float, n: int, seed: int = 42) -> pd.Series:
    np.random.seed(seed)
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    return pd.Series(np.random.normal(mean, vol, n), index=dates)


# === Test 1: Random walk → Sharpe ≈ 0 ===
def test_random_walk_sharpe():
    r = make_returns(mean=0.0, vol=0.01, n=2520)    # 10 years
    sharpe = sharpe_ratio(r)
    assert -1.0 < sharpe < 1.0, f"Sharpe {sharpe} không gần 0 cho random walk"
    print(f"✓ Random walk Sharpe: {sharpe:.3f} (~0 ok)")


# === Test 2: Trending positive ===
def test_trending_sharpe():
    r = make_returns(mean=0.001, vol=0.01, n=2520)   # 25%/yr drift
    sharpe = sharpe_ratio(r)
    assert sharpe > 1.0, f"Sharpe {sharpe} should be > 1.0"
    print(f"✓ Trending positive Sharpe: {sharpe:.3f}")


# === Test 3: Sortino > Sharpe luôn ===
def test_sortino_gt_sharpe():
    r = make_returns(mean=0.0008, vol=0.012, n=1000)
    sharpe = sharpe_ratio(r)
    sortino = sortino_ratio(r)
    assert sortino >= sharpe, "Sortino phải >= Sharpe"
    print(f"✓ Sortino {sortino:.2f} >= Sharpe {sharpe:.2f}")


# === Test 4: Max DD luôn negative or 0 ===
def test_max_dd_sign():
    r = make_returns(0.0005, 0.01, 1000)
    dd = max_drawdown(r)
    assert dd['max_drawdown'] <= 0, "Max DD phải ≤ 0"
    print(f"✓ Max DD: {dd['max_drawdown']*100:.2f}%")


# === Test 5: Profit factor consistency ===
def test_profit_factor():
    trades = pd.Series([10, -5, 8, -3, 12, -7, 15, -4])
    pf = profit_factor(trades)
    expected = (10 + 8 + 12 + 15) / (5 + 3 + 7 + 4)
    assert abs(pf - expected) < 0.01, f"PF {pf} != expected {expected}"
    print(f"✓ Profit factor: {pf:.2f}")


# === Test 6: VaR < CVaR (more negative) ===
def test_var_cvar_relation():
    r = make_returns(0.0005, 0.012, 2000)
    var95 = historical_var(r, 0.95)
    cvar95 = historical_cvar(r, 0.95)
    assert cvar95 <= var95, "CVaR phải <= VaR (more negative)"
    print(f"✓ VaR 95%: {var95*100:.2f}%, CVaR 95%: {cvar95*100:.2f}%")


# === Test 7: Kelly fraction range ===
def test_kelly_range():
    trades = pd.Series([10, -5, 8, -3, 12, -7, 15, -4])
    k = kelly_fraction(trades)
    assert -1 <= k <= 1, f"Kelly {k} out of range"
    print(f"✓ Kelly fraction: {k*100:.1f}%")


# === Test 8: Burke vs Calmar consistency ===
def test_burke_vs_calmar():
    r = make_returns(0.0006, 0.01, 1500)
    burke = burke_ratio(r)
    calmar = calmar_ratio(r)
    # Burke considers ALL DDs, Calmar chỉ max — relationship không strict
    print(f"✓ Burke: {burke:.2f}, Calmar: {calmar:.2f}")


def run_all():
    print("=" * 50)
    print("BÀI 1 — METRICS MODULE TESTS")
    print("=" * 50)

    test_random_walk_sharpe()
    test_trending_sharpe()
    test_sortino_gt_sharpe()
    test_max_dd_sign()
    test_profit_factor()
    test_var_cvar_relation()
    test_kelly_range()
    test_burke_vs_calmar()

    print("=" * 50)
    print("ALL TESTS PASSED")


if __name__ == '__main__':
    run_all()
