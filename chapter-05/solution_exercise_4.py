"""
QuantCFD Chương 5 — Solution Bài tập 4: Multi-strategy portfolio

Test 3 strategies trên XAUUSD:
    - Trend: MA(20,50) crossover, weight 0.4
    - Breakout: Donchian(20), weight 0.3
    - Mean reversion: RSI(2) extreme, weight 0.3

Câu trả lời mẫu sẽ phụ thuộc vào data thực, dưới đây là logic + cách phân tích.
"""
import numpy as np
import pandas as pd
from multi_strategy import (
    run_multi_strategy_backtest,
    trend_strategy, breakout_strategy, mean_reversion_strategy,
)


def solve(csv_path: str = None):
    if csv_path:
        df = pd.read_csv(csv_path, parse_dates=['datetime'], index_col='datetime')
    else:
        # Synthetic XAUUSD-like
        np.random.seed(42)
        n = 1500
        dates = pd.date_range('2020-01-01', periods=n, freq='D')
        # Trending base với regime shifts
        trend = np.cumsum(np.random.normal(0.001, 0.015, n))
        regime_change_idx = [400, 800, 1200]
        for idx in regime_change_idx:
            trend[idx:] *= -0.7
        close = 1500 * np.exp(trend)
        df = pd.DataFrame({
            'close': close,
            'high':  close * (1 + np.abs(np.random.normal(0, 0.005, n))),
            'low':   close * (1 - np.abs(np.random.normal(0, 0.005, n))),
        }, index=dates)

    strategies = {
        'trend':    lambda d: trend_strategy(d, fast=20, slow=50),
        'breakout': lambda d: breakout_strategy(d, n=20),
        'mean_rev': lambda d: mean_reversion_strategy(d, n=2),
    }
    weights = {'trend': 0.4, 'breakout': 0.3, 'mean_rev': 0.3}

    portfolio = run_multi_strategy_backtest(df, strategies, weights)

    # Individual backtests
    individuals = {}
    for name, fn in strategies.items():
        single = run_multi_strategy_backtest(df, {name: fn}, {name: 1.0})
        individuals[name] = single

    print("=== Individual strategies ===")
    for name, r in individuals.items():
        print(f"  {name:10s}  return={r['total_return']*100:+.1f}%  "
              f"sharpe={r['portfolio_sharpe']:.2f}  DD={r['max_dd']*100:.1f}%")

    print("\n=== Portfolio (weighted combo) ===")
    print(f"  return={portfolio['total_return']*100:+.1f}%  "
          f"sharpe={portfolio['portfolio_sharpe']:.2f}  DD={portfolio['max_dd']*100:.1f}%")

    print("\n=== Correlation matrix ===")
    print(portfolio['correlation_matrix'].round(2))

    avg_indiv_sharpe = np.mean([r['portfolio_sharpe'] for r in individuals.values()])
    print(f"\n=== Diversification analysis ===")
    print(f"Avg individual Sharpe: {avg_indiv_sharpe:.2f}")
    print(f"Portfolio Sharpe:      {portfolio['portfolio_sharpe']:.2f}")
    print(f"Lift:                  {(portfolio['portfolio_sharpe']/avg_indiv_sharpe - 1)*100:+.1f}%")

    avg_pairwise = portfolio['correlation_matrix'].values[
        np.triu_indices(3, k=1)
    ].mean()
    print(f"Avg pairwise correlation: {avg_pairwise:.2f}")

    print("\n=== Verdict ===")
    if portfolio['portfolio_sharpe'] > avg_indiv_sharpe * 1.1 and avg_pairwise < 0.5:
        print("✓ Portfolio worth deploying — diversification benefits clear")
    elif avg_pairwise > 0.7:
        print("✗ Strategies too correlated — reduce overlap or replace one")
    else:
        print("~ Marginal — consider reweighting hoặc thêm strategy uncorrelated")


if __name__ == '__main__':
    import sys
    solve(sys.argv[1] if len(sys.argv) > 1 else None)
