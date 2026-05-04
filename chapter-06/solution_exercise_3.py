"""
Bài 3 — Portfolio analysis (3 strategies)

Combine 3 strategies thành portfolio. Compute correlation, MCR, risk parity, HHI.
"""
import numpy as np
import pandas as pd
from portfolio_metrics import (
    strategy_correlation_matrix, correlation_summary,
    portfolio_diversification_test, marginal_contribution_to_risk,
    risk_parity_weights, herfindahl_index,
)


def main():
    np.random.seed(42)
    n = 1500
    dates = pd.date_range('2020-01-01', periods=n, freq='D')

    strategies = {
        'xauusd_ma':       pd.Series(np.random.normal(0.0006, 0.011, n), index=dates),
        'eurusd_donchian': pd.Series(np.random.normal(0.0004, 0.008, n), index=dates),
        'btc_trend':       pd.Series(np.random.normal(0.0010, 0.025, n), index=dates),
    }

    print("=== Bài 3: Portfolio Analysis ===\n")

    print("--- Correlation Matrix ---")
    corr = strategy_correlation_matrix(strategies)
    print(corr.round(3))
    print(f"\nSummary: {correlation_summary(corr)}\n")

    print("--- Equal Weight Portfolio ---")
    eq = portfolio_diversification_test(strategies)
    print(f"  Avg individual Sharpe: {eq['average_individual']:.2f}")
    print(f"  Portfolio Sharpe:      {eq['portfolio_sharpe']:.2f}")
    print(f"  Diversification lift:  {eq['diversification_lift']:+.1f}%\n")

    print("--- Risk Parity ---")
    rp = risk_parity_weights(strategies)
    for name, w in rp.items():
        print(f"  {name:20s} {w*100:.1f}%")

    rp_test = portfolio_diversification_test(strategies, weights=rp)
    print(f"\n  RP Portfolio Sharpe: {rp_test['portfolio_sharpe']:.2f}")
    print(f"  RP Lift vs equal:    {rp_test['diversification_lift']:+.1f}%\n")

    print("--- MCR (equal weights) ---")
    eq_w = {k: 1/3 for k in strategies}
    mcr = marginal_contribution_to_risk(strategies, eq_w)
    for name, info in mcr.items():
        print(f"  {name:20s} contrib {info['pct_contribution']:.1f}%")

    print("\n--- Herfindahl ---")
    print(f"  Equal: {herfindahl_index(eq_w)}")
    print(f"  RP:    {herfindahl_index(rp)}")

    print("\n=== VERDICT ===")
    if rp_test['portfolio_sharpe'] > eq['portfolio_sharpe']:
        print(f"Risk parity cải thiện Sharpe {(rp_test['portfolio_sharpe']/eq['portfolio_sharpe']-1)*100:+.1f}%")
    else:
        print("Equal weight tốt hơn — strategies có vol balance sẵn")


if __name__ == '__main__':
    main()
