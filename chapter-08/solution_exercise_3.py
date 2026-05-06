"""
QuantCFD — Chương 8, Bài tập 3
Z-score Cross-sectional MR (60 phút)

Yêu cầu:
- Universe 6 FX pairs
- Compute Z-score của each vs its own rolling mean
- Long bottom-2 (most oversold), short top-2 (most overbought)
- Rebalance weekly
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def cross_sectional_zscore(
    prices: pd.DataFrame,
    lookback: int = 20,
    n_long: int = 2,
    n_short: int = 2,
    rebalance_period: int = 5,
    cost: float = 0.0008,
) -> dict:
    """
    Cross-sectional Z-score MR.

    Each asset Z-score computed vs its own rolling history.
    Long n_long most oversold, short n_short most overbought.
    Rebalance every rebalance_period bars.
    """
    # Compute per-asset Z-scores
    rolling_mean = prices.rolling(lookback).mean()
    rolling_std = prices.rolling(lookback).std()
    zscores = ((prices - rolling_mean) / rolling_std).shift(1)

    # Rebalance dates
    rebalance_idx = pd.Series(False, index=prices.index)
    rebalance_idx.iloc[::rebalance_period] = True

    # Build target weights
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    last_weights = pd.Series(0.0, index=prices.columns)

    for i in range(len(prices)):
        if rebalance_idx.iloc[i]:
            z_row = zscores.iloc[i].dropna()
            if len(z_row) < (n_long + n_short):
                weights.iloc[i] = last_weights
                continue

            sorted_z = z_row.sort_values()
            longs = sorted_z.head(n_long).index   # most negative Z = oversold
            shorts = sorted_z.tail(n_short).index  # most positive Z = overbought

            new_weights = pd.Series(0.0, index=prices.columns)
            new_weights[longs] = 1.0 / n_long
            new_weights[shorts] = -1.0 / n_short
            last_weights = new_weights

        weights.iloc[i] = last_weights

    # Returns
    asset_returns = prices.pct_change()
    portfolio_returns = (weights * asset_returns).sum(axis=1)

    # Transaction cost when weights change
    weight_changes = weights.diff().abs().sum(axis=1)
    tc = weight_changes * cost / 2  # fraction
    portfolio_returns_net = portfolio_returns - tc

    portfolio_returns_net = portfolio_returns_net.dropna()

    if len(portfolio_returns_net) < 30 or portfolio_returns_net.std() == 0:
        return {"error": "insufficient data"}

    sharpe = (
        portfolio_returns_net.mean() / portfolio_returns_net.std()
        * np.sqrt(252)
    )
    cagr = (1 + portfolio_returns_net.mean()) ** 252 - 1
    equity = (1 + portfolio_returns_net).cumprod()
    max_dd = (equity / equity.cummax() - 1).min()

    return {
        "sharpe": sharpe,
        "cagr": cagr,
        "max_dd": max_dd,
        "equity_curve": equity,
        "weights_history": weights,
    }


if __name__ == "__main__":
    np.random.seed(42)
    n = 5 * 252
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    # 6 FX pairs synthetic
    asset_names = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "USDCAD", "NZDUSD"]
    common_factor = np.cumsum(np.random.randn(n) * 0.005)

    prices = {}
    for name in asset_names:
        idiosync = np.cumsum(np.random.randn(n) * 0.003)
        # Add MR component
        for i in range(1, n):
            idiosync[i] = 0.95 * idiosync[i-1] + np.random.randn() * 0.003
        prices[name] = np.exp(common_factor * 0.3 + idiosync)

    prices_df = pd.DataFrame(prices, index=dates)

    print("=" * 60)
    print("Bài tập 3 — Cross-sectional Z-score MR (6 FX pairs)")
    print("=" * 60)

    print(f"\nUniverse: {asset_names}")
    print(f"Period:   {dates[0].date()} → {dates[-1].date()}")
    print(f"Strategy: long 2 most oversold, short 2 most overbought, rebalance weekly")

    result = cross_sectional_zscore(
        prices_df, lookback=20, n_long=2, n_short=2, rebalance_period=5,
    )

    print(f"\nSharpe:    {result['sharpe']:.3f}")
    print(f"CAGR:      {result['cagr']*100:.2f}%")
    print(f"Max DD:    {result['max_dd']*100:.2f}%")

    # Sensitivity: lookback
    print("\n--- Sensitivity to lookback ---")
    for lb in [10, 20, 40, 60]:
        r = cross_sectional_zscore(prices_df, lookback=lb)
        print(f"  Lookback {lb:2d}: Sharpe={r['sharpe']:6.3f}  "
              f"CAGR={r['cagr']*100:6.2f}%  DD={r['max_dd']*100:6.2f}%")

    # Sensitivity: rebalance frequency
    print("\n--- Sensitivity to rebalance frequency ---")
    for rb in [1, 5, 10, 20]:
        r = cross_sectional_zscore(prices_df, rebalance_period=rb)
        print(f"  Rebalance {rb:2d} bars: Sharpe={r['sharpe']:6.3f}  "
              f"CAGR={r['cagr']*100:6.2f}%")

    print("\nLessons:")
    print("  - Universe ≥ 6 cần thiết cho cross-sectional ranking")
    print("  - Daily rebalance: high turnover → cost drag")
    print("  - Weekly rebalance thường sweet spot cho retail")
    print("  - Long-short structure: hedged against common factor")
