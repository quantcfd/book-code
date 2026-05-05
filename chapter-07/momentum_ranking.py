"""
QuantCFD — Chương 7
Cross-sectional Momentum Ranking

Reference: Ch7.6.
Rank N instruments by past return, long top, short bottom.
Common implementation cho FX, indices, crypto baskets.
"""

import numpy as np
import pandas as pd


def lookback_return(prices: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """
    Compute lookback returns cho all assets.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Wide format: rows = dates, columns = symbols.
    lookback : int
        Number of bars to look back.
    
    Returns
    -------
    pd.DataFrame
        Returns over lookback period (shifted by 1 to avoid look-ahead).
    """
    return prices.pct_change(lookback).shift(1)


def momentum_ranks(prices: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """
    Rank assets by momentum (1 = strongest, N = weakest).
    """
    rets = lookback_return(prices, lookback)
    # Higher return = lower rank number (1 is best)
    ranks = rets.rank(axis=1, ascending=False, method="min")
    return ranks


def momentum_portfolio_signals(prices: pd.DataFrame, lookback: int = 60,
                                top_n: int = 2, bottom_n: int = 2) -> pd.DataFrame:
    """
    Generate long/short signals based on momentum ranking.
    
    Long top_n assets, short bottom_n assets.
    
    Returns DataFrame same shape as prices: +1 (long), -1 (short), 0 (flat).
    """
    ranks = momentum_ranks(prices, lookback)
    n_assets = prices.shape[1]
    
    long_signals = (ranks <= top_n).astype(int)
    short_signals = -(ranks >= n_assets - bottom_n + 1).astype(int)
    
    return long_signals + short_signals


def backtest_momentum(prices: pd.DataFrame, lookback: int = 60,
                      top_n: int = 2, bottom_n: int = 2,
                      cost_per_trade: float = 0.0010,
                      rebalance_period: int = 5) -> dict:
    """
    Backtest cross-sectional momentum portfolio.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Wide format prices.
    rebalance_period : int
        Rebalance every N bars (default 5 = weekly for daily data).
    
    Returns
    -------
    dict with sharpe, cagr, max_dd, equity_curve.
    """
    signals = momentum_portfolio_signals(prices, lookback, top_n, bottom_n)
    
    # Only rebalance every N bars
    rebalance_mask = pd.Series(False, index=signals.index)
    rebalance_mask.iloc[::rebalance_period] = True
    
    signals_held = signals.copy()
    last_signals = pd.Series(0, index=signals.columns)
    held_signals_list = []
    for i in range(len(signals)):
        if rebalance_mask.iloc[i]:
            last_signals = signals.iloc[i].copy()
        held_signals_list.append(last_signals.copy())
    signals_held = pd.DataFrame(held_signals_list, index=signals.index)
    
    # Asset returns
    asset_returns = prices.pct_change()
    
    # Position size: equal weight per active position
    n_active = signals_held.abs().sum(axis=1).replace(0, np.nan)
    weights = signals_held.div(n_active, axis=0).fillna(0)
    
    # Strategy returns (gross)
    strat_returns = (weights * asset_returns).sum(axis=1)
    
    # Transaction cost when weights change
    weight_changes = weights.diff().abs().sum(axis=1)
    tc = weight_changes * cost_per_trade
    strat_returns_net = strat_returns - tc
    
    strat_returns_net = strat_returns_net.dropna()
    
    if len(strat_returns_net) < 30:
        return {"sharpe": np.nan}
    
    sharpe = (strat_returns_net.mean() / strat_returns_net.std()
              * np.sqrt(252)) if strat_returns_net.std() > 0 else 0
    cagr = (1 + strat_returns_net.mean()) ** 252 - 1
    equity = (1 + strat_returns_net).cumprod()
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
    dates = pd.date_range("2018-01-01", "2024-12-31", freq="D")
    n_assets = 6
    asset_names = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "US500"]
    
    # Synthetic correlated returns with different drifts
    drifts = [0.0005, 0.0001, 0.0002, -0.0001, 0.0010, 0.0006]
    vols = [0.012, 0.007, 0.009, 0.006, 0.040, 0.012]
    
    prices_data = {}
    for i, name in enumerate(asset_names):
        rets = np.random.randn(len(dates)) * vols[i] + drifts[i]
        prices_data[name] = 100 * np.exp(np.cumsum(rets))
    
    prices = pd.DataFrame(prices_data, index=dates)
    
    print("=" * 60)
    print("Cross-sectional Momentum Backtest")
    print("=" * 60)
    print(f"Universe: {asset_names}")
    print(f"Lookback: 60 days, rebalance weekly")
    
    result = backtest_momentum(prices, lookback=60, top_n=2, bottom_n=2,
                                rebalance_period=5)
    
    print(f"\nSharpe:    {result['sharpe']:.3f}")
    print(f"CAGR:      {result['cagr']*100:.2f}%")
    print(f"Max DD:    {result['max_dd']*100:.2f}%")
    
    # Different lookback periods
    print("\n" + "-" * 60)
    print("Sensitivity to lookback period:")
    for lb in [20, 60, 120, 180, 252]:
        r = backtest_momentum(prices, lookback=lb, top_n=2, bottom_n=2)
        print(f"  Lookback {lb:3d} days: Sharpe = {r['sharpe']:.3f}, "
              f"CAGR = {r['cagr']*100:.1f}%, DD = {r['max_dd']*100:.1f}%")
