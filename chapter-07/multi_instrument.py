"""
QuantCFD — Chương 7
Multi-instrument Trend Portfolio

Reference: Ch7.10.
Run cùng strategy trên 6 CFD instruments uncorrelated.
Combine signals + position sizing across portfolio.
"""

import numpy as np
import pandas as pd
from atr_sizing import compute_atr, vol_target_size


def run_strategy_per_instrument(prices_dict: dict, strategy_func,
                                 strategy_kwargs: dict = None) -> dict:
    """
    Run strategy trên mỗi instrument trong universe.
    
    Parameters
    ----------
    prices_dict : dict
        {symbol: pd.DataFrame with OHLC}
    strategy_func : callable
        Function that takes df, returns df with 'signal' column.
    strategy_kwargs : dict
        kwargs to pass to strategy_func.
    
    Returns
    -------
    dict {symbol: signals_df}
    """
    strategy_kwargs = strategy_kwargs or {}
    results = {}
    for symbol, df in prices_dict.items():
        results[symbol] = strategy_func(df, **strategy_kwargs)
    return results


def combine_returns(signals_dict: dict, weight_method: str = "equal") -> pd.Series:
    """
    Combine returns từ multiple instruments.
    
    Parameters
    ----------
    signals_dict : dict
        {symbol: df with 'signal' and 'close' columns}
    weight_method : str
        'equal' (1/N each), 'vol_parity' (inverse vol weight)
    """
    returns_per_asset = {}
    for symbol, df in signals_dict.items():
        rets = df["close"].pct_change()
        signal = df["signal"]
        strat_ret = signal * rets
        returns_per_asset[symbol] = strat_ret
    
    returns_df = pd.DataFrame(returns_per_asset)
    
    if weight_method == "equal":
        n = len(returns_df.columns)
        weights = pd.DataFrame(1/n, index=returns_df.index,
                              columns=returns_df.columns)
    elif weight_method == "vol_parity":
        # Inverse vol weighting (rolling)
        rolling_vol = returns_df.rolling(60).std()
        inv_vol = 1 / rolling_vol.replace(0, np.nan)
        weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0)
    else:
        raise ValueError(f"Unknown weight_method: {weight_method}")
    
    portfolio_return = (returns_df * weights).sum(axis=1)
    return portfolio_return


def correlation_matrix(returns_dict: dict, lookback: int = 252) -> pd.DataFrame:
    """
    Compute pairwise correlation matrix of strategy returns.
    """
    df = pd.DataFrame({k: v.tail(lookback) for k, v in returns_dict.items()})
    return df.corr()


def run_multi_instrument_trend(prices_dict: dict, fast: int = 20, slow: int = 50,
                                weight_method: str = "vol_parity",
                                cost: float = 0.0005) -> dict:
    """
    Full multi-instrument trend portfolio backtest.
    
    Returns dict with:
        portfolio_sharpe, portfolio_cagr, portfolio_max_dd,
        per_asset_metrics, correlation_matrix, equity_curve.
    """
    def ma_strategy(df, fast=fast, slow=slow):
        out = df.copy()
        out["ma_fast"] = out["close"].rolling(fast).mean()
        out["ma_slow"] = out["close"].rolling(slow).mean()
        out["signal"] = (out["ma_fast"] > out["ma_slow"]).astype(int).shift(1)
        return out
    
    signals_dict = run_strategy_per_instrument(prices_dict, ma_strategy)
    
    # Per-asset metrics
    per_asset = {}
    returns_dict = {}
    for symbol, df in signals_dict.items():
        rets = df["close"].pct_change()
        sig = df["signal"]
        pos_change = sig.diff().abs().fillna(0)
        strat_ret = sig * rets - pos_change * cost
        strat_ret = strat_ret.dropna()
        
        if len(strat_ret) > 30 and strat_ret.std() > 0:
            sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252)
            cagr = (1 + strat_ret.mean()) ** 252 - 1
            equity = (1 + strat_ret).cumprod()
            max_dd = (equity / equity.cummax() - 1).min()
        else:
            sharpe, cagr, max_dd = np.nan, np.nan, np.nan
        
        per_asset[symbol] = {
            "sharpe": sharpe, "cagr": cagr, "max_dd": max_dd,
            "trades": int(pos_change.sum() / 2),
        }
        returns_dict[symbol] = strat_ret
    
    portfolio_return = combine_returns(signals_dict, weight_method)
    portfolio_return = portfolio_return.dropna()
    
    # Subtract cost on portfolio level
    cost_drag_per_year = 0.001 * len(prices_dict)  # rough estimate
    
    if len(portfolio_return) > 30:
        portfolio_sharpe = (portfolio_return.mean() / portfolio_return.std()
                           * np.sqrt(252)) if portfolio_return.std() > 0 else 0
        portfolio_cagr = (1 + portfolio_return.mean()) ** 252 - 1
        portfolio_equity = (1 + portfolio_return).cumprod()
        portfolio_max_dd = (portfolio_equity / portfolio_equity.cummax() - 1).min()
    else:
        portfolio_sharpe = portfolio_cagr = portfolio_max_dd = np.nan
        portfolio_equity = None
    
    corr_matrix = correlation_matrix(returns_dict)
    
    return {
        "portfolio_sharpe": portfolio_sharpe,
        "portfolio_cagr": portfolio_cagr,
        "portfolio_max_dd": portfolio_max_dd,
        "per_asset": per_asset,
        "correlation_matrix": corr_matrix,
        "equity_curve": portfolio_equity,
    }


if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range("2018-01-01", "2024-12-31", freq="D")
    
    # Synthetic correlated multi-asset
    asset_specs = {
        "XAUUSD": {"drift": 0.0005, "vol": 0.011},
        "EURUSD": {"drift": 0.0001, "vol": 0.006},
        "GBPUSD": {"drift": 0.0002, "vol": 0.008},
        "USDJPY": {"drift": -0.0001, "vol": 0.006},
        "BTCUSD": {"drift": 0.0010, "vol": 0.040},
        "US500":  {"drift": 0.0006, "vol": 0.011},
    }
    
    prices_dict = {}
    for symbol, spec in asset_specs.items():
        rets = np.random.randn(len(dates)) * spec["vol"] + spec["drift"]
        prices = 100 * np.exp(np.cumsum(rets))
        df = pd.DataFrame({
            "close": prices,
            "high": prices * (1 + np.abs(np.random.randn(len(dates))) * 0.005),
            "low": prices * (1 - np.abs(np.random.randn(len(dates))) * 0.005),
        }, index=dates)
        prices_dict[symbol] = df
    
    print("=" * 60)
    print("Multi-Instrument Trend Portfolio")
    print("=" * 60)
    
    result = run_multi_instrument_trend(prices_dict, fast=20, slow=50,
                                         weight_method="vol_parity")
    
    print("\nPer-asset metrics:")
    for symbol, m in result["per_asset"].items():
        print(f"  {symbol:8s}: Sharpe = {m['sharpe']:.2f}, CAGR = {m['cagr']*100:.1f}%, "
              f"DD = {m['max_dd']*100:.1f}%, trades = {m['trades']}")
    
    print(f"\nPortfolio (vol-parity weighted):")
    print(f"  Sharpe:  {result['portfolio_sharpe']:.3f}")
    print(f"  CAGR:    {result['portfolio_cagr']*100:.2f}%")
    print(f"  Max DD:  {result['portfolio_max_dd']*100:.2f}%")
    
    print(f"\nPortfolio Sharpe vs avg single-asset Sharpe:")
    avg_sharpe = np.mean([m["sharpe"] for m in result["per_asset"].values()
                          if not np.isnan(m["sharpe"])])
    print(f"  Avg single Sharpe: {avg_sharpe:.3f}")
    print(f"  Portfolio Sharpe:  {result['portfolio_sharpe']:.3f}")
    print(f"  Diversification benefit: {result['portfolio_sharpe'] - avg_sharpe:.3f}")
    
    print("\nCorrelation matrix:")
    print(result["correlation_matrix"].round(2))
