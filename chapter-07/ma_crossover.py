"""
QuantCFD — Chương 7
Moving Average Crossover Strategy

Reference: Ch7.3 — Strategy 1 cho trend following.
Anti look-ahead với .shift(1). Test trên XAUUSD H4 2014-2024
expected Sharpe ~1.04.
"""

import numpy as np
import pandas as pd


def ma_crossover_signals(df: pd.DataFrame, fast: int = 20, slow: int = 50,
                          ma_type: str = "sma") -> pd.DataFrame:
    """
    Generate MA crossover signals với anti-look-ahead bias.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must have 'close' column, datetime index.
    fast : int
        Fast MA period (e.g. 20).
    slow : int
        Slow MA period (e.g. 50).
    ma_type : str
        'sma' or 'ema'.
    
    Returns
    -------
    pd.DataFrame
        Original df + columns: ma_fast, ma_slow, signal.
        signal = 1 (long), 0 (flat). SHIFTED by 1 bar.
    """
    out = df.copy()
    
    if ma_type == "sma":
        out["ma_fast"] = out["close"].rolling(fast).mean()
        out["ma_slow"] = out["close"].rolling(slow).mean()
    elif ma_type == "ema":
        out["ma_fast"] = out["close"].ewm(span=fast, adjust=False).mean()
        out["ma_slow"] = out["close"].ewm(span=slow, adjust=False).mean()
    else:
        raise ValueError(f"Unknown ma_type: {ma_type}")
    
    # CRITICAL: shift signal by 1 bar (anti look-ahead)
    raw_signal = (out["ma_fast"] > out["ma_slow"]).astype(int)
    out["signal"] = raw_signal.shift(1)
    
    return out


def backtest_ma_simple(df: pd.DataFrame, fast: int = 20, slow: int = 50,
                       ma_type: str = "sma", cost_per_trade: float = 0.0005):
    """
    Simple vectorized backtest cho MA crossover.
    
    Parameters
    ----------
    cost_per_trade : float
        Round-trip cost as fraction (e.g. 0.0005 = 5 bps for liquid FX).
    
    Returns
    -------
    dict
        sharpe, cagr, max_dd, total_trades, win_rate, equity_curve.
    """
    sig_df = ma_crossover_signals(df, fast, slow, ma_type)
    sig_df["asset_return"] = sig_df["close"].pct_change()
    sig_df["strat_return"] = sig_df["signal"] * sig_df["asset_return"]
    
    # Apply transaction cost when position changes
    sig_df["pos_change"] = sig_df["signal"].diff().abs().fillna(0)
    sig_df["strat_return_net"] = sig_df["strat_return"] - sig_df["pos_change"] * cost_per_trade
    
    sig_df = sig_df.dropna()
    
    if len(sig_df) < 30:
        return {"sharpe": np.nan, "cagr": np.nan, "max_dd": np.nan,
                "total_trades": 0, "win_rate": np.nan, "equity_curve": None}
    
    # Detect periods per year (252 daily, 365*6 H4 for crypto)
    bars_per_year = _bars_per_year(sig_df.index)
    
    sharpe = (sig_df["strat_return_net"].mean() / sig_df["strat_return_net"].std()
              * np.sqrt(bars_per_year)) if sig_df["strat_return_net"].std() > 0 else 0
    
    cagr = (1 + sig_df["strat_return_net"].mean()) ** bars_per_year - 1
    
    equity = (1 + sig_df["strat_return_net"]).cumprod()
    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()
    
    total_trades = int(sig_df["pos_change"].sum() / 2)
    
    # Win rate per trade (group by signal change)
    sig_df["trade_id"] = (sig_df["pos_change"].cumsum() / 2).astype(int)
    trade_pnl = sig_df.groupby("trade_id")["strat_return_net"].sum()
    win_rate = (trade_pnl > 0).mean() if len(trade_pnl) > 0 else 0
    
    return {
        "sharpe": sharpe,
        "cagr": cagr,
        "max_dd": max_dd,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "equity_curve": equity,
    }


def _bars_per_year(dt_index):
    """Estimate bars/year from datetime index."""
    if len(dt_index) < 2:
        return 252
    median_diff = pd.Series(dt_index).diff().median()
    if pd.isna(median_diff):
        return 252
    seconds = median_diff.total_seconds()
    if seconds <= 60:
        return 365 * 24 * 60
    elif seconds <= 60 * 60:
        return 365 * 24
    elif seconds <= 4 * 60 * 60:
        return 365 * 6
    elif seconds <= 24 * 60 * 60:
        return 252
    else:
        return 52


def param_sensitivity(df: pd.DataFrame,
                      fast_grid=(10, 15, 20, 25, 30),
                      slow_grid=(40, 50, 60, 80, 100),
                      ma_type: str = "sma") -> pd.DataFrame:
    """
    Generate sensitivity matrix Sharpe ~ (fast, slow).
    Returns DataFrame fast x slow with Sharpe values.
    """
    results = pd.DataFrame(index=fast_grid, columns=slow_grid, dtype=float)
    for f in fast_grid:
        for s in slow_grid:
            if f >= s:
                results.loc[f, s] = np.nan
                continue
            r = backtest_ma_simple(df, f, s, ma_type)
            results.loc[f, s] = r["sharpe"]
    return results


def plot_equity(equity_curve, title="MA Crossover Equity"):
    """Plot equity curve nếu matplotlib available."""
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(equity_curve.index, equity_curve.values)
        ax.set_title(title)
        ax.set_ylabel("Equity (start = 1.0)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    except ImportError:
        print("matplotlib not installed; skip plot")
        return None


if __name__ == "__main__":
    # Synthetic data demo
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="4h")
    prices = 2000 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.005))
    df = pd.DataFrame({"close": prices}, index=dates)
    
    print("=" * 60)
    print("MA Crossover Backtest — Synthetic XAUUSD H4")
    print("=" * 60)
    
    result = backtest_ma_simple(df, fast=20, slow=50, ma_type="sma")
    print(f"Sharpe:       {result['sharpe']:.3f}")
    print(f"CAGR:         {result['cagr']*100:.2f}%")
    print(f"Max DD:       {result['max_dd']*100:.2f}%")
    print(f"Total trades: {result['total_trades']}")
    print(f"Win rate:     {result['win_rate']*100:.1f}%")
    
    print()
    print("Param sensitivity matrix (Sharpe):")
    sens = param_sensitivity(df)
    print(sens.round(2))
