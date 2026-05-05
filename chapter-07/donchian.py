"""
QuantCFD — Chương 7
Donchian Channel Breakout (Turtle System 1 + 2)

Reference: Ch7.4. Donchian breakout là chiến lược trend cổ điển.
- System 1: 20-day breakout entry, 10-day breakout exit
- System 2: 55-day breakout entry, 20-day breakout exit
"""

import numpy as np
import pandas as pd


def donchian_channels(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Compute Donchian channels (highest high / lowest low over period).
    
    Returns df with added columns:
        donchian_high, donchian_low, donchian_mid
    SHIFTED by 1 bar (anti look-ahead).
    """
    out = df.copy()
    out["donchian_high"] = out["high"].rolling(period).max().shift(1)
    out["donchian_low"] = out["low"].rolling(period).min().shift(1)
    out["donchian_mid"] = (out["donchian_high"] + out["donchian_low"]) / 2
    return out


def turtle_system_1(df: pd.DataFrame, entry_period: int = 20,
                    exit_period: int = 10) -> pd.DataFrame:
    """
    Turtle System 1 — short-term breakout strategy.
    
    Entry: long when price > 20-day high (use H/L for breakout, not close)
    Exit: when price < 10-day low
    
    Returns df with 'position' column (1 = long, 0 = flat).
    """
    out = df.copy()
    
    out["high_entry"] = out["high"].rolling(entry_period).max().shift(1)
    out["low_exit"] = out["low"].rolling(exit_period).min().shift(1)
    
    position = 0
    positions = []
    
    for i in range(len(out)):
        row = out.iloc[i]
        if position == 0:
            # Check entry
            if not pd.isna(row["high_entry"]) and row["high"] > row["high_entry"]:
                position = 1
        else:
            # Check exit
            if not pd.isna(row["low_exit"]) and row["low"] < row["low_exit"]:
                position = 0
        positions.append(position)
    
    out["position"] = positions
    return out


def turtle_system_2(df: pd.DataFrame, entry_period: int = 55,
                    exit_period: int = 20) -> pd.DataFrame:
    """
    Turtle System 2 — long-term breakout strategy.
    Entry: 55-day breakout. Exit: 20-day low.
    """
    return turtle_system_1(df, entry_period, exit_period)


def backtest_turtle(df: pd.DataFrame, system: str = "S1",
                    cost_per_trade: float = 0.0005) -> dict:
    """
    Backtest Turtle system với cost.
    
    Parameters
    ----------
    system : str
        'S1' (20/10) or 'S2' (55/20).
    cost_per_trade : float
        Round-trip cost as fraction.
    
    Returns dict with sharpe, cagr, max_dd, total_trades, win_rate.
    """
    if system == "S1":
        result_df = turtle_system_1(df, entry_period=20, exit_period=10)
    elif system == "S2":
        result_df = turtle_system_2(df, entry_period=55, exit_period=20)
    else:
        raise ValueError(f"Unknown system: {system}")
    
    result_df["asset_return"] = result_df["close"].pct_change()
    result_df["strat_return"] = result_df["position"].shift(0) * result_df["asset_return"]
    
    # Transaction cost
    result_df["pos_change"] = result_df["position"].diff().abs().fillna(0)
    result_df["strat_return_net"] = (
        result_df["strat_return"] - result_df["pos_change"] * cost_per_trade
    )
    
    df_clean = result_df.dropna()
    
    if len(df_clean) < 30:
        return {"sharpe": np.nan, "total_trades": 0}
    
    bars_per_year = _bars_per_year(df_clean.index)
    
    sharpe = 0
    if df_clean["strat_return_net"].std() > 0:
        sharpe = (df_clean["strat_return_net"].mean() / df_clean["strat_return_net"].std()
                  * np.sqrt(bars_per_year))
    
    cagr = (1 + df_clean["strat_return_net"].mean()) ** bars_per_year - 1
    
    equity = (1 + df_clean["strat_return_net"]).cumprod()
    max_dd = (equity / equity.cummax() - 1).min()
    
    total_trades = int(df_clean["pos_change"].sum() / 2)
    
    df_clean["trade_id"] = (df_clean["pos_change"].cumsum() / 2).astype(int)
    trade_pnl = df_clean.groupby("trade_id")["strat_return_net"].sum()
    win_rate = (trade_pnl > 0).mean() if len(trade_pnl) > 0 else 0
    
    avg_win = trade_pnl[trade_pnl > 0].mean() if (trade_pnl > 0).any() else 0
    avg_loss = trade_pnl[trade_pnl < 0].mean() if (trade_pnl < 0).any() else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    
    return {
        "sharpe": sharpe,
        "cagr": cagr,
        "max_dd": max_dd,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "win_loss_ratio": win_loss_ratio,
        "equity_curve": equity,
    }


def combined_turtle(df: pd.DataFrame,
                    system_weights: dict = None) -> pd.DataFrame:
    """
    Combine S1 (50%) + S2 (50%) như Turtle original.
    
    Returns df with 'combined_position' (0, 0.5, or 1.0).
    """
    if system_weights is None:
        system_weights = {"S1": 0.5, "S2": 0.5}
    
    s1 = turtle_system_1(df, 20, 10)
    s2 = turtle_system_2(df, 55, 20)
    
    out = df.copy()
    out["s1_position"] = s1["position"]
    out["s2_position"] = s2["position"]
    out["combined_position"] = (
        out["s1_position"] * system_weights["S1"] +
        out["s2_position"] * system_weights["S2"]
    )
    return out


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


if __name__ == "__main__":
    # Synthetic data demo
    np.random.seed(42)
    dates = pd.date_range("2018-01-01", "2024-12-31", freq="D")
    
    # Random walk with trend
    returns = np.random.randn(len(dates)) * 0.012 + 0.0003
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLC from close
    df = pd.DataFrame({
        "close": prices,
        "high": prices * (1 + np.abs(np.random.randn(len(dates))) * 0.005),
        "low": prices * (1 - np.abs(np.random.randn(len(dates))) * 0.005),
        "open": np.roll(prices, 1),
    }, index=dates)
    df.loc[df.index[0], "open"] = prices[0]
    
    print("=" * 60)
    print("Turtle Donchian Backtest — Synthetic Daily")
    print("=" * 60)
    
    for system in ["S1", "S2"]:
        result = backtest_turtle(df, system=system)
        print(f"\nSystem {system}:")
        print(f"  Sharpe:           {result['sharpe']:.3f}")
        print(f"  CAGR:             {result['cagr']*100:.2f}%")
        print(f"  Max DD:           {result['max_dd']*100:.2f}%")
        print(f"  Trades:           {result['total_trades']}")
        print(f"  Win rate:         {result['win_rate']*100:.1f}%")
        print(f"  Win/Loss ratio:   {result['win_loss_ratio']:.2f}")
