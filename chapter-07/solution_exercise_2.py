"""
QuantCFD — Chương 7, Bài tập 2
Donchian implementation (60 phút)

Yêu cầu:
- Implement Donchian breakout system 1 (20/10) và system 2 (55/20)
- Test trên BTCUSD daily 2018-2024
- Compare results
"""

import numpy as np
import pandas as pd


def donchian_solution(df: pd.DataFrame, entry_period: int = 20,
                      exit_period: int = 10, cost: float = 0.0010) -> dict:
    """
    Donchian breakout implementation.
    
    Entry: long khi giá phá highest high của entry_period
    Exit: short signal hoặc giá phá lowest low của exit_period
    """
    out = df.copy()
    
    # Compute channels (shifted to avoid look-ahead)
    out["high_band"] = out["high"].rolling(entry_period).max().shift(1)
    out["low_band"] = out["low"].rolling(exit_period).min().shift(1)
    
    # State machine for position tracking
    position = 0
    positions = []
    
    for i in range(len(out)):
        row = out.iloc[i]
        if pd.isna(row["high_band"]) or pd.isna(row["low_band"]):
            positions.append(0)
            continue
        
        if position == 0:
            # Check entry: today's high breaks above yesterday's high_band
            if row["high"] > row["high_band"]:
                position = 1
        else:
            # Check exit: today's low breaks below low_band
            if row["low"] < row["low_band"]:
                position = 0
        positions.append(position)
    
    out["position"] = positions
    out["asset_return"] = out["close"].pct_change()
    out["strat_return"] = out["position"] * out["asset_return"]
    out["pos_change"] = out["position"].diff().abs().fillna(0)
    out["strat_return_net"] = out["strat_return"] - out["pos_change"] * cost
    
    out_clean = out.dropna()
    if len(out_clean) < 30 or out_clean["strat_return_net"].std() == 0:
        return {"error": "insufficient data"}
    
    sharpe = (out_clean["strat_return_net"].mean() / out_clean["strat_return_net"].std()
              * np.sqrt(365))  # crypto = 365 days
    cagr = (1 + out_clean["strat_return_net"].mean()) ** 365 - 1
    equity = (1 + out_clean["strat_return_net"]).cumprod()
    max_dd = (equity / equity.cummax() - 1).min()
    total_trades = int(out_clean["pos_change"].sum() / 2)
    
    # Win rate
    out_clean = out_clean.copy()
    out_clean["trade_id"] = (out_clean["pos_change"].cumsum() / 2).astype(int)
    trade_pnl = out_clean.groupby("trade_id")["strat_return_net"].sum()
    win_rate = (trade_pnl > 0).mean() if len(trade_pnl) > 0 else 0
    
    return {
        "sharpe": sharpe, "cagr": cagr, "max_dd": max_dd,
        "total_trades": total_trades, "win_rate": win_rate,
        "equity_curve": equity,
    }


if __name__ == "__main__":
    # Synthetic BTC-like data
    np.random.seed(42)
    dates = pd.date_range("2018-01-01", "2024-12-31", freq="D")
    
    # Bull-bear cycle
    n = len(dates)
    returns = np.zeros(n)
    returns[:n//4] = np.random.randn(n//4) * 0.030 + 0.001        # bull
    returns[n//4:n//2] = np.random.randn(n//4) * 0.040 - 0.0005   # crash
    returns[n//2:3*n//4] = np.random.randn(n//4) * 0.025 + 0.0001 # winter
    returns[3*n//4:] = np.random.randn(n - 3*n//4) * 0.025 + 0.0008  # recovery
    
    prices = 10000 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({
        "close": prices,
        "high": prices * (1 + np.abs(np.random.randn(n)) * 0.015),
        "low": prices * (1 - np.abs(np.random.randn(n)) * 0.015),
    }, index=dates)
    
    print("=" * 60)
    print("Bài tập 2 — Donchian Breakout (BTCUSD synthetic)")
    print("=" * 60)
    
    print("\nSystem 1 (20/10):")
    s1 = donchian_solution(df, 20, 10)
    print(f"  Sharpe:        {s1['sharpe']:.3f}")
    print(f"  CAGR:          {s1['cagr']*100:.2f}%")
    print(f"  Max DD:        {s1['max_dd']*100:.2f}%")
    print(f"  Trades:        {s1['total_trades']}")
    print(f"  Win rate:      {s1['win_rate']*100:.1f}%")
    
    print("\nSystem 2 (55/20):")
    s2 = donchian_solution(df, 55, 20)
    print(f"  Sharpe:        {s2['sharpe']:.3f}")
    print(f"  CAGR:          {s2['cagr']*100:.2f}%")
    print(f"  Max DD:        {s2['max_dd']*100:.2f}%")
    print(f"  Trades:        {s2['total_trades']}")
    print(f"  Win rate:      {s2['win_rate']*100:.1f}%")
    
    print("\nLessons:")
    print("  - System 1 nhiều trades hơn nhưng win rate similar")
    print("  - System 2 ít trades, profit factor cao hơn (catch big trends)")
    print("  - Combine 50/50 thường tốt hơn từng system riêng")
