"""
QuantCFD — Chương 7
Pyramiding — Turtle Add-on Rules

Reference: Ch7.7.5.
- Add unit khi giá move 0.5N favorable từ last unit
- Max 4 units total
- Stop loss của ALL units move lên cùng nhau
- Risk per unit ≤ 0.5%, total max 2% per trade
"""

import numpy as np
import pandas as pd
from atr_sizing import compute_atr


def trend_with_pyramiding(df: pd.DataFrame, fast: int = 20, slow: int = 50,
                          atr_period: int = 20, add_interval_n: float = 0.5,
                          max_units: int = 4, initial_stop_n: float = 2.0,
                          capital: float = 10000,
                          risk_per_unit: float = 0.005) -> pd.DataFrame:
    """
    MA crossover trend strategy với Turtle pyramiding.
    
    Parameters
    ----------
    risk_per_unit : float
        Risk per unit. Default 0.005 = 0.5% account per unit.
        With max_units=4: total max risk = 2% per trade.
    
    Returns
    -------
    pd.DataFrame
        Trade log with columns: date, units, entry_avg, exit, pnl, equity.
    """
    out = df.copy()
    out["ma_fast"] = out["close"].rolling(fast).mean()
    out["ma_slow"] = out["close"].rolling(slow).mean()
    out["atr"] = compute_atr(out, atr_period)
    out["raw_signal"] = (out["ma_fast"] > out["ma_slow"]).astype(int)
    out["signal"] = out["raw_signal"].shift(1)
    
    # State
    position_units = 0
    units = []  # list of {entry_price, size}
    stop_price = None
    last_add_price = None
    n_at_entry = None
    equity = capital
    trades = []
    
    for i in range(slow + atr_period, len(out)):
        row = out.iloc[i]
        date = row.name
        price = row["close"]
        atr = row["atr"]
        sig = row["signal"]
        
        if pd.isna(atr) or pd.isna(sig):
            continue
        
        # ENTRY (1st unit)
        if position_units == 0 and sig == 1:
            n_at_entry = atr
            unit_size = (equity * risk_per_unit) / (initial_stop_n * atr)
            units.append({"entry": price, "size": unit_size})
            position_units = 1
            last_add_price = price
            stop_price = price - initial_stop_n * atr
        
        # ADD-ON (units 2-4)
        elif (position_units > 0 and position_units < max_units
              and price >= last_add_price + add_interval_n * n_at_entry):
            unit_size = (equity * risk_per_unit) / (initial_stop_n * n_at_entry)
            units.append({"entry": price, "size": unit_size})
            position_units += 1
            last_add_price = price
            # CRITICAL: move stop của ALL units up
            stop_price = price - initial_stop_n * n_at_entry
        
        # EXIT (stop hit hoặc signal flip)
        elif position_units > 0:
            exit_now = False
            exit_price = price
            
            if price <= stop_price:
                exit_now = True
                exit_price = stop_price  # assume fill at stop
            elif sig == 0:
                exit_now = True
            
            if exit_now:
                avg_entry = sum(u["entry"] * u["size"] for u in units) / sum(u["size"] for u in units)
                pnl = sum((exit_price - u["entry"]) * u["size"] for u in units)
                equity += pnl
                trades.append({
                    "date": date,
                    "units": position_units,
                    "entry_avg": avg_entry,
                    "exit": exit_price,
                    "pnl": pnl,
                    "pnl_pct": pnl / capital,
                    "equity": equity,
                })
                position_units = 0
                units = []
                stop_price = None
                last_add_price = None
                n_at_entry = None
    
    return pd.DataFrame(trades)


def compare_pyramiding_levels(df: pd.DataFrame, capital: float = 10000) -> pd.DataFrame:
    """
    Compare strategy with 1, 2, 4, 6 units pyramiding.
    
    Returns DataFrame with comparison metrics.
    """
    results = []
    for max_u in [1, 2, 4, 6]:
        # Adjust risk_per_unit to keep total risk roughly constant
        risk_per_unit = 0.02 / max_u  # 2% total
        
        trades = trend_with_pyramiding(df, max_units=max_u,
                                       capital=capital,
                                       risk_per_unit=risk_per_unit)
        
        if len(trades) == 0:
            results.append({"max_units": max_u, "trades": 0,
                            "final_equity": capital, "max_dd": 0})
            continue
        
        equity_curve = pd.concat([
            pd.Series([capital], index=[df.index[0]]),
            trades.set_index("date")["equity"]
        ])
        
        max_dd = ((equity_curve / equity_curve.cummax()) - 1).min()
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] < 0]
        
        results.append({
            "max_units": max_u,
            "trades": len(trades),
            "final_equity": trades["equity"].iloc[-1],
            "total_return_pct": (trades["equity"].iloc[-1] / capital - 1) * 100,
            "win_rate": len(wins) / len(trades) * 100,
            "avg_win_pct": wins["pnl_pct"].mean() * 100 if len(wins) > 0 else 0,
            "avg_loss_pct": losses["pnl_pct"].mean() * 100 if len(losses) > 0 else 0,
            "max_dd_pct": max_dd * 100,
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range("2018-01-01", "2024-12-31", freq="D")
    returns = np.random.randn(len(dates)) * 0.012 + 0.0003
    prices = 2000 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({
        "close": prices,
        "high": prices * (1 + np.abs(np.random.randn(len(dates))) * 0.005),
        "low": prices * (1 - np.abs(np.random.randn(len(dates))) * 0.005),
    }, index=dates)
    
    print("=" * 60)
    print("Pyramiding Comparison — Synthetic XAUUSD")
    print("=" * 60)
    
    comparison = compare_pyramiding_levels(df, capital=10000)
    print(comparison.to_string(index=False))
    
    print("\nKey insight: pyramiding tăng avg_win, max_dd cũng tăng.")
    print("Sweet spot thường là 2-4 units cho retail.")
