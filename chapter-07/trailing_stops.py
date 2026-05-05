"""
QuantCFD — Chương 7
Trailing Stops — 5 phương pháp

Reference: Ch7.8.
1. Fixed % trailing
2. ATR-based trailing
3. Donchian trailing (n-day low)
4. Parabolic SAR
5. Chandelier exit (= ATR variant)
"""

import numpy as np
import pandas as pd
from typing import Tuple


def fixed_pct_trail(entry_price: float, current_price: float,
                    high_water: float, trail_pct: float = 0.05) -> Tuple[float, float]:
    """
    Fixed percentage trailing stop.
    
    Stop = high_water × (1 - trail_pct) for long.
    
    Returns
    -------
    (new_high_water, stop_price)
    """
    new_hw = max(high_water, current_price)
    stop = new_hw * (1 - trail_pct)
    return new_hw, stop


def atr_trail(entry_price: float, high_water: float, atr: float,
              n: float = 3.0) -> Tuple[float, float]:
    """
    ATR-based trailing stop.
    Stop = high_water - n × ATR (for long).
    """
    stop = high_water - n * atr
    return high_water, stop


def donchian_trail(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    Donchian trailing stop = lowest low of last `period` bars.
    For long position. Use highest high for short.
    
    Returns Series of stop prices (shifted by 1 bar to avoid look-ahead).
    """
    return df["low"].rolling(period).min().shift(1)


def chandelier_exit(df: pd.DataFrame, atr_period: int = 22,
                     atr_mult: float = 3.0,
                     direction: str = "long") -> pd.Series:
    """
    Chandelier exit (Charles Le Beau).
    
    For long: stop = highest_high(period) - atr_mult × ATR
    For short: stop = lowest_low(period) + atr_mult × ATR
    
    Returns Series of stop prices.
    """
    from atr_sizing import compute_atr
    atr = compute_atr(df, period=atr_period, method="ema")
    
    if direction == "long":
        highest_high = df["high"].rolling(atr_period).max()
        stop = highest_high - atr_mult * atr
    elif direction == "short":
        lowest_low = df["low"].rolling(atr_period).min()
        stop = lowest_low + atr_mult * atr
    else:
        raise ValueError(f"Unknown direction: {direction}")
    
    return stop.shift(1)


def parabolic_sar(df: pd.DataFrame, af_start: float = 0.02,
                  af_step: float = 0.02, af_max: float = 0.20) -> pd.Series:
    """
    Parabolic SAR (Welles Wilder).
    
    Returns Series of SAR values (shifted by 1).
    """
    sar = pd.Series(index=df.index, dtype=float)
    
    if len(df) < 2:
        return sar
    
    # Initialize
    is_long = True
    extreme_point = df["high"].iloc[0]
    af = af_start
    sar.iloc[0] = df["low"].iloc[0]
    
    for i in range(1, len(df)):
        prev_sar = sar.iloc[i-1]
        
        if is_long:
            new_sar = prev_sar + af * (extreme_point - prev_sar)
            new_sar = min(new_sar, df["low"].iloc[i-1],
                          df["low"].iloc[i-2] if i >= 2 else df["low"].iloc[i-1])
            
            if df["low"].iloc[i] < new_sar:
                # Reversal
                is_long = False
                new_sar = extreme_point
                extreme_point = df["low"].iloc[i]
                af = af_start
            else:
                if df["high"].iloc[i] > extreme_point:
                    extreme_point = df["high"].iloc[i]
                    af = min(af + af_step, af_max)
        else:
            new_sar = prev_sar + af * (extreme_point - prev_sar)
            new_sar = max(new_sar, df["high"].iloc[i-1],
                          df["high"].iloc[i-2] if i >= 2 else df["high"].iloc[i-1])
            
            if df["high"].iloc[i] > new_sar:
                # Reversal
                is_long = True
                new_sar = extreme_point
                extreme_point = df["high"].iloc[i]
                af = af_start
            else:
                if df["low"].iloc[i] < extreme_point:
                    extreme_point = df["low"].iloc[i]
                    af = min(af + af_step, af_max)
        
        sar.iloc[i] = new_sar
    
    return sar.shift(1)


def backtest_with_trailing_stop(df: pd.DataFrame, entry_signal: pd.Series,
                                 method: str = "atr",
                                 method_kwargs: dict = None) -> dict:
    """
    Generic backtest with trailing stop.
    
    Parameters
    ----------
    entry_signal : pd.Series
        1 = enter long, 0 = no entry. Should be shifted by 1.
    method : str
        'fixed_pct', 'atr', 'donchian', 'chandelier'.
    method_kwargs : dict
        Method-specific params.
    """
    method_kwargs = method_kwargs or {}
    
    out = df.copy()
    out["signal"] = entry_signal
    
    position = 0
    entry_price = 0
    high_water = 0
    stop_price = 0
    pnl_list = []
    positions = []
    stops = []
    
    if method == "atr":
        from atr_sizing import compute_atr
        atrs = compute_atr(df, period=method_kwargs.get("atr_period", 14)).values
    elif method == "donchian":
        donchian_stops = donchian_trail(df, period=method_kwargs.get("period", 10)).values
    elif method == "chandelier":
        chand_stops = chandelier_exit(df,
                                       atr_period=method_kwargs.get("atr_period", 22),
                                       atr_mult=method_kwargs.get("atr_mult", 3.0)).values
    
    for i in range(len(out)):
        row = out.iloc[i]
        price = row["close"]
        
        if position == 0:
            if row["signal"] == 1:
                position = 1
                entry_price = price
                high_water = price
                
                if method == "fixed_pct":
                    high_water, stop_price = fixed_pct_trail(
                        entry_price, price, high_water,
                        trail_pct=method_kwargs.get("trail_pct", 0.05))
                elif method == "atr":
                    if not np.isnan(atrs[i]):
                        _, stop_price = atr_trail(entry_price, high_water,
                                                   atrs[i],
                                                   n=method_kwargs.get("n", 3.0))
                elif method == "donchian":
                    stop_price = donchian_stops[i] if not np.isnan(donchian_stops[i]) else 0
                elif method == "chandelier":
                    stop_price = chand_stops[i] if not np.isnan(chand_stops[i]) else 0
        else:
            high_water = max(high_water, price)
            
            if method == "fixed_pct":
                _, stop_price = fixed_pct_trail(entry_price, price, high_water,
                                                 trail_pct=method_kwargs.get("trail_pct", 0.05))
            elif method == "atr":
                if not np.isnan(atrs[i]):
                    _, stop_price = atr_trail(entry_price, high_water, atrs[i],
                                               n=method_kwargs.get("n", 3.0))
            elif method == "donchian":
                if not np.isnan(donchian_stops[i]):
                    stop_price = donchian_stops[i]
            elif method == "chandelier":
                if not np.isnan(chand_stops[i]):
                    stop_price = chand_stops[i]
            
            if price < stop_price:
                pnl = (price - entry_price) / entry_price
                pnl_list.append(pnl)
                position = 0
                entry_price = 0
                high_water = 0
                stop_price = 0
        
        positions.append(position)
        stops.append(stop_price if position == 1 else np.nan)
    
    out["position"] = positions
    out["stop_price"] = stops
    
    if len(pnl_list) == 0:
        return {"trades": 0, "avg_pnl": 0, "win_rate": 0}
    
    pnl_array = np.array(pnl_list)
    return {
        "trades": len(pnl_list),
        "avg_pnl": pnl_array.mean(),
        "win_rate": (pnl_array > 0).mean(),
        "total_return": pnl_array.sum(),
        "max_loss": pnl_array.min(),
        "max_win": pnl_array.max(),
        "profit_factor": (pnl_array[pnl_array > 0].sum() /
                          abs(pnl_array[pnl_array < 0].sum())
                          if (pnl_array < 0).any() else np.inf),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
    returns = np.random.randn(len(dates)) * 0.012 + 0.0005
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        "close": prices,
        "high": prices * (1 + np.abs(np.random.randn(len(dates))) * 0.005),
        "low": prices * (1 - np.abs(np.random.randn(len(dates))) * 0.005),
        "open": np.roll(prices, 1),
    }, index=dates)
    df.loc[df.index[0], "open"] = prices[0]
    
    # Simple MA crossover entry signal
    df["ma_fast"] = df["close"].rolling(20).mean()
    df["ma_slow"] = df["close"].rolling(50).mean()
    raw = (df["ma_fast"] > df["ma_slow"]).astype(int)
    entry_signal = ((raw == 1) & (raw.shift(1) == 0)).astype(int).shift(1).fillna(0)
    
    print("=" * 60)
    print("Trailing Stop Comparison — same entry signal")
    print("=" * 60)
    
    methods = [
        ("fixed_pct", {"trail_pct": 0.05}),
        ("atr", {"atr_period": 14, "n": 3.0}),
        ("donchian", {"period": 10}),
        ("chandelier", {"atr_period": 22, "atr_mult": 3.0}),
    ]
    
    for name, kwargs in methods:
        result = backtest_with_trailing_stop(df, entry_signal, method=name,
                                              method_kwargs=kwargs)
        print(f"\n{name.upper()} ({kwargs}):")
        print(f"  Trades:        {result['trades']}")
        print(f"  Win rate:      {result['win_rate']*100:.1f}%")
        print(f"  Avg PnL:       {result.get('avg_pnl', 0)*100:.3f}%")
        print(f"  Total return:  {result.get('total_return', 0)*100:.2f}%")
        print(f"  Profit factor: {result.get('profit_factor', 0):.2f}")
