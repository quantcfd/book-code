"""
QuantCFD — Chương 7
Regime Filters cho Trend Strategy

Reference: Ch7.9.
- Filter 1: ADX (trend strength)
- Filter 2: SMA200 (long-term trend filter)
- Filter 3: Volatility regime
- Filter 4: Multi-timeframe alignment
"""

import numpy as np
import pandas as pd


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute ADX (Average Directional Index).
    
    Returns df with: plus_di, minus_di, adx (all shifted by 1).
    """
    out = df.copy()
    
    # True Range
    high_low = out["high"] - out["low"]
    high_close = (out["high"] - out["close"].shift(1)).abs()
    low_close = (out["low"] - out["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = out["high"] - out["high"].shift(1)
    down_move = out["low"].shift(1) - out["low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth using Wilder's RMA
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=out.index).ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=out.index).ewm(alpha=1/period, adjust=False).mean() / atr
    
    # ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1/period, adjust=False).mean()
    
    out["plus_di"] = plus_di.shift(1)
    out["minus_di"] = minus_di.shift(1)
    out["adx"] = adx_val.shift(1)
    return out


def adx_filter(df: pd.DataFrame, period: int = 14, threshold: float = 25) -> pd.Series:
    """
    Returns Series: 1 nếu ADX > threshold (trending), 0 nếu không.
    """
    adx_df = adx(df, period)
    return (adx_df["adx"] > threshold).astype(int)


def sma_filter(df: pd.DataFrame, period: int = 200,
               direction: str = "long") -> pd.Series:
    """
    Long-term trend filter using SMA200.
    
    direction == 'long': returns 1 if close > SMA200
    direction == 'short': returns 1 if close < SMA200
    direction == 'either': returns 1 always (no filter)
    """
    sma = df["close"].rolling(period).mean().shift(1)
    
    if direction == "long":
        return (df["close"] > sma).astype(int)
    elif direction == "short":
        return (df["close"] < sma).astype(int)
    elif direction == "either":
        return pd.Series(1, index=df.index)
    else:
        raise ValueError(f"Unknown direction: {direction}")


def vol_regime_filter(df: pd.DataFrame, period: int = 30,
                      vol_min: float = 0.05, vol_max: float = 0.50) -> pd.Series:
    """
    Volatility regime filter: trade only when realized vol within range.
    
    vol_min/max in annualized terms (0.05 = 5%, 0.50 = 50%).
    """
    returns = df["close"].pct_change()
    realized_vol = returns.rolling(period).std() * np.sqrt(252)
    realized_vol_shifted = realized_vol.shift(1)
    
    in_range = ((realized_vol_shifted >= vol_min) &
                (realized_vol_shifted <= vol_max)).astype(int)
    return in_range


def multi_tf_filter(df_higher: pd.DataFrame, df_lower: pd.DataFrame,
                    fast: int = 20, slow: int = 50) -> pd.Series:
    """
    Multi-timeframe alignment filter.
    Trade lower TF only when higher TF in same direction.
    
    Parameters
    ----------
    df_higher : pd.DataFrame
        Higher timeframe data (e.g. daily for H4 strategy).
    df_lower : pd.DataFrame
        Lower timeframe data (e.g. H4).
    
    Returns
    -------
    pd.Series indexed like df_lower:
        1 if higher TF MA crossover is bullish, 0 otherwise.
    """
    higher_ma_fast = df_higher["close"].rolling(fast).mean()
    higher_ma_slow = df_higher["close"].rolling(slow).mean()
    higher_signal = (higher_ma_fast > higher_ma_slow).astype(int).shift(1)
    
    # Reindex to lower TF (forward fill)
    higher_signal_lower = higher_signal.reindex(df_lower.index, method="ffill")
    return higher_signal_lower.fillna(0).astype(int)


def combine_filters(filters: dict, mode: str = "all") -> pd.Series:
    """
    Combine multiple filters.
    
    Parameters
    ----------
    filters : dict
        Dict of {name: pd.Series} where each Series is 0/1.
    mode : str
        'all': trade only if all filters pass (AND)
        'any': trade if any filter passes (OR)
        'majority': trade if majority pass
        'weighted': filters values are weights (0..1)
    """
    if not filters:
        raise ValueError("Empty filters dict")
    
    df_filters = pd.DataFrame(filters)
    
    if mode == "all":
        return df_filters.min(axis=1)
    elif mode == "any":
        return df_filters.max(axis=1)
    elif mode == "majority":
        return (df_filters.mean(axis=1) > 0.5).astype(int)
    elif mode == "weighted":
        return df_filters.mean(axis=1)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def production_filter_stack(df: pd.DataFrame,
                             adx_threshold: float = 25,
                             sma_period: int = 200,
                             vol_min: float = 0.05,
                             vol_max: float = 0.40) -> pd.Series:
    """
    Production filter stack tôi dùng cho XAUUSD H4:
    - ADX > 25 (trend strength)
    - Close > SMA200 (long-term bullish for long-only)
    - Realized vol in (5%, 40%) range
    
    Returns combined filter (1 = pass all, 0 = fail any).
    """
    f_adx = adx_filter(df, threshold=adx_threshold)
    f_sma = sma_filter(df, period=sma_period, direction="long")
    f_vol = vol_regime_filter(df, vol_min=vol_min, vol_max=vol_max)
    
    return combine_filters({
        "adx": f_adx,
        "sma": f_sma,
        "vol": f_vol,
    }, mode="all")


if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range("2018-01-01", "2024-12-31", freq="D")
    returns = np.random.randn(len(dates)) * 0.012 + 0.0003
    prices = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({
        "close": prices,
        "high": prices * (1 + np.abs(np.random.randn(len(dates))) * 0.005),
        "low": prices * (1 - np.abs(np.random.randn(len(dates))) * 0.005),
    }, index=dates)
    
    print("=" * 60)
    print("Regime Filters Demo")
    print("=" * 60)
    
    f_adx = adx_filter(df, threshold=25)
    f_sma = sma_filter(df, period=200, direction="long")
    f_vol = vol_regime_filter(df, vol_min=0.05, vol_max=0.40)
    f_combined = production_filter_stack(df)
    
    print(f"\nADX > 25:                 {f_adx.mean()*100:.1f}% of bars pass")
    print(f"Close > SMA200:           {f_sma.mean()*100:.1f}% of bars pass")
    print(f"Vol in [5%, 40%]:         {f_vol.mean()*100:.1f}% of bars pass")
    print(f"All 3 combined (AND):     {f_combined.mean()*100:.1f}% of bars pass")
    
    print("\nFilter reduces trade frequency to ~%.0f%% of unfiltered."
          % (f_combined.mean() * 100))
