"""
QuantCFD — Chương 7
ATR-based Position Sizing + Volatility Targeting

Reference: Ch7.7. Turtle method position sizing.
- ATR (Average True Range) measures volatility
- Position size scaled inversely to volatility
- Vol targeting cho portfolio level
"""

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    """
    Compute True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    """
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr


def compute_atr(df: pd.DataFrame, period: int = 14, method: str = "ema") -> pd.Series:
    """
    Compute Average True Range.
    
    Parameters
    ----------
    method : str
        'sma' (simple), 'ema' (exponential), or 'wilder' (Wilder's smoothing).
    """
    tr = true_range(df)
    
    if method == "sma":
        atr = tr.rolling(period).mean()
    elif method == "ema":
        atr = tr.ewm(span=period, adjust=False).mean()
    elif method == "wilder":
        # Wilder's RMA: alpha = 1/period
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return atr


def turtle_unit_size(equity: float, atr: float, contract_value_per_point: float,
                     risk_pct: float = 0.01, n_stop: float = 2.0) -> float:
    """
    Turtle position sizing.
    
    1 Unit = (equity × risk_pct) / (n_stop × ATR × contract_value_per_point)
    
    Parameters
    ----------
    equity : float
        Account equity in USD.
    atr : float
        ATR in price units.
    contract_value_per_point : float
        Dollar value per 1 point/pip move per 1 unit.
        - XAUUSD: $1 per cent move per oz, 100 oz/lot → $100/$1 → $100 per $1 move per lot
        - EURUSD: $10 per pip per standard lot
    risk_pct : float
        % equity risked per unit (default 0.01 = 1%).
    n_stop : float
        Stop distance in N (ATR multiples). Default 2.0.
    
    Returns
    -------
    float
        Position size in lots.
    """
    risk_amount = equity * risk_pct
    risk_per_unit = n_stop * atr * contract_value_per_point
    if risk_per_unit <= 0:
        return 0
    units = risk_amount / risk_per_unit
    return units


def vol_target_size(returns: pd.Series, target_vol_annual: float = 0.10,
                    lookback: int = 30) -> pd.Series:
    """
    Compute leverage scaling cho vol targeting.
    
    leverage_t = target_vol / realized_vol_t
    
    Parameters
    ----------
    returns : pd.Series
        Asset returns (daily).
    target_vol_annual : float
        Target annualized vol of strategy returns (default 10%).
    lookback : int
        Rolling window cho realized vol (default 30 days).
    
    Returns
    -------
    pd.Series
        Leverage multiplier (clipped 0.1 to 3.0 cho safety).
    """
    realized_vol = returns.rolling(lookback).std() * np.sqrt(252)
    leverage = target_vol_annual / realized_vol.replace(0, np.nan)
    leverage = leverage.clip(0.1, 3.0)
    return leverage


def position_size_from_risk(equity: float, entry_price: float, stop_price: float,
                            risk_pct: float = 0.01,
                            contract_value_per_point: float = 1.0) -> float:
    """
    Generic position sizing from $ risk.
    
    Risk per trade = equity × risk_pct
    Stop distance = abs(entry - stop)
    Position size = risk / (stop_distance × contract_value)
    """
    risk_amount = equity * risk_pct
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return 0
    size = risk_amount / (stop_distance * contract_value_per_point)
    return size


def asset_class_sizing_table():
    """
    Return reference table cho position sizing across CFD asset classes.
    """
    return pd.DataFrame([
        {"instrument": "XAUUSD", "lot_size": "100 oz", "tick_value_per_lot": 1.0,
         "typical_atr_daily": 25, "risk_2pct_10k": 0.04},
        {"instrument": "EURUSD", "lot_size": "100k EUR", "tick_value_per_lot": 10.0,
         "typical_atr_daily": 0.0070, "risk_2pct_10k": 0.10},
        {"instrument": "GBPUSD", "lot_size": "100k GBP", "tick_value_per_lot": 10.0,
         "typical_atr_daily": 0.0090, "risk_2pct_10k": 0.08},
        {"instrument": "BTCUSD", "lot_size": "1 BTC", "tick_value_per_lot": 1.0,
         "typical_atr_daily": 1500, "risk_2pct_10k": 0.067},
        {"instrument": "US500", "lot_size": "$25/pt", "tick_value_per_lot": 25.0,
         "typical_atr_daily": 50, "risk_2pct_10k": 0.08},
        {"instrument": "WTI", "lot_size": "1000 bbl", "tick_value_per_lot": 10.0,
         "typical_atr_daily": 1.50, "risk_2pct_10k": 0.067},
    ])


if __name__ == "__main__":
    # Demo: Turtle unit sizing
    print("=" * 60)
    print("Turtle Unit Sizing — Worked Examples")
    print("=" * 60)
    
    examples = [
        {"name": "XAUUSD", "equity": 10000, "atr": 25, "contract": 100,
         "risk_pct": 0.01},
        {"name": "EURUSD", "equity": 10000, "atr": 0.0070, "contract": 100000,
         "risk_pct": 0.01},
        {"name": "BTCUSD", "equity": 10000, "atr": 1500, "contract": 1,
         "risk_pct": 0.005},  # 0.5% for high-vol crypto
    ]
    
    for ex in examples:
        units = turtle_unit_size(ex["equity"], ex["atr"], ex["contract"],
                                 risk_pct=ex["risk_pct"], n_stop=2.0)
        print(f"\n{ex['name']}:")
        print(f"  Equity: ${ex['equity']:,}")
        print(f"  ATR: {ex['atr']}")
        print(f"  Risk %: {ex['risk_pct']*100:.1f}%")
        print(f"  Position size: {units:.4f} lots")
    
    # Vol targeting demo
    print("\n" + "=" * 60)
    print("Vol Targeting Demo")
    print("=" * 60)
    
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    returns = pd.Series(np.random.randn(len(dates)) * 0.015, index=dates)
    
    leverage = vol_target_size(returns, target_vol_annual=0.10)
    print(f"\nMean leverage: {leverage.mean():.3f}")
    print(f"Min leverage:  {leverage.min():.3f}")
    print(f"Max leverage:  {leverage.max():.3f}")
    
    print("\n" + "=" * 60)
    print("Asset Class Sizing Reference")
    print("=" * 60)
    print(asset_class_sizing_table().to_string(index=False))
