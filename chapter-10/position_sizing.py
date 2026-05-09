"""
QuantCFD — Chương 10.3
Position Sizing Methods

4 methods:
1. Fixed dollar amount
2. Fixed fractional (% equity)
3. ATR-based sizing (volatility-adjusted)
4. Volatility targeting (portfolio level)
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def fixed_dollar_size(
    risk_dollar: float,
    stop_distance: float,
    contract_value_per_point: float = 1.0,
) -> float:
    """
    Fixed dollar risk per trade. Simplest method, doesn't scale with equity.

    Args:
        risk_dollar: Fixed dollar amount risked.
        stop_distance: Distance from entry to stop (price units).
        contract_value_per_point: Dollar per 1 point per 1 lot.

    Returns:
        Position size in lots.
    """
    if stop_distance <= 0:
        return 0
    return risk_dollar / (stop_distance * contract_value_per_point)


def fixed_fractional_size(
    equity: float,
    risk_pct: float,
    stop_distance: float,
    contract_value_per_point: float = 1.0,
    min_lot: float = 0.01,
    max_lot: float = 100.0,
) -> float:
    """
    Risk fixed % of equity per trade. Scales with account.

    Args:
        equity: Current account equity.
        risk_pct: Fraction of equity at risk (0.01 = 1%).
        stop_distance: Distance to stop in price units.
        contract_value_per_point: Dollar per 1 point per 1 lot.
        min_lot/max_lot: Broker constraints.

    Returns:
        Position size in lots, rounded to 0.01.
    """
    if stop_distance <= 0:
        return 0
    risk_amount = equity * risk_pct
    raw = risk_amount / (stop_distance * contract_value_per_point)
    return round(max(min_lot, min(raw, max_lot)), 2)


def atr_sized_position(
    equity: float,
    risk_pct: float,
    atr: float,
    atr_stop_multiplier: float = 2.0,
    contract_value_per_point: float = 1.0,
    min_lot: float = 0.01,
    max_lot: float = 100.0,
) -> float:
    """
    ATR-based sizing. Stop = atr_stop_multiplier × ATR.
    Position scales with vol → constant real dollar risk.

    Args:
        equity: Current equity.
        risk_pct: Risk fraction (0.01 = 1%).
        atr: Current Average True Range value.
        atr_stop_multiplier: Stop distance in ATR multiples.
        contract_value_per_point: Dollar per point per lot.

    Returns:
        Position size in lots.
    """
    stop_distance = atr_stop_multiplier * atr
    if stop_distance <= 0:
        return 0
    risk_amount = equity * risk_pct
    raw = risk_amount / (stop_distance * contract_value_per_point)
    return round(max(min_lot, min(raw, max_lot)), 2)


def vol_targeted_size(
    equity: float,
    target_vol_annual: float,
    asset_vol_annual: float,
    asset_price: float,
    contract_value_per_point: float = 1.0,
    max_leverage: float = 3.0,
    min_leverage: float = 0.1,
) -> float:
    """
    Vol targeting: scale notional exposure to achieve target portfolio vol.

    Args:
        equity: Account equity.
        target_vol_annual: Desired portfolio vol (vd 0.10 = 10%).
        asset_vol_annual: Asset's annualized vol (vd 0.20 = 20%).
        asset_price: Current asset price.
        contract_value_per_point: Dollar per point.
        max/min_leverage: Constraints.

    Returns:
        Position size in lots.
    """
    if asset_vol_annual <= 0:
        return 0
    leverage = target_vol_annual / asset_vol_annual
    leverage = max(min_leverage, min(leverage, max_leverage))
    notional = equity * leverage
    if asset_price * contract_value_per_point <= 0:
        return 0
    return round(notional / (asset_price * contract_value_per_point), 2)


def estimate_realized_vol(
    returns: pd.Series,
    lookback_days: int = 63,
    use_ewma: bool = True,
) -> float:
    """
    Estimate annualized realized volatility from returns.

    Args:
        returns: Daily return series.
        lookback_days: Window length.
        use_ewma: If True, use EWMA weighting (more recent matters more).

    Returns:
        Annualized realized vol.
    """
    if len(returns) < 30:
        return None

    recent = returns.iloc[-lookback_days:]
    if len(recent) < 30:
        return None

    if use_ewma:
        weights = np.exp(np.linspace(-1, 0, len(recent)))
        weights /= weights.sum()
        weighted_var = np.sum(weights * (recent - recent.mean()) ** 2)
        daily_vol = np.sqrt(weighted_var)
    else:
        daily_vol = recent.std()

    return daily_vol * np.sqrt(252)


if __name__ == "__main__":
    print("=" * 70)
    print("Position Sizing — 4 methods comparison")
    print("=" * 70)

    # Sample trade scenario
    equity = 10000
    risk_pct = 0.01
    atr = 20  # XAUUSD ATR ~$20
    asset_price = 2030
    asset_vol = 0.16  # XAUUSD ~16% annualized
    contract_value = 100  # $100 per point per lot for XAUUSD

    print(f"\nScenario: $10k equity, 1% risk, XAUUSD")
    print(f"  ATR: $20, Price: $2030, Annual vol: 16%")
    print(f"  Contract value: $100/point/lot")

    print(f"\n--- Method comparison ---")

    # Fixed dollar
    fd = fixed_dollar_size(100, 50, contract_value)
    print(f"\nFixed dollar ($100 risk, 50pt stop):")
    print(f"  Size: {fd:.4f} lots")

    # Fixed fractional
    ff = fixed_fractional_size(equity, risk_pct, 50, contract_value)
    print(f"\nFixed fractional (1%, 50pt stop):")
    print(f"  Size: {ff:.4f} lots")

    # ATR sizing
    atr_size = atr_sized_position(
        equity, risk_pct, atr, atr_stop_multiplier=2.5, contract_value_per_point=contract_value,
    )
    print(f"\nATR sizing (2.5 ATR stop = $50):")
    print(f"  Size: {atr_size:.4f} lots")

    # Vol targeting
    vt = vol_targeted_size(
        equity=equity, target_vol_annual=0.10, asset_vol_annual=asset_vol,
        asset_price=asset_price, contract_value_per_point=contract_value,
    )
    print(f"\nVol targeting (10% target):")
    print(f"  Leverage: {0.10/asset_vol:.2f}x")
    print(f"  Size: {vt:.4f} lots")

    # Sensitivity to ATR (ATR-based vs others)
    print(f"\n--- ATR sensitivity (calm vs vol regime) ---")
    print(f"\nCalm market (ATR $10):")
    calm_atr = atr_sized_position(equity, risk_pct, 10, 2.5, contract_value)
    print(f"  ATR sizing: {calm_atr:.4f} lots (larger position)")
    print(f"  Fixed fractional same stop ($25): {fixed_fractional_size(equity, risk_pct, 25, contract_value):.4f}")

    print(f"\nVol market (ATR $40):")
    vol_atr = atr_sized_position(equity, risk_pct, 40, 2.5, contract_value)
    print(f"  ATR sizing: {vol_atr:.4f} lots (smaller position)")
    print(f"  Fixed fractional same stop ($100): {fixed_fractional_size(equity, risk_pct, 100, contract_value):.4f}")

    print(f"\nKey insight: ATR sizing keeps real risk constant")
    print(f"  Calm market: bigger position, same dollar risk")
    print(f"  Vol market:  smaller position, same dollar risk")
    print(f"  Fixed fractional: same position regardless of vol")
