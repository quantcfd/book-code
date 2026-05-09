"""
QuantCFD — Chương 10.7
Volatility Targeting at Portfolio Level

Adjust position sizes to maintain constant portfolio volatility.
Adapts automatically to market regime changes.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def compute_realized_vol(
    returns: pd.Series,
    lookback_days: int = 63,
    use_ewma: bool = True,
) -> float:
    """
    Annualized realized vol estimate.

    Args:
        returns: Return series.
        lookback_days: Rolling window.
        use_ewma: Exponential weighting.

    Returns:
        Annualized vol (e.g. 0.15 = 15%).
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


def vol_targeted_leverage(
    realized_vol_annual: float,
    target_vol_annual: float = 0.10,
    max_leverage: float = 3.0,
    min_leverage: float = 0.1,
) -> float:
    """
    Compute leverage to achieve target portfolio vol.

    Args:
        realized_vol_annual: Current portfolio realized vol.
        target_vol_annual: Target vol (default 10%).
        max/min_leverage: Constraints.

    Returns:
        Leverage multiplier.
    """
    if realized_vol_annual is None or realized_vol_annual <= 0:
        return min_leverage
    leverage = target_vol_annual / realized_vol_annual
    return max(min_leverage, min(leverage, max_leverage))


def apply_vol_targeting(
    base_returns: pd.Series,
    target_vol_annual: float = 0.10,
    lookback_days: int = 63,
    rebalance_freq: int = 21,
    max_leverage: float = 3.0,
    min_leverage: float = 0.1,
) -> dict:
    """
    Apply vol targeting overlay to a return series.

    Args:
        base_returns: Underlying strategy returns.
        target_vol_annual: Target portfolio vol.
        lookback_days: Vol estimation window.
        rebalance_freq: Days between leverage updates.

    Returns:
        Dict with vol-targeted returns and leverage history.
    """
    leverage_history = pd.Series(1.0, index=base_returns.index)

    for i in range(lookback_days, len(base_returns), rebalance_freq):
        window = base_returns.iloc[max(0, i - lookback_days):i]
        realized_vol = compute_realized_vol(window, lookback_days=lookback_days)
        leverage = vol_targeted_leverage(
            realized_vol, target_vol_annual,
            max_leverage=max_leverage, min_leverage=min_leverage,
        )
        leverage_history.iloc[i:min(i + rebalance_freq, len(base_returns))] = leverage

    targeted_returns = base_returns * leverage_history.shift(1).fillna(1.0)

    # Compute statistics
    base_clean = base_returns.dropna()
    target_clean = targeted_returns.dropna()

    base_realized_vol = base_clean.std() * np.sqrt(252) if base_clean.std() > 0 else 0
    target_realized_vol = target_clean.std() * np.sqrt(252) if target_clean.std() > 0 else 0

    base_sharpe = (
        (base_clean.mean() / base_clean.std()) * np.sqrt(252)
        if base_clean.std() > 0 else 0
    )
    target_sharpe = (
        (target_clean.mean() / target_clean.std()) * np.sqrt(252)
        if target_clean.std() > 0 else 0
    )

    base_eq = (1 + base_clean).cumprod()
    target_eq = (1 + target_clean).cumprod()
    base_dd = (base_eq / base_eq.cummax() - 1).min()
    target_dd = (target_eq / target_eq.cummax() - 1).min()

    return {
        "base_returns": base_returns,
        "targeted_returns": targeted_returns,
        "leverage_history": leverage_history,
        "base_realized_vol": base_realized_vol,
        "target_realized_vol": target_realized_vol,
        "base_sharpe": base_sharpe,
        "target_sharpe": target_sharpe,
        "base_max_dd": base_dd,
        "target_max_dd": target_dd,
        "avg_leverage": leverage_history.mean(),
        "min_leverage_used": leverage_history.min(),
        "max_leverage_used": leverage_history.max(),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Vol Targeting — Demo")
    print("=" * 70)

    np.random.seed(42)
    n = 1500  # 6 năm daily
    dates = pd.date_range("2018-01-01", periods=n, freq="D")

    # Synthetic returns with vol regime changes
    rets = np.zeros(n)
    for i in range(0, n, 200):
        end = min(i + 200, n)
        regime = np.random.choice(["calm", "normal", "vol", "crash"])
        if regime == "calm":
            rets[i:end] = np.random.randn(end - i) * 0.005
        elif regime == "normal":
            rets[i:end] = np.random.randn(end - i) * 0.012
        elif regime == "vol":
            rets[i:end] = np.random.randn(end - i) * 0.025
        else:  # crash
            rets[i:end] = np.random.randn(end - i) * 0.04 - 0.005

    base_returns = pd.Series(rets, index=dates)

    # Apply vol targeting
    print(f"\nApplying 10% vol target...")
    result = apply_vol_targeting(
        base_returns,
        target_vol_annual=0.10,
        lookback_days=63,
        rebalance_freq=21,
    )

    print(f"\n{'─' * 70}")
    print(f"{'Metric':<28} {'Base':>15} {'Vol-Targeted':>15}")
    print(f"{'─' * 70}")
    print(f"{'Realized vol':<28} {result['base_realized_vol']*100:>14.2f}% {result['target_realized_vol']*100:>14.2f}%")
    print(f"{'Sharpe ratio':<28} {result['base_sharpe']:>15.3f} {result['target_sharpe']:>15.3f}")
    print(f"{'Max DD':<28} {result['base_max_dd']*100:>14.2f}% {result['target_max_dd']*100:>14.2f}%")

    print(f"\nLeverage statistics:")
    print(f"  Average:  {result['avg_leverage']:.3f}x")
    print(f"  Min:      {result['min_leverage_used']:.3f}x  (deleveraged in vol regime)")
    print(f"  Max:      {result['max_leverage_used']:.3f}x  (leveraged in calm regime)")

    print(f"\nKey insight:")
    print(f"  - Base portfolio: vol varies wildly, DD large")
    print(f"  - Vol-targeted: vol stable, DD reduced")
    print(f"  - Sharpe similar (vol target doesn't add edge, just smooths)")
