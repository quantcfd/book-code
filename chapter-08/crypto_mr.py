"""
QuantCFD — Chương 8.10.7
Crypto Mean Reversion Strategies

Crypto-specific MR setups:
1. Funding rate mean reversion (perpetual futures)
2. Cross-coin ratio MR (BTC/ETH, ETH/SOL)
3. Basis trade (futures vs spot arbitrage)
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def funding_rate_strategy(
    funding_history: pd.Series,
    threshold_high: float = 0.0005,  # 0.05% per 8h = 22% annual
    threshold_low: float = -0.0002,  # -0.02% per 8h = -9% annual
    target_normal: float = 0.0001,
) -> pd.Series:
    """
    Mean reversion on perpetual funding rate.

    When funding rate extreme, take opposite position to capture funding.
    Position: -1 = short perp + long spot (when funding too high)
              +1 = long perp + short spot (when funding too negative)
              0  = flat (when funding normal)

    Args:
        funding_history: 8-hourly funding rates as decimals.
        threshold_high: Trigger short-perp when funding > this.
        threshold_low: Trigger long-perp when funding < this.
        target_normal: Exit when |funding| < this.

    Returns:
        Series of positions {-1, 0, +1}.
    """
    signals = pd.Series(0, index=funding_history.index, dtype=int)
    position = 0

    for i in range(len(funding_history)):
        rate = funding_history.iloc[i]
        if pd.isna(rate):
            signals.iloc[i] = position
            continue

        if position == 0:
            if rate > threshold_high:
                position = -1
            elif rate < threshold_low:
                position = 1
        elif position == -1 and rate < target_normal:
            position = 0
        elif position == 1 and rate > -target_normal:
            position = 0

        signals.iloc[i] = position

    return signals


def coin_ratio_mr(
    coin_a: pd.Series,
    coin_b: pd.Series,
    lookback: int = 60,
    z_entry: float = 1.5,
    z_exit: float = 0.5,
    z_stop: float = 3.0,
) -> pd.DataFrame:
    """
    Mean reversion on cross-coin ratio (e.g. BTC/ETH, ETH/SOL).

    When ratio Z-score extreme, bet revert.

    Args:
        coin_a: Numerator coin prices.
        coin_b: Denominator coin prices.
        lookback: Window for mean/std of ratio.
        z_entry: Entry threshold |z| > entry.
        z_exit: Exit when |z| < exit.
        z_stop: Stop loss |z| > stop.

    Returns:
        DataFrame with ratio, zscore, position, pnl.
    """
    common = coin_a.dropna().index.intersection(coin_b.dropna().index)
    a = coin_a.loc[common]
    b = coin_b.loc[common]

    ratio = a / b
    out = pd.DataFrame({"ratio": ratio})
    out["mean"] = ratio.rolling(lookback).mean().shift(1)
    out["std"] = ratio.rolling(lookback).std().shift(1)
    out["zscore"] = (ratio - out["mean"]) / out["std"]

    position = 0
    positions = []
    for i in range(len(out)):
        z = out["zscore"].iloc[i]
        if pd.isna(z):
            positions.append(0)
            continue

        if position == 0:
            if z < -z_entry:
                position = 1   # ratio low → long A, short B
            elif z > z_entry:
                position = -1  # ratio high → short A, long B
        elif position == 1:
            if z >= -z_exit or z < -z_stop:
                position = 0
        elif position == -1:
            if z <= z_exit or z > z_stop:
                position = 0

        positions.append(position)

    out["position"] = positions
    # PnL: long ratio profit when ratio rises
    out["ratio_change"] = out["ratio"].pct_change()
    out["pnl"] = out["position"].shift(1) * out["ratio_change"]
    out["equity"] = (1 + out["pnl"].fillna(0)).cumprod()

    return out


def basis_trade(
    spot_price: pd.Series,
    futures_price: pd.Series,
    days_to_expiry: pd.Series,
    entry_basis_annual: float = 0.05,  # 5% annualized
    exit_at_basis: float = 0.005,       # 0.5% annualized
) -> pd.DataFrame:
    """
    Cash-and-carry basis trade.

    When futures > spot (contango) by significant amount:
    - Long spot, short futures
    - Lock in basis as profit until expiry

    Args:
        spot_price: Spot price series.
        futures_price: Futures price series.
        days_to_expiry: Days until futures expiry (per row).
        entry_basis_annual: Annualized basis to trigger entry.
        exit_at_basis: Annualized basis to exit (basis compressed).

    Returns:
        DataFrame with basis_pct, basis_annual, position, pnl.
    """
    common = spot_price.dropna().index.intersection(futures_price.dropna().index)
    spot = spot_price.loc[common]
    fut = futures_price.loc[common]
    dte = days_to_expiry.loc[common]

    out = pd.DataFrame({"spot": spot, "futures": fut, "dte": dte})
    out["basis_pct"] = (fut - spot) / spot
    out["basis_annual"] = out["basis_pct"] * (365 / dte.replace(0, 1))

    position = 0
    positions = []
    for i in range(len(out)):
        annual = out["basis_annual"].iloc[i]
        if pd.isna(annual):
            positions.append(0)
            continue

        if position == 0:
            if annual > entry_basis_annual:
                position = 1   # long spot, short futures
            elif annual < -entry_basis_annual:
                position = -1  # short spot, long futures (backwardation)
        elif position == 1 and annual < exit_at_basis:
            position = 0
        elif position == -1 and annual > -exit_at_basis:
            position = 0

        positions.append(position)

    out["position"] = positions

    # PnL: basis convergence captures the premium over time
    # Per day, capture roughly basis_annual / 365 if hedged
    out["daily_capture"] = out["position"].shift(1) * out["basis_annual"].shift(1) / 365
    out["equity"] = (1 + out["daily_capture"].fillna(0)).cumprod()

    return out


if __name__ == "__main__":
    print("=" * 70)
    print("Crypto Mean Reversion Strategies — Demo")
    print("=" * 70)

    # Demo 1: Funding rate strategy
    print("\n--- Funding Rate Mean Reversion ---")
    np.random.seed(42)
    n = 365 * 3  # 8-hourly bars across 3 days = 9 bars/day
    n = 365 * 3  # 1095 days
    dates = pd.date_range("2023-01-01", periods=n * 3, freq="8h")

    # Simulate funding: normally ~0.01%, occasional extremes
    funding = pd.Series(
        np.random.randn(len(dates)) * 0.0001 + 0.00005,
        index=dates,
    )
    # Inject extreme periods
    funding.iloc[100:120] = 0.0008   # high funding period (bull squeeze)
    funding.iloc[500:515] = -0.0005  # negative funding (bear squeeze)
    funding.iloc[800:830] = 0.0006

    signals = funding_rate_strategy(funding)
    n_short_perp = (signals == -1).sum()
    n_long_perp = (signals == 1).sum()
    print(f"Bars short-perp:    {n_short_perp} ({n_short_perp/len(signals)*100:.1f}%)")
    print(f"Bars long-perp:     {n_long_perp} ({n_long_perp/len(signals)*100:.1f}%)")
    print(f"Bars flat:          {(signals == 0).sum()}")

    # Demo 2: Cross-coin ratio MR
    print("\n--- BTC/ETH Ratio Mean Reversion ---")
    np.random.seed(7)
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="D")

    btc = 30000 * np.exp(np.cumsum(np.random.randn(n) * 0.03))
    # ETH correlated to BTC but with mean-reverting deviation
    eth_factor = np.cumsum(np.random.randn(n) * 0.01)
    # Add MR component to deviation
    for i in range(1, n):
        eth_factor[i] = 0.95 * eth_factor[i-1] + np.random.randn() * 0.01
    eth = btc / 17 * np.exp(eth_factor)

    btc_s = pd.Series(btc, index=dates)
    eth_s = pd.Series(eth, index=dates)

    result = coin_ratio_mr(btc_s, eth_s, lookback=30, z_entry=1.5, z_exit=0.3)
    n_trades = int((result["position"].diff().abs() > 0).sum() / 2)
    final_equity = result["equity"].iloc[-1]
    print(f"Total trades:       {n_trades}")
    print(f"Final equity:       {final_equity:.3f}x")
    print(f"Total return:       {(final_equity-1)*100:.1f}%")

    # Demo 3: Basis trade
    print("\n--- Basis Trade (cash-and-carry) ---")
    np.random.seed(42)
    n = 90  # 3 months
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    spot = pd.Series(
        43000 * np.exp(np.cumsum(np.random.randn(n) * 0.02)),
        index=dates,
    )
    # Futures with declining basis (contango compresses near expiry)
    initial_basis = 0.02  # 2%
    basis_decay = np.linspace(initial_basis, 0, n)
    futures = spot * (1 + basis_decay)
    dte = pd.Series(np.arange(90, 0, -1), index=dates)

    result = basis_trade(spot, futures, dte, entry_basis_annual=0.05)
    print(f"Days long basis:   {(result['position'] == 1).sum()}")
    print(f"Days flat:         {(result['position'] == 0).sum()}")
    print(f"Final equity:      {result['equity'].iloc[-1]:.4f}x")
    print(f"Annualized return: {(result['equity'].iloc[-1] - 1) * (365/n) * 100:.2f}%")
