"""
QuantCFD — Chương 8.7.5
Risk Management for Mean Reversion

MR has different risk profile than trend:
- Higher win rate but lower avg profit
- Lower max single-trade loss but higher tail risk
- Position sizing must be more conservative
- Time-based stops critical
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def mr_position_size(
    equity: float,
    entry_zscore: float,
    stop_zscore: float,
    realized_atr: float,
    contract_value_per_point: float = 1.0,
    risk_pct: float = 0.005,
) -> float:
    """
    Position sizing for MR with conservative risk_pct.

    Default 0.5% per trade (vs trend 1-2%) — because MR has tail risk.

    Args:
        equity: Account equity in USD.
        entry_zscore: Entry Z-score (typically -2 oversold, +2 overbought).
        stop_zscore: Stop loss Z-score (typically ±3.5 extreme).
        realized_atr: Recent ATR or std of price.
        contract_value_per_point: Dollar per 1 point per 1 lot.
        risk_pct: Max risk per trade (default 0.5%).

    Returns:
        Position size in lots.
    """
    risk_amount = equity * risk_pct
    stop_distance_pts = abs(entry_zscore - stop_zscore) * realized_atr

    if stop_distance_pts <= 0:
        return 0

    size = risk_amount / (stop_distance_pts * contract_value_per_point)
    return size


def time_based_exit(
    df: pd.DataFrame,
    entry_signal: pd.Series,
    half_life_bars: int,
    max_holding_mult: float = 2.0,
) -> pd.Series:
    """
    Generate exit signal based on time held > 2× half_life.

    Args:
        df: Price data.
        entry_signal: 1 = enter, 0 = no entry.
        half_life_bars: Half-life of MR series.
        max_holding_mult: Multiplier (default 2.0).

    Returns:
        Series with 1 = position open, 0 = closed.
    """
    max_holding = int(half_life_bars * max_holding_mult)
    position = 0
    bars_held = 0
    positions = []

    for i in range(len(df)):
        sig = entry_signal.iloc[i] if i < len(entry_signal) else 0

        if position == 0:
            if sig == 1:
                position = 1
                bars_held = 1
        else:
            bars_held += 1
            if bars_held >= max_holding:
                position = 0
                bars_held = 0

        positions.append(position)

    return pd.Series(positions, index=df.index)


def correlation_alert(
    positions_returns: dict,
    threshold: float = 0.3,
    lookback_short: int = 10,
    lookback_long: int = 60,
) -> dict:
    """
    Alert when avg correlation jumps recently vs baseline.

    Crisis warning: correlations of MR positions jumping toward 1
    means crisis incoming → reduce size.

    Args:
        positions_returns: Dict of {position_name: returns_series}.
        threshold: Correlation jump triggering alert.
        lookback_short: Recent window.
        lookback_long: Baseline window.

    Returns:
        Dict with alert_triggered, recent_corr, baseline_corr, jump.
    """
    df = pd.DataFrame(positions_returns).dropna()

    if len(df) < lookback_long:
        return {
            "alert_triggered": False,
            "recent_corr": np.nan,
            "baseline_corr": np.nan,
            "jump": 0,
            "message": "Insufficient data",
        }

    # Recent correlation
    recent = df.tail(lookback_short)
    recent_corr_arr = recent.corr().abs().to_numpy().copy()
    np.fill_diagonal(recent_corr_arr, np.nan)
    avg_recent = np.nanmean(recent_corr_arr)

    # Baseline (older data, excluding recent)
    baseline = df.iloc[-(lookback_long):-(lookback_short)]
    base_corr_arr = baseline.corr().abs().to_numpy().copy()
    np.fill_diagonal(base_corr_arr, np.nan)
    avg_base = np.nanmean(base_corr_arr)

    jump = avg_recent - avg_base

    return {
        "alert_triggered": jump > threshold,
        "recent_corr": avg_recent,
        "baseline_corr": avg_base,
        "jump": jump,
        "message": (f"CRISIS WARNING: correlations jumped {jump:.2f} above baseline"
                    if jump > threshold else "Normal correlation regime"),
    }


def drawdown_scaledown_rules(current_dd: float) -> float:
    """
    Position size multiplier based on drawdown level.

    Pre-commit rules:
    - DD < -10%: full size (1.0×)
    - DD -10% to -20%: half size (0.5×)
    - DD -20% to -30%: quarter size (0.25×)
    - DD < -30%: stop new entries (0×)

    Args:
        current_dd: Current drawdown as negative fraction (e.g. -0.15).

    Returns:
        Size multiplier (0 to 1).
    """
    dd = abs(current_dd)
    if dd < 0.10:
        return 1.0
    elif dd < 0.20:
        return 0.5
    elif dd < 0.30:
        return 0.25
    else:
        return 0.0


def combined_size_with_risk_controls(
    equity: float,
    entry_zscore: float,
    stop_zscore: float,
    realized_atr: float,
    contract_value: float,
    current_dd: float,
    correlation_alert_active: bool,
    base_risk_pct: float = 0.005,
) -> float:
    """
    Final position sizing combining all risk controls:
    1. Base sizing (Z-score and stop)
    2. Drawdown scale-down
    3. Correlation crisis halving

    Returns:
        Position size in lots after all adjustments.
    """
    base_size = mr_position_size(
        equity, entry_zscore, stop_zscore, realized_atr,
        contract_value, base_risk_pct
    )

    # Apply drawdown scale-down
    dd_mult = drawdown_scaledown_rules(current_dd)

    # Apply correlation alert halving
    corr_mult = 0.5 if correlation_alert_active else 1.0

    final_size = base_size * dd_mult * corr_mult
    return final_size


if __name__ == "__main__":
    print("=" * 70)
    print("Mean Reversion Risk Management — Demo")
    print("=" * 70)

    # Demo: position sizing for XAUUSD
    print("\n--- Position Sizing for XAUUSD H4 ---")
    print("Setup: equity $10k, entry Z=-2, stop Z=-3.5, ATR=$25")
    size = mr_position_size(
        equity=10000,
        entry_zscore=-2.0,
        stop_zscore=-3.5,
        realized_atr=25,
        contract_value_per_point=100,  # 100 oz/lot for XAU
        risk_pct=0.005,
    )
    print(f"Position size: {size:.4f} lots ({size * 100:.0f} oz)")
    print(f"Risk: ${10000 * 0.005:.0f} (0.5% of $10k)")

    # Demo: drawdown scale-down
    print("\n--- Drawdown Scale-Down ---")
    for dd in [-0.05, -0.12, -0.22, -0.32]:
        mult = drawdown_scaledown_rules(dd)
        print(f"  DD = {dd*100:.0f}% → size multiplier {mult:.2f}×")

    # Demo: correlation alert
    print("\n--- Correlation Alert ---")
    np.random.seed(42)
    # Simulate normal then crisis correlation
    normal_returns = {f"pos_{i}": np.random.randn(60) * 0.01 for i in range(4)}
    # In recent 10 days, correlate them all
    crisis_factor = np.random.randn(10) * 0.02
    for k in normal_returns:
        normal_returns[k][-10:] = crisis_factor + np.random.randn(10) * 0.005
    normal_returns_df = {k: pd.Series(v) for k, v in normal_returns.items()}

    alert = correlation_alert(normal_returns_df, threshold=0.2)
    print(f"  Baseline correlation: {alert['baseline_corr']:.3f}")
    print(f"  Recent correlation:   {alert['recent_corr']:.3f}")
    print(f"  Jump:                 {alert['jump']:.3f}")
    print(f"  Alert:                {alert['alert_triggered']}")
    print(f"  Message:              {alert['message']}")

    # Demo: combined sizing
    print("\n--- Combined Risk-Adjusted Size ---")
    for scenario in [
        ("Normal", -0.05, False),
        ("Mild DD", -0.15, False),
        ("Crisis DD", -0.25, True),
        ("Catastrophic", -0.35, True),
    ]:
        name, dd, corr_alert = scenario
        size = combined_size_with_risk_controls(
            equity=10000, entry_zscore=-2, stop_zscore=-3.5,
            realized_atr=25, contract_value=100,
            current_dd=dd, correlation_alert_active=corr_alert,
        )
        print(f"  {name:15s} DD={dd*100:.0f}% corr={corr_alert} → "
              f"size {size:.4f} lots")
