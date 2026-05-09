"""
QuantCFD — Chương 10.6
Correlation Tracker

Track pairwise correlations between strategies/assets.
Detect correlation spikes (crisis indicator).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from itertools import combinations


def compute_pairwise_correlations(
    returns_dict: dict, lookback: int = 60,
) -> pd.DataFrame:
    """
    Compute rolling pairwise correlations between all pairs of strategies.

    Args:
        returns_dict: {name: return_series}.
        lookback: Rolling window length.

    Returns:
        DataFrame with rolling correlations, columns named "name1_vs_name2".
    """
    df = pd.DataFrame(returns_dict)
    correlations = pd.DataFrame(index=df.index)

    for name1, name2 in combinations(df.columns, 2):
        col = f"{name1}_vs_{name2}"
        correlations[col] = df[name1].rolling(lookback).corr(df[name2])

    return correlations


def correlation_matrix(
    returns_dict: dict, lookback_days: int = 60,
) -> pd.DataFrame:
    """
    Single correlation matrix for the most recent lookback_days.
    """
    df = pd.DataFrame(returns_dict).tail(lookback_days)
    return df.corr()


def detect_correlation_spike(
    correlations: pd.DataFrame,
    threshold: float = 0.5,
    lookback: int = 5,
) -> dict:
    """
    Alert when avg pairwise correlation spikes.

    Args:
        correlations: DataFrame from compute_pairwise_correlations.
        threshold: Alert if |correlation| > threshold.
        lookback: Average over last N bars.

    Returns:
        Dict with alert flag, current correlations, recommended action.
    """
    if len(correlations) < lookback:
        return {"alert": False, "reason": "insufficient data"}

    recent = correlations.tail(lookback).abs().mean()
    high_pairs = recent[recent > threshold]

    if len(high_pairs) > 0:
        return {
            "alert": True,
            "high_correlation_pairs": high_pairs.to_dict(),
            "max_correlation": recent.max(),
            "avg_correlation": recent.mean(),
            "recommended_action": (
                "Halt new entries, reduce existing positions 50%, "
                "wait for correlation normalization"
            ),
        }
    return {
        "alert": False,
        "max_correlation": recent.max(),
        "avg_correlation": recent.mean(),
    }


def compute_diversification_ratio(
    returns_dict: dict, weights: dict = None, lookback: int = 60,
) -> float:
    """
    Diversification ratio = weighted_avg_vol / portfolio_vol.

    Higher = better diversification benefit.
    Range typically 1.0 (no diversification) to 2.5+ (strong diversification).
    """
    df = pd.DataFrame(returns_dict).tail(lookback)
    if weights is None:
        weights = {name: 1 / len(df.columns) for name in df.columns}

    individual_vols = df.std()
    weighted_avg_vol = sum(
        weights.get(name, 0) * individual_vols[name] for name in df.columns
    )

    portfolio_returns = pd.Series(0.0, index=df.index)
    for name in df.columns:
        portfolio_returns += df[name] * weights.get(name, 0)
    portfolio_vol = portfolio_returns.std()

    if portfolio_vol == 0:
        return 0
    return weighted_avg_vol / portfolio_vol


def crisis_correlation_estimate(
    normal_correlation: float,
    crisis_multiplier: float = 4.0,
    cap: float = 0.85,
) -> float:
    """
    Estimate crisis correlation from normal correlation.

    Empirical: crisis correlation ~3-5x normal, capped at 0.85.
    """
    return min(normal_correlation * crisis_multiplier, cap)


if __name__ == "__main__":
    print("=" * 70)
    print("Correlation Tracker — Demo")
    print("=" * 70)

    # Generate synthetic 3-strategy returns
    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2022-01-01", periods=n, freq="D")

    # Normal regime: low correlation
    base = np.random.randn(n, 3) * 0.012

    # Inject correlation spike around day 500-600
    common_factor = np.random.randn(n) * 0.02
    crisis_period = slice(500, 600)
    base[crisis_period] += common_factor[crisis_period].reshape(-1, 1) * 0.7

    returns_dict = {
        "trend": pd.Series(base[:, 0], index=dates),
        "mr": pd.Series(base[:, 1], index=dates),
        "vol_bo": pd.Series(base[:, 2], index=dates),
    }

    # Compute pairwise correlations
    print("\nComputing rolling correlations (60-day window)...")
    correlations = compute_pairwise_correlations(returns_dict, lookback=60)

    # Correlation matrix at end
    print(f"\nCorrelation matrix (last 60 days):")
    print(correlation_matrix(returns_dict, lookback_days=60).round(3))

    # Check normal vs crisis periods
    print(f"\nNormal period correlations (day 100-200):")
    normal_period = correlations.iloc[100:200]
    print(normal_period.mean().round(3))

    print(f"\nCrisis period correlations (day 500-600):")
    crisis_period = correlations.iloc[500:600]
    print(crisis_period.mean().round(3))

    # Spike detection
    print(f"\n--- Spike detection during crisis ---")
    crisis_section = correlations.iloc[500:560]
    alert = detect_correlation_spike(crisis_section, threshold=0.4)
    print(f"Alert: {alert['alert']}")
    if alert["alert"]:
        print(f"Max correlation: {alert['max_correlation']:.3f}")
        print(f"Avg correlation: {alert['avg_correlation']:.3f}")
        print(f"Recommended action: {alert['recommended_action']}")

    # Diversification ratio
    print(f"\n--- Diversification ratio ---")
    div_ratio = compute_diversification_ratio(returns_dict)
    print(f"DR = {div_ratio:.3f}")
    print(f"  > 2.0 = strong diversification")
    print(f"  1.5-2.0 = moderate")
    print(f"  < 1.5 = weak (strategies too similar)")

    # Crisis correlation estimate
    print(f"\n--- Crisis correlation estimates ---")
    for normal in [0.1, 0.2, 0.3]:
        crisis = crisis_correlation_estimate(normal)
        print(f"  Normal {normal:.1f} → Crisis {crisis:.2f}")
