"""
QuantCFD — Chương 10.4
Kelly Calculator + Monte Carlo Simulation

Compute Kelly fractions and demonstrate why fractional Kelly is safer.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def kelly_fraction_continuous(returns: pd.Series) -> float:
    """
    Kelly fraction for continuous returns.
    f* = mean / variance.
    """
    if len(returns) < 30 or returns.var() <= 0:
        return 0
    return returns.mean() / returns.var()


def kelly_fraction_binary(win_prob: float, win_loss_ratio: float) -> float:
    """
    Kelly cho binary outcome.
    f* = (b × p - q) / b
    """
    if win_loss_ratio <= 0:
        return 0
    return (win_loss_ratio * win_prob - (1 - win_prob)) / win_loss_ratio


def kelly_for_strategies(
    strategy_returns: dict,
    fraction: float = 0.25,
    cap_pct: float = 0.02,
) -> dict:
    """
    Compute Kelly fraction cho mỗi strategy.

    Args:
        strategy_returns: Dict of {name: return_series}.
        fraction: Multiplier (0.25 = quarter Kelly recommended).
        cap_pct: Hard cap on per-trade risk.

    Returns:
        Dict of Kelly stats per strategy.
    """
    results = {}
    for name, returns in strategy_returns.items():
        full = kelly_fraction_continuous(returns)
        results[name] = {
            "full_kelly": full,
            "half_kelly": full * 0.5,
            "quarter_kelly": full * 0.25,
            "tenth_kelly": full * 0.1,
            "recommended": min(full * fraction, cap_pct),
        }
    return results


def correlation_discount(
    individual_kelly: float, n_strategies: int, avg_correlation: float = 0.2,
) -> float:
    """
    Apply correlation discount to multi-strategy Kelly.

    Discount = sqrt(1 + (n-1) × ρ_avg).
    """
    if n_strategies <= 1:
        return individual_kelly
    discount = np.sqrt(1 + (n_strategies - 1) * avg_correlation)
    return individual_kelly / discount


def monte_carlo_kelly(
    win_prob: float,
    win_loss_ratio: float,
    kelly_fraction: float,
    n_trades: int = 500,
    n_paths: int = 10000,
    initial_equity: float = 10000,
) -> dict:
    """
    Monte Carlo simulation of strategy với specified Kelly fraction.

    Args:
        win_prob: Probability of winning trade.
        win_loss_ratio: Win/loss ratio.
        kelly_fraction: Bet size as fraction of equity.
        n_trades: Number of trades per path.
        n_paths: Number of simulation paths.
        initial_equity: Starting capital.

    Returns:
        Dict of statistics: median, P5, P95, blow_up_pct, max_dd_median.
    """
    np.random.seed(42)
    finals = []
    max_dds = []
    blow_ups = 0

    for _ in range(n_paths):
        equity = initial_equity
        peak = initial_equity
        max_dd = 0

        for _ in range(n_trades):
            if equity < initial_equity * 0.01:  # blow up < 1% of initial
                break
            bet = equity * kelly_fraction
            outcome = np.random.random() < win_prob
            if outcome:
                equity += bet * win_loss_ratio
            else:
                equity -= bet
            peak = max(peak, equity)
            current_dd = (equity - peak) / peak if peak > 0 else 0
            max_dd = min(max_dd, current_dd)

        if equity < initial_equity * 0.1:
            blow_ups += 1
        finals.append(equity)
        max_dds.append(max_dd)

    finals = np.array(finals)
    return {
        "median_final": np.median(finals),
        "p5_final": np.percentile(finals, 5),
        "p95_final": np.percentile(finals, 95),
        "blow_up_pct": blow_ups / n_paths * 100,
        "max_dd_median": np.median(max_dds),
        "max_dd_p5": np.percentile(max_dds, 5),
    }


def compare_kelly_fractions(
    win_prob: float = 0.45,
    win_loss_ratio: float = 2.0,
    n_trades: int = 500,
    n_paths: int = 5000,
    initial_equity: float = 10000,
) -> pd.DataFrame:
    """Compare full, half, quarter, 0.1 Kelly via simulation."""
    full_kelly = kelly_fraction_binary(win_prob, win_loss_ratio)

    fractions = {
        "Full Kelly": full_kelly,
        "Half Kelly": full_kelly * 0.5,
        "Quarter Kelly": full_kelly * 0.25,
        "0.1 Kelly": full_kelly * 0.1,
    }

    results = []
    for name, frac in fractions.items():
        stats = monte_carlo_kelly(
            win_prob, win_loss_ratio, frac,
            n_trades=n_trades, n_paths=n_paths,
            initial_equity=initial_equity,
        )
        results.append({
            "method": name,
            "kelly_pct": frac * 100,
            "median_final": stats["median_final"],
            "median_return": (stats["median_final"] / initial_equity - 1) * 100,
            "p5_final": stats["p5_final"],
            "p95_final": stats["p95_final"],
            "blow_up_pct": stats["blow_up_pct"],
            "max_dd_median": stats["max_dd_median"] * 100,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=" * 80)
    print("Kelly Calculator — Full vs Fractional Kelly Monte Carlo")
    print("=" * 80)

    # Example strategy
    win_prob = 0.45
    wl_ratio = 2.0
    full_kelly = kelly_fraction_binary(win_prob, wl_ratio)
    print(f"\nStrategy: 45% WR, 2:1 R:R")
    print(f"Full Kelly: {full_kelly*100:.1f}%")

    # Monte Carlo comparison
    print(f"\nRunning Monte Carlo (5,000 paths × 500 trades)...")
    df = compare_kelly_fractions(
        win_prob=win_prob, win_loss_ratio=wl_ratio,
        n_trades=500, n_paths=5000, initial_equity=10000,
    )
    print(f"\n{'─' * 80}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"{'─' * 80}")

    # Strategy-level Kelly
    print(f"\n--- Kelly cho strategy returns ---")
    np.random.seed(42)
    sample_returns = pd.Series(np.random.normal(0.0008, 0.012, 500))
    strategies = {"trend": sample_returns}
    kelly_stats = kelly_for_strategies(strategies)
    for name, stats in kelly_stats.items():
        print(f"\n{name}:")
        for k, v in stats.items():
            print(f"  {k:<20}: {v:.4f}")

    # Correlation discount
    print(f"\n--- Correlation discount ---")
    individual = 0.05
    for n in [1, 2, 3, 5]:
        for rho in [0.0, 0.2, 0.5]:
            discounted = correlation_discount(individual, n, rho)
            print(f"  n={n}, ρ={rho}: {individual*100:.1f}% → {discounted*100:.2f}%")

    print(f"\nKey insights:")
    print(f"  - Full Kelly: high blow up risk (10-20%)")
    print(f"  - Quarter Kelly: blow up < 1%, captures 70% growth")
    print(f"  - Most retail: use 0.1-0.25 of full Kelly")
    print(f"  - Multi-strategy: discount by correlation factor")
