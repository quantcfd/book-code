"""
QuantCFD — Chương 10
Solution Exercise 2 — Kelly Fraction Calculator + Monte Carlo

Compute Kelly fractions, run MC simulation, plot DD distributions.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from kelly_calculator import (
    kelly_fraction_continuous,
    kelly_fraction_binary,
    monte_carlo_kelly,
    compare_kelly_fractions,
    correlation_discount,
)


def detailed_kelly_analysis(
    win_prob: float = 0.45,
    win_loss_ratio: float = 2.0,
    n_trades: int = 500,
    n_paths: int = 5000,
    initial_equity: float = 10000,
) -> dict:
    """Run comprehensive Kelly analysis."""
    full_kelly = kelly_fraction_binary(win_prob, win_loss_ratio)
    print(f"Strategy: WR {win_prob*100}%, R:R {win_loss_ratio}:1")
    print(f"Full Kelly: {full_kelly*100:.2f}%")

    # Compare fractions
    print(f"\n{'═' * 80}")
    print(f"Monte Carlo: {n_paths} paths × {n_trades} trades")
    print(f"{'═' * 80}")

    df = compare_kelly_fractions(
        win_prob=win_prob, win_loss_ratio=win_loss_ratio,
        n_trades=n_trades, n_paths=n_paths,
        initial_equity=initial_equity,
    )
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Risk-adjusted score
    print(f"\nRisk-adjusted analysis:")
    for _, row in df.iterrows():
        median_return = row["median_return"]
        blow_up = row["blow_up_pct"]
        risk_adj_score = median_return - blow_up * 5  # heavy penalty on blow up
        print(
            f"  {row['method']:<15}: "
            f"median return {median_return:>6.1f}%, "
            f"blow up {blow_up:>5.1f}%, "
            f"risk-adj score {risk_adj_score:>7.1f}"
        )

    # DD distribution analysis (run separate sims for distribution)
    print(f"\nDD distribution analysis:")
    for fraction_name, multiplier in [("Quarter", 0.25), ("Half", 0.5), ("Full", 1.0)]:
        kelly_pct = full_kelly * multiplier
        stats = monte_carlo_kelly(
            win_prob, win_loss_ratio, kelly_pct,
            n_trades=n_trades, n_paths=n_paths, initial_equity=initial_equity,
        )
        print(f"\n  {fraction_name} Kelly ({kelly_pct*100:.1f}%):")
        print(f"    Median DD: {stats['max_dd_median']*100:.1f}%")
        print(f"    P5 DD:     {stats['max_dd_p5']*100:.1f}%")
        print(f"    Median final equity: ${stats['median_final']:,.0f}")
        print(f"    P5 final equity:     ${stats['p5_final']:,.0f}")
        print(f"    Blow up %:           {stats['blow_up_pct']:.2f}%")

    return {"full_kelly": full_kelly, "comparison": df}


if __name__ == "__main__":
    print("=" * 80)
    print("Bài 2 — Kelly Fraction Monte Carlo")
    print("=" * 80)

    # Test 1: Trend strategy (asymmetric payoff)
    print("\n### Test 1: Trend strategy (40% WR, 2.5:1 R:R)")
    detailed_kelly_analysis(
        win_prob=0.40, win_loss_ratio=2.5, n_paths=3000,
    )

    # Test 2: MR strategy (high WR, low R:R)
    print("\n\n### Test 2: MR strategy (65% WR, 0.8:1 R:R)")
    detailed_kelly_analysis(
        win_prob=0.65, win_loss_ratio=0.8, n_paths=3000,
    )

    # Test 3: Vol BO (medium WR, 1.5:1)
    print("\n\n### Test 3: Vol BO strategy (50% WR, 1.5:1 R:R)")
    detailed_kelly_analysis(
        win_prob=0.50, win_loss_ratio=1.5, n_paths=3000,
    )

    # Continuous returns Kelly
    print("\n\n### Test 4: Kelly cho continuous returns")
    np.random.seed(42)
    sample_returns = pd.Series(np.random.normal(0.001, 0.015, 500))
    full_k = kelly_fraction_continuous(sample_returns)
    print(f"Sample strategy: mean {sample_returns.mean()*100:.3f}%, "
          f"std {sample_returns.std()*100:.3f}%")
    print(f"Full Kelly: {full_k*100:.2f}%")
    print(f"Quarter Kelly (recommended): {full_k*0.25*100:.2f}%")
    print(f"Capped at 2%: {min(full_k*0.25, 0.02)*100:.2f}%")

    # Multi-strategy correlation discount
    print("\n### Test 5: Multi-strategy correlation discount")
    individual = 0.04
    print(f"Individual strategy Kelly: {individual*100}%")
    for n in [2, 3, 5]:
        for rho in [0.0, 0.2, 0.5, 0.7]:
            disc = correlation_discount(individual, n, rho)
            pct_change = (disc - individual) / individual * 100
            print(f"  n={n}, ρ={rho}: {disc*100:.2f}% ({pct_change:+.0f}% from base)")

    print(f"\n--- Conclusions ---")
    print("1. Full Kelly often blow up risk 10-20% in 5 năm")
    print("2. Quarter Kelly is sweet spot: 70% growth, 0.1% blow up")
    print("3. For multi-strategy: apply correlation discount")
    print("4. Cap risk per trade at 1-2% regardless of Kelly calculation")
