"""
QuantCFD — Chương 10
Solution Exercise 7 (BONUS) — Stress Test Framework

Production stress test framework:
- 5 historical scenarios (2008, 2010, 2015, 2020, 2022)
- Per-scenario PASS/FAIL automation
- Parameter adjustment recommendations
- Multi-portfolio comparison
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from stress_test import (
    HISTORICAL_SCENARIOS,
    stress_test_portfolio,
    run_all_stress_tests,
    stress_test_report,
)


def generate_synthetic_history(
    n_strategies: int = 3,
    start_date: str = "2007-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
) -> dict:
    """Generate synthetic strategy returns covering all stress periods."""
    np.random.seed(seed)
    dates = pd.date_range(start_date, end_date, freq="D")
    n = len(dates)

    strategy_specs = [
        {"name": "trend", "vol": 0.012, "mean": 0.0003, "crisis_drag": 0.6},
        {"name": "mr", "vol": 0.008, "mean": 0.0002, "crisis_drag": 1.5},
        {"name": "vol_bo", "vol": 0.018, "mean": 0.0005, "crisis_drag": 0.4},
    ]

    returns_dict = {}
    for spec in strategy_specs[:n_strategies]:
        rets = np.random.randn(n) * spec["vol"] + spec["mean"]
        returns_dict[spec["name"]] = pd.Series(rets, index=dates)

    # Inject crisis losses for known scenarios
    crisis_dates = [
        ("2008-09-01", "2009-03-31", 0.0040),
        ("2010-05-06", "2010-05-15", 0.0080),
        ("2015-01-15", "2015-01-25", 0.0060),
        ("2020-02-19", "2020-03-23", 0.0050),
        ("2022-02-24", "2022-04-30", 0.0035),
    ]

    for start, end, daily_drag in crisis_dates:
        mask = (dates >= start) & (dates <= end)
        for spec in strategy_specs[:n_strategies]:
            ret_series = returns_dict[spec["name"]]
            ret_series.loc[mask] -= daily_drag * spec["crisis_drag"]
            returns_dict[spec["name"]] = ret_series

    return returns_dict


def recommend_adjustments(results_df: pd.DataFrame) -> list:
    """Generate parameter adjustment recommendations based on stress results."""
    recs = []
    failed = results_df[results_df["verdict"] == "FAIL"]

    if len(failed) == 0:
        recs.append("✓ Portfolio passes all 5 stress scenarios")
        recs.append("  No adjustments needed — current parameters robust")
        return recs

    recs.append(f"⚠ {len(failed)}/{len(results_df)} scenarios FAILED")

    for _, row in failed.iterrows():
        scenario = row["scenario"]
        max_dd = row["max_dd"]
        recs.append(f"\n{scenario}:")
        recs.append(f"  Max DD: {max_dd*100:.2f}%")

        if max_dd < -0.40:
            recs.append("  CRITICAL — strategy failed catastrophically")
            recs.append("  Recommendations:")
            recs.append("    - Reduce per-trade risk 50% (1% → 0.5%)")
            recs.append("    - Lower vol target (10% → 7%)")
            recs.append("    - Add tail hedge (long VIX or OTM puts)")
            recs.append("    - Consider retiring strategies that failed most")
        elif max_dd < -0.30:
            recs.append("  Recommendations:")
            recs.append("    - Reduce per-trade risk 30% (1% → 0.7%)")
            recs.append("    - Tighten total DD halt (-20% → -15%)")
            recs.append("    - Strengthen correlation budget (0.6 → 0.4)")
        else:
            recs.append("  Recommendations:")
            recs.append("    - Minor: tighten daily loss limit (-3% → -2%)")
            recs.append("    - Add equity curve filter for affected strategies")

    return recs


def compare_portfolio_configurations(
    strategies_returns: dict, configurations: dict,
) -> pd.DataFrame:
    """Compare multiple portfolio weight configurations across stress scenarios."""
    all_results = []
    for config_name, weights in configurations.items():
        results = run_all_stress_tests(strategies_returns, weights)
        for _, row in results.iterrows():
            all_results.append({
                "config": config_name,
                "scenario": row["scenario"],
                "max_dd": row["max_dd"],
                "total_return": row["total_return"],
                "verdict": row["verdict"],
            })
    return pd.DataFrame(all_results)


if __name__ == "__main__":
    print("=" * 80)
    print("Bài 7 (BONUS) — Stress Test Framework với 5 Scenarios")
    print("=" * 80)

    # Generate 17-year synthetic history
    print("\nGenerating 17-year synthetic strategy returns...")
    strategies_returns = generate_synthetic_history(
        n_strategies=3, start_date="2007-01-01", end_date="2024-12-31",
    )
    print(f"Period: 2007-2024 ({sum(len(v) for v in strategies_returns.values())//3} days)")

    # Default portfolio
    weights = {"trend": 0.45, "mr": 0.30, "vol_bo": 0.25}

    # Run stress tests
    print("\nRunning stress tests across 5 historical scenarios...\n")
    results = run_all_stress_tests(strategies_returns, weights)
    print(stress_test_report(results))

    # Recommendations
    print(f"\n{'═' * 80}")
    print("PARAMETER ADJUSTMENT RECOMMENDATIONS")
    print(f"{'═' * 80}")
    recs = recommend_adjustments(results)
    for rec in recs:
        print(rec)

    # Compare 3 portfolio configurations
    print(f"\n{'═' * 80}")
    print("MULTI-CONFIG COMPARISON")
    print(f"{'═' * 80}")

    configs = {
        "balanced":      {"trend": 0.45, "mr": 0.30, "vol_bo": 0.25},
        "trend_heavy":   {"trend": 0.70, "mr": 0.15, "vol_bo": 0.15},
        "low_vol":       {"trend": 0.30, "mr": 0.50, "vol_bo": 0.20},
    }

    comparison = compare_portfolio_configurations(strategies_returns, configs)
    print(f"\n{comparison.to_string(index=False, float_format=lambda x: f'{x:.4f}')}")

    # Per-config summary
    print(f"\n--- Per-config summary ---")
    for config_name in configs.keys():
        subset = comparison[comparison["config"] == config_name]
        n_pass = (subset["verdict"] == "PASS").sum()
        n_total = len(subset)
        avg_max_dd = subset["max_dd"].mean()
        avg_return = subset["total_return"].mean()
        print(f"\n  {config_name}:")
        print(f"    PASS rate:     {n_pass}/{n_total}")
        print(f"    Avg Max DD:    {avg_max_dd*100:.2f}%")
        print(f"    Avg return:    {avg_return*100:+.2f}%")

    # Best configuration
    print(f"\n{'═' * 80}")
    print("BEST CONFIGURATION")
    print(f"{'═' * 80}")
    summary = comparison.groupby("config").agg(
        pass_rate=("verdict", lambda x: (x == "PASS").mean()),
        avg_max_dd=("max_dd", "mean"),
        avg_return=("total_return", "mean"),
    )
    print(f"\n{summary.to_string(float_format=lambda x: f'{x:.4f}')}")

    best = summary.sort_values(
        ["pass_rate", "avg_max_dd"], ascending=[False, False],
    ).index[0]
    print(f"\n✓ Recommended configuration: {best}")
    print(f"  Highest PASS rate, lowest avg DD")

    print(f"\n{'═' * 80}")
    print("STRESS TEST SCHEDULE")
    print(f"{'═' * 80}")
    print("""
Recommended frequency:
  Quarterly: full stress test với all 5 scenarios
  Monthly: spot check 1 random scenario
  Annually: review scenario list, add new (e.g., 2025 events)
  Pre-deployment: full stress test khi changing risk parameters

Pass/fail criteria:
  PASS:  max_dd > -30% in ALL scenarios
  FAIL:  any scenario max_dd < -40% OR total return < -50%

If FAIL:
  1. Identify failing scenarios (which years, which strategies)
  2. Apply recommendations từ recommend_adjustments()
  3. Re-test with adjusted parameters
  4. Iterate until PASS
""")
