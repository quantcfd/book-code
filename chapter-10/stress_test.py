"""
QuantCFD — Chương 10.16
Portfolio Stress Testing

Apply portfolio strategies to historical scenarios.
Verify survives 2008, 2010, 2015, 2020, 2022.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


HISTORICAL_SCENARIOS = {
    "2008_GFC": {
        "start": "2008-09-01",
        "end": "2009-03-31",
        "description": "Global Financial Crisis — Lehman collapse to market bottom",
        "expected_max_dd": -0.50,
    },
    "2010_FlashCrash": {
        "start": "2010-05-01",
        "end": "2010-05-31",
        "description": "Flash Crash — May 6 2010, single-day crash",
        "expected_max_dd": -0.10,
    },
    "2015_SNB_Shock": {
        "start": "2015-01-15",
        "end": "2015-01-30",
        "description": "Swiss Franc unpegging — CHF +30% intraday",
        "expected_max_dd": -0.30,
    },
    "2020_COVID": {
        "start": "2020-02-19",
        "end": "2020-04-30",
        "description": "COVID-19 crash and initial recovery",
        "expected_max_dd": -0.34,
    },
    "2022_Russia_Ukraine": {
        "start": "2022-02-24",
        "end": "2022-05-31",
        "description": "War + Fed hike cycle, commodities + crypto crash",
        "expected_max_dd": -0.30,
    },
}


def stress_test_portfolio(
    strategies_returns: dict,
    weights: dict,
    scenario_period: tuple,
) -> dict:
    """
    Apply portfolio to historical scenario.

    Args:
        strategies_returns: Dict {name: full_return_series}.
        weights: Dict {name: portfolio_weight}.
        scenario_period: Tuple (start_date, end_date).

    Returns:
        Dict with scenario stats.
    """
    start, end = scenario_period
    portfolio_ret = pd.Series(dtype=float)

    for name, returns in strategies_returns.items():
        # Slice scenario period
        scenario_ret = returns.loc[start:end] if isinstance(returns.index[0], pd.Timestamp) else returns
        if len(scenario_ret) == 0:
            continue
        weighted = scenario_ret * weights.get(name, 0)
        if len(portfolio_ret) == 0:
            portfolio_ret = weighted.copy()
        else:
            portfolio_ret = portfolio_ret.add(weighted, fill_value=0)

    if len(portfolio_ret) < 2:
        return {"error": "insufficient data"}

    eq = (1 + portfolio_ret).cumprod()
    max_dd = float((eq / eq.cummax() - 1).min())
    total_return = float(eq.iloc[-1] - 1)
    n_days = len(portfolio_ret)

    # Days to recovery (from trough)
    trough_idx = (eq / eq.cummax() - 1).idxmin()
    after_trough = eq.loc[trough_idx:]
    peak_value = eq.loc[:trough_idx].max()
    recovery = after_trough[after_trough >= peak_value]
    days_to_recovery = (
        (recovery.index[0] - trough_idx).days
        if len(recovery) > 0 else None
    )

    return {
        "scenario_start": start,
        "scenario_end": end,
        "n_days": n_days,
        "total_return": total_return,
        "max_dd": max_dd,
        "days_to_recovery": days_to_recovery,
    }


def run_all_stress_tests(
    strategies_returns: dict,
    weights: dict,
    scenarios: dict = None,
    pass_threshold_dd: float = -0.30,
) -> pd.DataFrame:
    """
    Run portfolio through all standard scenarios.

    Args:
        strategies_returns: {name: return_series}.
        weights: {name: weight}.
        scenarios: dict of scenario configs (default uses HISTORICAL_SCENARIOS).
        pass_threshold_dd: Max acceptable DD for PASS.

    Returns:
        DataFrame of results per scenario.
    """
    if scenarios is None:
        scenarios = HISTORICAL_SCENARIOS

    results = []
    for name, config in scenarios.items():
        result = stress_test_portfolio(
            strategies_returns, weights,
            (config["start"], config["end"]),
        )
        if "error" in result:
            results.append({
                "scenario": name,
                "description": config["description"],
                "status": "NO DATA",
                "max_dd": None, "total_return": None, "verdict": "N/A",
            })
            continue

        verdict = "PASS" if result["max_dd"] >= pass_threshold_dd else "FAIL"
        results.append({
            "scenario": name,
            "description": config["description"],
            "n_days": result["n_days"],
            "total_return": result["total_return"],
            "max_dd": result["max_dd"],
            "days_to_recovery": result["days_to_recovery"],
            "verdict": verdict,
        })

    return pd.DataFrame(results)


def stress_test_report(results_df: pd.DataFrame) -> str:
    """Generate text report from stress test results."""
    lines = [
        "=" * 70,
        "PORTFOLIO STRESS TEST REPORT",
        "=" * 70,
    ]

    for _, row in results_df.iterrows():
        lines.append(f"\n{row['scenario']}: {row['description']}")
        if row["verdict"] == "N/A":
            lines.append(f"  Status: NO DATA")
            continue
        lines.append(f"  Days:           {row['n_days']}")
        lines.append(f"  Total return:   {row['total_return']*100:+.2f}%")
        lines.append(f"  Max DD:         {row['max_dd']*100:.2f}%")
        if row.get("days_to_recovery"):
            lines.append(f"  Recovery:       {row['days_to_recovery']} days")
        else:
            lines.append(f"  Recovery:       not yet recovered")
        lines.append(f"  Verdict:        {row['verdict']}")

    # Overall summary
    valid = results_df[results_df["verdict"].isin(["PASS", "FAIL"])]
    if len(valid) > 0:
        n_pass = (valid["verdict"] == "PASS").sum()
        lines.append(f"\n{'─' * 70}")
        lines.append(f"OVERALL: {n_pass}/{len(valid)} scenarios PASSED")
        if n_pass == len(valid):
            lines.append("✓ Portfolio robust across all stress scenarios")
        else:
            lines.append("✗ Portfolio needs refinement — review failed scenarios")

    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 70)
    print("Portfolio Stress Test — Demo")
    print("=" * 70)

    # Build synthetic 7-year return series with realistic crisis events
    np.random.seed(42)
    dates = pd.date_range("2018-01-01", "2024-12-31", freq="D")
    n = len(dates)

    # Trend strategy
    trend_rets = np.random.randn(n) * 0.010 + 0.0002

    # MR strategy
    mr_rets = np.random.randn(n) * 0.007 + 0.0001

    # Vol BO strategy
    vol_bo_rets = np.random.randn(n) * 0.015 + 0.0003

    # Inject crisis losses at appropriate dates
    crisis_dates = [
        ("2020-02-19", "2020-03-23", -0.005),  # COVID
        ("2022-02-24", "2022-04-15", -0.003),  # Russia
    ]
    for start_str, end_str, daily_drag in crisis_dates:
        mask = (dates >= start_str) & (dates <= end_str)
        trend_rets[mask] += daily_drag * 0.7
        mr_rets[mask] += daily_drag * 1.2  # MR fails in crisis
        vol_bo_rets[mask] += daily_drag * 0.5

    strategies_returns = {
        "trend": pd.Series(trend_rets, index=dates),
        "mr": pd.Series(mr_rets, index=dates),
        "vol_bo": pd.Series(vol_bo_rets, index=dates),
    }

    weights = {"trend": 0.45, "mr": 0.30, "vol_bo": 0.25}

    # Run stress tests
    print("\nRunning stress tests across 5 historical scenarios...\n")
    results = run_all_stress_tests(strategies_returns, weights)
    print(stress_test_report(results))
