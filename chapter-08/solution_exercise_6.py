"""
QuantCFD — Chương 8, Bài tập 6 (BONUS, 90 phút)
Statistical tests applied to 5 instruments

Yêu cầu:
- Run Hurst, ADF, half-life trên 5 candidate series
- Identify which suitable cho MR strategy
- Practice interpreting results
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")

from statistical_tests import mr_validation_report


def synthesize_instrument(kind: str, n: int = 1500, seed: int = 42) -> pd.Series:
    """Generate synthetic series với specified MR/trend properties."""
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    if kind == "ou_mr":
        # Strong mean-reverting (Ornstein-Uhlenbeck)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = x[i-1] + 0.15 * (0.0 - x[i-1]) + np.random.randn()
        return pd.Series(x, index=dates)

    elif kind == "random_walk":
        return pd.Series(np.cumsum(np.random.randn(n)), index=dates)

    elif kind == "trending":
        drift = 0.001
        rets = np.random.randn(n) * 0.01 + drift
        return pd.Series(np.cumsum(rets), index=dates)

    elif kind == "mr_then_trending":
        # First half MR, second half trending (regime change)
        x = np.zeros(n)
        for i in range(1, n // 2):
            x[i] = x[i-1] + 0.15 * (0.0 - x[i-1]) + np.random.randn()
        # Trending second half
        rets = np.random.randn(n - n // 2) * 0.5 + 0.05
        x[n // 2:] = x[n // 2 - 1] + np.cumsum(rets)
        return pd.Series(x, index=dates)

    elif kind == "spread":
        # Cointegrated spread
        common = np.cumsum(np.random.randn(n) * 0.01)
        a = common + np.cumsum(np.random.randn(n) * 0.005)
        b = 0.85 * common + np.cumsum(np.random.randn(n) * 0.005)
        spread = a - 0.85 * b
        return pd.Series(spread, index=dates)

    else:
        raise ValueError(f"Unknown kind: {kind}")


if __name__ == "__main__":
    instruments = [
        ("OU mean-reverting", "ou_mr", 42),
        ("Random walk", "random_walk", 100),
        ("Trending (drift)", "trending", 7),
        ("Regime-change", "mr_then_trending", 13),
        ("Cointegrated spread", "spread", 99),
    ]

    print("=" * 70)
    print("Bài tập 6 (BONUS) — Statistical Tests on 5 Instruments")
    print("=" * 70)

    summary = []
    for name, kind, seed in instruments:
        series = synthesize_instrument(kind, seed=seed)
        report = mr_validation_report(series, name)

        print(f"\n{'─' * 70}")
        print(f"Instrument: {name}")
        print(f"{'─' * 70}")
        print(f"  Hurst exponent:    {report['hurst']:.3f}  "
              f"({'✓' if report['hurst_pass'] else '✗'})")
        if report['adf_p_value'] is not None:
            print(f"  ADF p-value:       {report['adf_p_value']:.4f}  "
                  f"({'✓' if report['adf_pass'] else '✗'})")
        if (not np.isnan(report['half_life_bars'])
                and np.isfinite(report['half_life_bars'])):
            print(f"  Half-life:         {report['half_life_bars']:.1f} bars  "
                  f"({'✓' if report['half_life_valid'] else '✗'})")
        else:
            print(f"  Half-life:         not finite (not MR)")
        print(f"  Tests passed:      {report['tests_passed']}")
        print(f"  Verdict:           {report['verdict']}")

        summary.append({
            "instrument": name,
            "hurst": report["hurst"],
            "adf_p": report.get("adf_p_value"),
            "half_life": report["half_life_bars"],
            "tests": report["tests_passed"],
            "verdict": report["verdict"][:25],
        })

    print("\n" + "=" * 70)
    print("Summary Table")
    print("=" * 70)
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    print("\n" + "─" * 70)
    print("LESSONS:")
    print("─" * 70)
    print("  - Hurst < 0.5 + ADF p < 0.05 + half-life finite = MR-deployable")
    print("  - Random walk fails ADF (p ≥ 0.05) — KHÔNG MR")
    print("  - Trending series fails Hurst (H ≥ 0.5) — KHÔNG MR")
    print("  - Regime-change: tests pass partially → unstable, AVOID")
    print("  - Cointegrated spread: thường strong MR (3/3 tests)")
    print("\n  Best practice: monitor Hurst rolling 6-month — abandon strategy")
    print("  nếu instrument shifts từ MR → trending (Hurst climbs > 0.55)")
