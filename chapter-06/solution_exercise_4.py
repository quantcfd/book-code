"""
Bài 4 — Pitfall detection trên 3 simulated vendor tear sheets

Apply 7-trick checklist cho 3 simulated vendors. Output 1-page report mỗi vendor.
"""
import numpy as np
import pandas as pd
from pitfall_detection import (
    detect_cherry_picking, start_date_sensitivity,
    metric_combination_check, full_audit_report,
)


def evaluate_vendor(name: str, returns: pd.Series, claims: dict, period_claimed: tuple = None) -> dict:
    score = 10  # max 10
    flags_found = []

    metric_flags = metric_combination_check(claims)
    flags_found.extend(metric_flags)
    score -= len(metric_flags)

    if period_claimed:
        cherry = detect_cherry_picking(returns, period_claimed[0], period_claimed[1])
        if cherry['flag']:
            flags_found.append(f"Cherry-picking: claimed Sharpe {cherry['sharpe_claimed']:.2f} "
                               f"vs full {cherry['sharpe_full']:.2f}")
            score -= 3

        sens = start_date_sensitivity(returns, period_claimed[0], [3, 6, 12])
        if len(sens) > 1:
            min_sharpe = sens['sharpe'].min()
            max_sharpe = sens['sharpe'].max()
            if (max_sharpe - min_sharpe) > 1.0:
                flags_found.append(f"Start date sensitivity: range {min_sharpe:.2f} → {max_sharpe:.2f}")
                score -= 2

    return {
        'name':         name,
        'score':        max(0, score),
        'flags':        flags_found,
        'verdict':      'AVOID' if score <= 4 else ('CAUTION' if score <= 7 else 'OK'),
    }


def main():
    np.random.seed(7)
    dates = pd.date_range('2018-01-01', periods=2200, freq='D')

    # Vendor A: cherry-picked
    base_a = np.random.normal(0.0001, 0.012, 2200)
    base_a[(dates >= '2022-01-01') & (dates < '2024-07-01')] += 0.0015
    returns_a = pd.Series(base_a, index=dates)

    # Vendor B: legitimate
    returns_b = pd.Series(np.random.normal(0.0005, 0.010, 2200), index=dates)

    # Vendor C: leverage-inflated (high return, high vol)
    returns_c = pd.Series(np.random.normal(0.0010, 0.030, 2200), index=dates)

    vendors = [
        ('Vendor A — "Holy Grail" Telegram',
         returns_a,
         {'cagr': 0.63, 'sharpe': 3.84, 'max_dd': -0.082},
         ('2022-01-01', '2024-06-30')),
        ('Vendor B — Mid-size hedge fund',
         returns_b,
         {'cagr': 0.124, 'sharpe': 1.36, 'max_dd': -0.167},
         ('2014-01-01', '2024-12-31')),
        ('Vendor C — "10× leverage signals"',
         returns_c,
         {'cagr': 0.85, 'sharpe': 2.5, 'max_dd': -0.40},
         None),
    ]

    print("=" * 70)
    print("BÀI 4 — VENDOR AUDIT REPORT")
    print("=" * 70)

    for name, returns, claims, period in vendors:
        result = evaluate_vendor(name, returns, claims, period)
        print(f"\n--- {result['name']} ---")
        print(f"Score: {result['score']}/10")
        print(f"Verdict: {result['verdict']}")
        if result['flags']:
            print("Flags found:")
            for f in result['flags']:
                print(f"  - {f}")
        else:
            print("No major flags")


if __name__ == '__main__':
    main()
