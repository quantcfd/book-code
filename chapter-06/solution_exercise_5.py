"""
Bài 5 — Stress test 5 scenarios

Apply 5 stress scenarios cho strategy của riêng anh em.
"""
import numpy as np
import pandas as pd
from metrics import sharpe_ratio, max_drawdown


def stress_test_strategy(
    backtest_returns: pd.Series,
    scenarios: dict,
) -> pd.DataFrame:
    """Inject synthetic shock vào ngày middle period."""
    results = []
    base_dd = max_drawdown(backtest_returns)
    base_sharpe = sharpe_ratio(backtest_returns)

    results.append({
        'scenario':         'baseline',
        'shock':            'none',
        'stressed_max_dd':  base_dd['max_drawdown'] * 100,
        'stressed_sharpe':  base_sharpe,
        'recovered':        base_dd['recovered'],
    })

    for name, shock_fn in scenarios.items():
        shocked = shock_fn(backtest_returns.copy())
        dd = max_drawdown(shocked)
        sharpe = sharpe_ratio(shocked)
        results.append({
            'scenario':        name,
            'shock':           '',
            'stressed_max_dd': dd['max_drawdown'] * 100,
            'stressed_sharpe': sharpe,
            'recovered':       dd['recovered'],
        })

    return pd.DataFrame(results)


def main():
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    returns = pd.Series(np.random.normal(0.0006, 0.012, n), index=dates)

    mid = n // 2

    def flash_crash(r):
        r.iloc[mid] = -0.10
        return r

    def multi_day_crash(r):
        for i in range(5):
            r.iloc[mid + i] = -0.03
        return r

    def vol_spike(r):
        for i in range(20):
            r.iloc[mid + i] *= 2.5
        return r

    def trend_reversal(r):
        r.iloc[mid:mid+90] *= -1
        return r

    def slippage_doubling(r):
        r.iloc[mid:mid+30] -= 0.001    # +1 bps daily cost
        return r

    scenarios = {
        'flash_crash':       flash_crash,
        'multi_day_crash':   multi_day_crash,
        'vol_spike':         vol_spike,
        'trend_reversal':    trend_reversal,
        'slippage_doubling': slippage_doubling,
    }

    print("=" * 65)
    print("BÀI 5 — STRESS TEST RESULTS")
    print("=" * 65)
    df = stress_test_strategy(returns, scenarios)
    print(df.to_string(index=False))

    print("\n=== VERDICT ===")
    for _, row in df.iterrows():
        if row['scenario'] == 'baseline':
            continue
        if row['stressed_max_dd'] < -25:
            print(f"  🔴 {row['scenario']}: DD {row['stressed_max_dd']:.1f}% — STRATEGY KHÔNG SỐNG SÓT")
        elif row['stressed_max_dd'] < -15:
            print(f"  ⚠ {row['scenario']}: DD {row['stressed_max_dd']:.1f}% — chấp nhận được nhưng đau")
        else:
            print(f"  ✓ {row['scenario']}: DD {row['stressed_max_dd']:.1f}% — robust")


if __name__ == '__main__':
    main()
