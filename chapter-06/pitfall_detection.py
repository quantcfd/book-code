"""
QuantCFD Chương 6 — pitfall_detection.py
Detect 7 bad-faith backtest tricks (section 6.12 of book).

Usage:
    from pitfall_detection import detect_cherry_picking, parameter_sensitivity_check, ...
"""
import numpy as np
import pandas as pd
from metrics import sharpe_ratio


def detect_cherry_picking(
    returns_full: pd.Series,
    claimed_period_start: str,
    claimed_period_end: str,
) -> dict:
    """Trick 1: cherry-picked period.
    So Sharpe của claimed period vs toàn data."""
    sharpe_full = sharpe_ratio(returns_full)
    claimed = returns_full.loc[claimed_period_start:claimed_period_end]
    sharpe_claimed = sharpe_ratio(claimed)
    excluded_before = returns_full.loc[:claimed_period_start]
    excluded_after = returns_full.loc[claimed_period_end:]

    return {
        'sharpe_full':            sharpe_full,
        'sharpe_claimed':         sharpe_claimed,
        'sharpe_excluded_before': sharpe_ratio(excluded_before) if len(excluded_before) > 30 else None,
        'sharpe_excluded_after':  sharpe_ratio(excluded_after) if len(excluded_after) > 30 else None,
        'cherry_picking_score':   sharpe_claimed - sharpe_full,
        'flag':                   abs(sharpe_claimed - sharpe_full) > 1.0,
    }


def parameter_sensitivity_check(
    returns_function,
    best_params: dict,
    perturbation: float = 0.1,
) -> dict:
    """Trick 2: param overfit. Sharpe nên stable trong neighborhood."""
    base_sharpe = sharpe_ratio(returns_function(best_params))
    perturbed_sharpes = []
    for p_name, p_val in best_params.items():
        for delta in [-perturbation, perturbation]:
            new_params = dict(best_params)
            new_params[p_name] = p_val * (1 + delta)
            try:
                perturbed_sharpes.append(sharpe_ratio(returns_function(new_params)))
            except Exception:
                pass

    if not perturbed_sharpes:
        return {'base_sharpe': base_sharpe, 'is_robust': False, 'reason': 'No perturbations succeeded'}

    return {
        'base_sharpe':           base_sharpe,
        'min_perturbed_sharpe':  min(perturbed_sharpes),
        'mean_perturbed_sharpe': np.mean(perturbed_sharpes),
        'std_perturbed':         np.std(perturbed_sharpes),
        'stability_ratio':       min(perturbed_sharpes) / base_sharpe if base_sharpe > 0 else 0,
        'is_robust':             min(perturbed_sharpes) / base_sharpe > 0.7 if base_sharpe > 0 else False,
    }


def cost_attribution(
    gross_returns: pd.Series,
    net_returns: pd.Series,
    n_trades: int,
    initial_equity: float,
) -> dict:
    """Trick 4: gross vs net (no costs). Compute cost drag."""
    total_gross = (1 + gross_returns).prod() - 1
    total_net = (1 + net_returns).prod() - 1
    cost_drag = total_gross - total_net
    gross_sharpe = sharpe_ratio(gross_returns)
    net_sharpe = sharpe_ratio(net_returns)

    return {
        'gross_return':       total_gross,
        'net_return':         total_net,
        'cost_drag':          cost_drag,
        'cost_drag_pct':      cost_drag / abs(total_gross) * 100 if total_gross != 0 else 0,
        'avg_cost_per_trade': cost_drag * initial_equity / n_trades if n_trades > 0 else 0,
        'gross_sharpe':       gross_sharpe,
        'net_sharpe':         net_sharpe,
        'sharpe_decay':       1 - net_sharpe / gross_sharpe if gross_sharpe > 0 else 0,
        'red_flag_costs':     net_sharpe < gross_sharpe * 0.5,
    }


def leverage_check(
    claimed_sharpe: float,
    unleveraged_sharpe: float,
) -> dict:
    """Trick 5: leverage Sharpe inflation.
    Theoretically Sharpe invariant to leverage. Big difference = red flag."""
    diff = abs(claimed_sharpe - unleveraged_sharpe)
    return {
        'claimed_sharpe':       claimed_sharpe,
        'unleveraged_sharpe':   unleveraged_sharpe,
        'difference':           diff,
        'flag':                 diff > 0.3,
        'note':                 ('Sharpe should be invariant to leverage. '
                                 'Big difference suggests fee/cost calculation issue.'),
    }


def start_date_sensitivity(
    returns_full: pd.Series,
    claimed_start: str,
    test_offsets_months: list = None,
) -> pd.DataFrame:
    """Trick 6: pegged start date.
    Test starting 1, 3, 6 months earlier — Sharpe nên không drop nhiều."""
    if test_offsets_months is None:
        test_offsets_months = [1, 3, 6, 12]

    results = []
    claimed_start_ts = pd.Timestamp(claimed_start)
    sharpe_claimed = sharpe_ratio(returns_full.loc[claimed_start:])
    results.append({'offset_months': 0, 'start_date': claimed_start, 'sharpe': sharpe_claimed})

    for offset in test_offsets_months:
        earlier_start = claimed_start_ts - pd.DateOffset(months=offset)
        if earlier_start < returns_full.index[0]:
            continue
        sub = returns_full.loc[earlier_start:]
        if len(sub) < 100:
            continue
        results.append({
            'offset_months': -offset,
            'start_date':    earlier_start.date(),
            'sharpe':        sharpe_ratio(sub),
        })

    return pd.DataFrame(results)


def metric_combination_check(metrics_dict: dict) -> list:
    """Trick 7: CAGR-only or unrealistic combinations.
    Check if metrics dict claims unrealistic combinations."""
    flags = []

    cagr = metrics_dict.get('cagr', 0)
    sharpe = metrics_dict.get('sharpe', None)
    max_dd = metrics_dict.get('max_dd', None)

    if sharpe is None and cagr > 0.30:
        flags.append(f"⚠ CAGR {cagr*100:.0f}% reported without Sharpe — request risk-adjusted metric")

    if max_dd is None and cagr > 0.20:
        flags.append(f"⚠ CAGR {cagr*100:.0f}% reported without Max DD — request drawdown info")

    if sharpe is not None and max_dd is not None:
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        if calmar > 5.0:
            flags.append(f"🔴 Calmar {calmar:.1f} > 5.0 — Renaissance-tier, very rare. Investigate")

        if sharpe > 3.5:
            flags.append(f"🔴 Sharpe {sharpe:.1f} > 3.5 — likely overfit or cost ignored")

        # Sharpe vs vol consistency
        annual_vol_implied = (cagr - 0.05) / sharpe if sharpe > 0 else 0  # assume Rf=5%
        if 0 < annual_vol_implied < 0.03:
            flags.append(f"⚠ Implied vol {annual_vol_implied*100:.1f}% < 3% — verify, might be unrealistic")

    return flags


def full_audit_report(
    returns: pd.Series,
    metrics_claimed: dict = None,
    period_claimed: tuple = None,
) -> str:
    """Run all detection checks. Return text report."""
    lines = ["═══ FULL AUDIT REPORT ═══", ""]

    sharpe = sharpe_ratio(returns)
    lines.append(f"Recomputed Sharpe (full data):  {sharpe:.2f}")

    if period_claimed:
        cherry = detect_cherry_picking(returns, period_claimed[0], period_claimed[1])
        lines.append(f"\n--- Cherry-picking check ---")
        lines.append(f"  Full sharpe:          {cherry['sharpe_full']:.2f}")
        lines.append(f"  Claimed period sharpe: {cherry['sharpe_claimed']:.2f}")
        lines.append(f"  Cherry score:          {cherry['cherry_picking_score']:+.2f}")
        if cherry['flag']:
            lines.append(f"  🔴 FLAG: claimed period much better than full")

    if metrics_claimed:
        lines.append(f"\n--- Metric combination check ---")
        flags = metric_combination_check(metrics_claimed)
        if flags:
            for f in flags:
                lines.append(f"  {f}")
        else:
            lines.append(f"  ✓ No flags from claimed metrics")

    lines.append("\n═════════════════════════")
    return "\n".join(lines)


if __name__ == '__main__':
    np.random.seed(99)
    dates = pd.date_range('2018-01-01', periods=2200, freq='D')

    # Synthetic: tốt 2020-2022, kém các năm khác
    base = np.random.normal(0.0001, 0.012, 2200)
    good_period_mask = (dates >= '2020-01-01') & (dates < '2023-01-01')
    base[good_period_mask] += 0.0010
    returns = pd.Series(base, index=dates)

    print("=== Cherry-picking Detection ===")
    result = detect_cherry_picking(returns, '2020-01-01', '2022-12-31')
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k:30s} {v:+.3f}")
        else:
            print(f"  {k:30s} {v}")

    print("\n=== Start Date Sensitivity ===")
    sens = start_date_sensitivity(returns, '2020-01-01', [3, 6, 12, 24])
    print(sens.to_string(index=False))

    print("\n=== Metric Combination Check ===")
    bad = {'cagr': 0.45, 'sharpe': 4.2, 'max_dd': -0.05}
    flags = metric_combination_check(bad)
    for f in flags:
        print(f"  {f}")

    print("\n=== Full Audit Report ===")
    print(full_audit_report(returns,
                            metrics_claimed={'cagr': 0.20, 'sharpe': 2.5, 'max_dd': -0.08},
                            period_claimed=('2020-01-01', '2022-12-31')))
