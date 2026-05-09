"""
QuantCFD — Chương 10.5.5
Risk Budgeting

Equal risk contribution + risk parity allocation.
Different from capital allocation — allocate risk equally.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def equal_risk_contribution(strategy_vols: dict) -> dict:
    """
    Simple equal risk contribution allocation.
    Allocation_i ∝ 1 / vol_i.

    Args:
        strategy_vols: Dict {name: annualized_vol}.

    Returns:
        Dict {name: weight} normalized to sum 1.0.
    """
    inv_vols = {
        name: 1.0 / vol for name, vol in strategy_vols.items() if vol > 0
    }
    total = sum(inv_vols.values())
    if total == 0:
        return {name: 0 for name in strategy_vols}
    return {name: inv / total for name, inv in inv_vols.items()}


def risk_parity_weights(
    covariance_matrix: pd.DataFrame, max_iter: int = 1000,
) -> dict:
    """
    Risk parity allocation — each asset contributes equal risk.
    Solves: minimize variance of risk contributions.

    Args:
        covariance_matrix: DataFrame of asset return covariances.
        max_iter: Optimization iterations.

    Returns:
        Dict {name: weight}.
    """
    if not HAS_SCIPY:
        # Fallback to equal risk contribution
        vols = {
            name: np.sqrt(covariance_matrix.loc[name, name])
            for name in covariance_matrix.index
        }
        return equal_risk_contribution(vols)

    cov = covariance_matrix.values
    n = cov.shape[0]
    names = list(covariance_matrix.index)

    def risk_contribution(weights):
        weights = np.array(weights)
        port_var = weights @ cov @ weights
        if port_var <= 0:
            return np.ones(n) / n
        marginal = cov @ weights
        return weights * marginal / port_var

    def objective(weights):
        rc = risk_contribution(weights)
        target = 1.0 / n
        return np.sum((rc - target) ** 2)

    w0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.01, 0.99)] * n

    result = minimize(
        objective, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": max_iter},
    )
    return {names[i]: float(result.x[i]) for i in range(n)}


def compute_risk_contributions(
    weights: dict, covariance_matrix: pd.DataFrame,
) -> dict:
    """
    Compute each strategy's risk contribution to portfolio variance.
    """
    cov = covariance_matrix.values
    names = list(covariance_matrix.index)
    w = np.array([weights.get(name, 0) for name in names])
    port_var = w @ cov @ w
    if port_var <= 0:
        return {name: 0 for name in names}
    marginal = cov @ w
    rc = w * marginal / port_var
    return {names[i]: float(rc[i]) for i in range(len(names))}


def compare_allocations(returns_dict: dict, lookback: int = 252) -> dict:
    """
    Compare 3 allocation methods: equal capital, equal risk contrib, risk parity.
    """
    df = pd.DataFrame(returns_dict).tail(lookback)
    cov = df.cov() * 252  # annualized
    vols = {name: np.sqrt(cov.loc[name, name]) for name in cov.index}

    n = len(df.columns)
    allocations = {
        "equal_capital": {name: 1.0 / n for name in df.columns},
        "equal_risk_contrib": equal_risk_contribution(vols),
        "risk_parity": risk_parity_weights(cov),
    }

    results = {}
    for method, weights in allocations.items():
        rc = compute_risk_contributions(weights, cov)

        # Portfolio metrics
        portfolio_ret = pd.Series(0.0, index=df.index)
        for name, w in weights.items():
            if name in df.columns:
                portfolio_ret += df[name] * w

        port_vol = portfolio_ret.std() * np.sqrt(252)
        port_sharpe = (
            (portfolio_ret.mean() / portfolio_ret.std()) * np.sqrt(252)
            if portfolio_ret.std() > 0 else 0
        )

        results[method] = {
            "weights": weights,
            "risk_contributions": rc,
            "portfolio_vol": port_vol,
            "portfolio_sharpe": port_sharpe,
        }

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("Risk Budgeting — 3 allocation methods")
    print("=" * 70)

    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    # 3 strategies với different vols
    returns_dict = {
        "trend":   pd.Series(np.random.randn(n) * 0.010 + 0.0003, index=dates),  # vol 16%
        "mr":      pd.Series(np.random.randn(n) * 0.007 + 0.0002, index=dates),  # vol 11%
        "vol_bo":  pd.Series(np.random.randn(n) * 0.018 + 0.0005, index=dates),  # vol 28%
    }

    print(f"\nStrategies:")
    df = pd.DataFrame(returns_dict)
    for name in df.columns:
        annual_vol = df[name].std() * np.sqrt(252)
        print(f"  {name:<10}: annualized vol {annual_vol*100:.1f}%")

    # Compare 3 allocations
    print(f"\n{'─' * 70}")
    print("Comparing 3 allocation methods:")
    print(f"{'─' * 70}")

    results = compare_allocations(returns_dict, lookback=500)

    for method, r in results.items():
        print(f"\n{method.upper().replace('_', ' ')}:")
        print(f"  Weights:")
        for name, w in r["weights"].items():
            print(f"    {name:<10}: {w*100:>5.1f}%")
        print(f"  Risk contributions:")
        for name, rc in r["risk_contributions"].items():
            print(f"    {name:<10}: {rc*100:>5.1f}%")
        print(f"  Portfolio vol:     {r['portfolio_vol']*100:.2f}%")
        print(f"  Portfolio Sharpe:  {r['portfolio_sharpe']:.3f}")

    print(f"\nKey insights:")
    print(f"  - Equal capital: simple but volatile strategies dominate risk")
    print(f"  - Equal risk contrib: each strategy contributes equal risk")
    print(f"  - Risk parity: most balanced, but requires scipy")
    print(f"\nFor retail (1-5 strategies): use equal risk contribution")
    print(f"For institutional (5+ strategies): use full risk parity")
