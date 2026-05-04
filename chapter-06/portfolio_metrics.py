"""
QuantCFD Chương 6 — portfolio_metrics.py
Multi-strategy portfolio analytics: correlation, MCR, risk parity, concentration.

Usage:
    from portfolio_metrics import strategy_correlation_matrix, risk_parity_weights, ...

    returns_dict = {'xau': xau_returns, 'eur': eur_returns, 'btc': btc_returns}
    corr = strategy_correlation_matrix(returns_dict)
    rp_weights = risk_parity_weights(returns_dict)
"""
import numpy as np
import pandas as pd


def strategy_correlation_matrix(strategy_returns_dict: dict) -> pd.DataFrame:
    """Pearson correlation giữa N strategies."""
    df = pd.DataFrame(strategy_returns_dict).dropna()
    return df.corr()


def correlation_summary(corr_matrix: pd.DataFrame) -> dict:
    n = corr_matrix.shape[0]
    if n < 2:
        return {}
    upper = corr_matrix.values[np.triu_indices(n, k=1)]
    return {
        'n_pairs':       len(upper),
        'mean_corr':     upper.mean(),
        'max_corr':      upper.max(),
        'min_corr':      upper.min(),
        'pct_high_corr': (upper > 0.5).mean() * 100,
        'pct_neg_corr':  (upper < 0).mean() * 100,
    }


def portfolio_diversification_test(
    strategy_returns_dict: dict,
    weights: dict = None,
    periods_per_year: int = 252,
) -> dict:
    """Test portfolio Sharpe vs avg individual Sharpe."""
    df = pd.DataFrame(strategy_returns_dict).dropna()
    if weights is None:
        weights = {col: 1.0 / df.shape[1] for col in df.columns}

    individual_sharpes = {}
    for col in df.columns:
        s = (df[col].mean() / df[col].std() * np.sqrt(periods_per_year)
             if df[col].std() > 0 else 0)
        individual_sharpes[col] = s

    portfolio_returns = sum(df[col] * weights[col] for col in df.columns)
    portfolio_sharpe = (portfolio_returns.mean() / portfolio_returns.std()
                        * np.sqrt(periods_per_year)
                        if portfolio_returns.std() > 0 else 0)

    avg_sharpe = np.mean(list(individual_sharpes.values()))
    diversification_lift = (portfolio_sharpe / avg_sharpe - 1) * 100 if avg_sharpe != 0 else 0

    return {
        'individual_sharpes':   individual_sharpes,
        'average_individual':   avg_sharpe,
        'portfolio_sharpe':     portfolio_sharpe,
        'diversification_lift': diversification_lift,
        'portfolio_returns':    portfolio_returns,
    }


def marginal_contribution_to_risk(
    strategy_returns_dict: dict,
    weights: dict,
) -> dict:
    """MCR_i = w_i × (Σ × w)_i / σ_portfolio."""
    df = pd.DataFrame(strategy_returns_dict).dropna()
    cov = df.cov()
    w = pd.Series(weights)
    w = w / w.sum()

    portfolio_var = w @ cov @ w
    portfolio_vol = np.sqrt(portfolio_var)
    contributions = {}
    for name in df.columns:
        marginal = (cov @ w)[name]
        mcr = w[name] * marginal / portfolio_vol if portfolio_vol > 0 else 0
        contributions[name] = {
            'weight':           w[name],
            'marginal_var':     marginal,
            'contribution':     mcr,
            'pct_contribution': mcr / portfolio_vol * 100 if portfolio_vol > 0 else 0,
        }
    return contributions


def risk_parity_weights(strategy_returns_dict: dict) -> dict:
    """Naive risk parity: w_i ∝ 1/vol_i."""
    df = pd.DataFrame(strategy_returns_dict).dropna()
    inv_vols = 1 / df.std()
    weights = inv_vols / inv_vols.sum()
    return weights.to_dict()


def herfindahl_index(weights: dict) -> dict:
    """HHI = Σ w_i^2. Effective N = 1/HHI."""
    w = np.array(list(weights.values()))
    hhi = (w ** 2).sum()
    n = len(w)
    return {
        'hhi':           hhi,
        'effective_n':   1 / hhi if hhi > 0 else 0,
        'n_strategies':  n,
        'concentration': hhi * n,
    }


def leverage_metrics(positions: pd.DataFrame, equity: pd.Series) -> dict:
    """positions: DataFrame mỗi cột là position USD của 1 strategy."""
    gross_exposure = positions.abs().sum(axis=1)
    net_exposure = positions.sum(axis=1)
    leverage = gross_exposure / equity

    return {
        'max_leverage':     leverage.max(),
        'avg_leverage':     leverage.mean(),
        'median_leverage':  leverage.median(),
        'pct_above_2x':     (leverage > 2).mean() * 100,
        'pct_above_5x':     (leverage > 5).mean() * 100,
        'max_net_exposure': net_exposure.max() / equity.max() if equity.max() > 0 else 0,
    }


if __name__ == '__main__':
    np.random.seed(7)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')

    strats = {
        'xau_trend':   pd.Series(np.random.normal(0.0006, 0.015, 1000), index=dates),
        'eur_meanrev': pd.Series(np.random.normal(0.0004, 0.008, 1000), index=dates),
        'btc_break':   pd.Series(np.random.normal(0.0008, 0.025, 1000), index=dates),
        'us500_rsi':   pd.Series(np.random.normal(0.0005, 0.012, 1000), index=dates),
    }

    print("=== Correlation Matrix ===")
    corr = strategy_correlation_matrix(strats)
    print(corr.round(3))

    print("\n=== Correlation Summary ===")
    cs = correlation_summary(corr)
    for k, v in cs.items():
        print(f"  {k:20s} {v:.3f}")

    print("\n=== Equal Weight Portfolio ===")
    eq = portfolio_diversification_test(strats)
    print(f"  Avg individual Sharpe: {eq['average_individual']:.2f}")
    print(f"  Portfolio Sharpe:      {eq['portfolio_sharpe']:.2f}")
    print(f"  Diversification lift:  {eq['diversification_lift']:+.1f}%")

    print("\n=== Risk Parity Weights ===")
    rp = risk_parity_weights(strats)
    for name, w in rp.items():
        print(f"  {name:15s} {w*100:.1f}%")

    rp_test = portfolio_diversification_test(strats, weights=rp)
    print(f"\n  RP Portfolio Sharpe:   {rp_test['portfolio_sharpe']:.2f}")
    print(f"  RP Diversification:    {rp_test['diversification_lift']:+.1f}%")

    print("\n=== MCR (equal weight) ===")
    eq_weights = {k: 0.25 for k in strats.keys()}
    mcr = marginal_contribution_to_risk(strats, eq_weights)
    for name, info in mcr.items():
        print(f"  {name:15s} weight {info['weight']*100:.0f}%  "
              f"contribution {info['pct_contribution']:.1f}%")

    print("\n=== Herfindahl (equal vs RP) ===")
    print(f"  Equal:        {herfindahl_index(eq_weights)}")
    print(f"  Risk parity:  {herfindahl_index(rp)}")
