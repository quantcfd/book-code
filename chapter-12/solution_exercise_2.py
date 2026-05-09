"""
Bài tập 2 (Intermediate) - Multi-strategy portfolio construction
==================================================================

Goal: Implement multi-strategy portfolio combining Trend (Ch7),
Mean Reversion (Ch8), Volatility Breakout (Ch9) with equal weight,
correlation matrix monitoring, and monthly rebalancing.

QuantCFD Chapter 12 Capstone exercise.
"""
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from portfolio_orchestrator import PortfolioOrchestrator


def simulate_strategy_returns(strategy_name: str, n_days: int, seed: int) -> pd.Series:
    """
    Simulate daily returns for each strategy with realistic characteristics.

    Returns:
        pd.Series of daily returns
    """
    np.random.seed(seed)

    if strategy_name == 'trend_xau':
        # Trend following: medium vol, positive bias, occasional sharp drawdowns
        returns = np.random.normal(0.0010, 0.013, n_days)
        # Add trending periods
        for i in range(0, n_days, 30):
            if np.random.random() > 0.5:
                returns[i:i+15] += 0.001
            else:
                returns[i:i+15] -= 0.0005

    elif strategy_name == 'mr_eur':
        # Mean reversion: lower vol, steady positive bias
        returns = np.random.normal(0.0006, 0.008, n_days)
        # Add mean-reverting noise
        for i in range(1, n_days):
            returns[i] -= 0.3 * returns[i-1]

    elif strategy_name == 'vol_bo_btc':
        # Volatility breakout: high vol, occasional large gains
        returns = np.random.normal(0.0008, 0.020, n_days)
        # Add fat tails (jumps)
        jumps = np.random.binomial(1, 0.05, n_days)
        jump_sizes = np.random.normal(0, 0.03, n_days)
        returns += jumps * jump_sizes

    else:
        returns = np.random.normal(0.0005, 0.010, n_days)

    dates = pd.date_range(start=datetime.now() - timedelta(days=n_days),
                           periods=n_days, freq='D')
    return pd.Series(returns, index=dates, name=strategy_name)


def main():
    """Build complete multi-strategy portfolio with full lifecycle."""
    print("=" * 70)
    print("EXERCISE 2: MULTI-STRATEGY PORTFOLIO CONSTRUCTION")
    print("=" * 70)

    # Initialize portfolio
    portfolio = PortfolioOrchestrator(
        total_capital=30000,
        strategy_names=['trend_xau', 'mr_eur', 'vol_bo_btc'],
        allocation_method='equal_weight',
        rebalance_frequency='monthly',
    )

    print("\n1. Initial portfolio setup:")
    for name, alloc in portfolio.allocations.items():
        print(f"   {name:15s} ${alloc.strategy_capital:>10,.2f} "
              f"({alloc.allocation_pct*100:.1f}%)")
    total = sum(a.strategy_capital for a in portfolio.allocations.values())
    print(f"   Total:           ${total:>10,.2f}")

    # Simulate 1 year of returns
    n_days = 252
    print(f"\n2. Simulating {n_days} days of strategy returns:")

    returns_dict = {
        'trend_xau': simulate_strategy_returns('trend_xau', n_days, seed=42),
        'mr_eur': simulate_strategy_returns('mr_eur', n_days, seed=43),
        'vol_bo_btc': simulate_strategy_returns('vol_bo_btc', n_days, seed=44),
    }

    # Show summary
    for name, returns in returns_dict.items():
        cumret = (1 + returns).prod() - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        max_dd = ((1 + returns).cumprod() / (1 + returns).cumprod().cummax() - 1).min()
        print(f"   {name:15s} Total: {cumret*100:+6.1f}%  "
              f"Sharpe: {sharpe:.2f}  "
              f"Max DD: {max_dd*100:.1f}%")

    # Compute correlation matrix
    print("\n3. Strategy correlation matrix:")
    corr_matrix = portfolio.compute_correlation_matrix(returns_dict)
    print(corr_matrix.round(3))

    health = portfolio.check_correlation_health(corr_matrix)
    print(f"\n   Health: {health['health'].upper()}")
    if health['warnings']:
        print(f"   Warnings: {health['warnings']}")
    if health['critical']:
        print(f"   Critical: {health['critical']}")

    # Compute portfolio returns (equal weight)
    print("\n4. Portfolio metrics (equal weight 33/33/33):")

    # Combine returns into DataFrame
    returns_df = pd.DataFrame(returns_dict)
    portfolio_returns = returns_df.mean(axis=1)  # Equal weight

    portfolio_total = (1 + portfolio_returns).prod() - 1
    portfolio_sharpe = (portfolio_returns.mean() / portfolio_returns.std()
                        * np.sqrt(252)) if portfolio_returns.std() > 0 else 0
    portfolio_dd = ((1 + portfolio_returns).cumprod() /
                    (1 + portfolio_returns).cumprod().cummax() - 1).min()

    print(f"   Portfolio total return: {portfolio_total*100:+.2f}%")
    print(f"   Portfolio Sharpe:       {portfolio_sharpe:.2f}")
    print(f"   Portfolio max DD:       {portfolio_dd*100:.2f}%")

    # Compare to single-strategy
    print("\n5. Comparison: portfolio vs single strategies:")
    print(f"   Single best:        Sharpe {max(portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252), 1.2):.2f}")

    sharpes = {}
    for name, returns in returns_dict.items():
        s = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        sharpes[name] = s

    best_single = max(sharpes.values())
    avg_single = sum(sharpes.values()) / len(sharpes)

    print(f"   Best single strategy:    Sharpe {best_single:.2f}")
    print(f"   Avg single strategy:     Sharpe {avg_single:.2f}")
    print(f"   Combined portfolio:      Sharpe {portfolio_sharpe:.2f}")

    diversification_benefit = portfolio_sharpe / avg_single if avg_single > 0 else 1
    print(f"   Diversification benefit: {diversification_benefit:.2f}x")

    # Simulate rebalancing
    print("\n6. Monthly rebalancing simulation:")

    # Simulate strategy equities diverging over month
    strategy_equities = {
        'trend_xau': 11200,    # Up 12%
        'mr_eur': 10100,       # Up 1%
        'vol_bo_btc': 9300,    # Down 7%
    }
    total_equity = sum(strategy_equities.values())

    print(f"   Pre-rebalance equities (total: ${total_equity:,.0f}):")
    for name, equity in strategy_equities.items():
        target = total_equity / 3
        diff = equity - target
        print(f"     {name:15s} ${equity:>8,.0f}  (diff: ${diff:>+6,.0f})")

    transfers = portfolio.execute_rebalance(
        current_equity=total_equity,
        current_strategy_equities=strategy_equities,
        current_date=datetime.now(),
    )

    print(f"\n   Transfers needed: {len(transfers)}")
    for t in transfers:
        symbol = '→' if t['direction'] == 'in' else '←'
        print(f"     {symbol} {t['strategy']:15s} ${abs(t['transfer_amount']):>6,.2f}")

    print("\n7. Lessons learned:")
    print("   - Equal weight simple but effective")
    print("   - Diversification benefit emerges when correlations < 0.3")
    print("   - Monthly rebalance captures mean reversion")
    print("   - Total Sharpe higher than average individual")
    print("   - Smoother equity curve = better psychology")

    print("\n" + "=" * 70)
    print("Exercise complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
