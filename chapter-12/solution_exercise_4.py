"""
Bài tập 4 (Advanced) - Performance attribution
================================================

Goal: Implement multi-strategy performance attribution với
per-strategy P&L decomposition, rolling Sharpe analysis,
and allocation effectiveness assessment.

Simulates 6 months of trading data across 3 strategies.

QuantCFD Chapter 12 Capstone exercise.
"""
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from performance_attribution import PerformanceAttribution


def main():
    """Run comprehensive 6-month performance attribution analysis."""
    print("=" * 70)
    print("EXERCISE 4: PERFORMANCE ATTRIBUTION")
    print("=" * 70)

    np.random.seed(42)

    attr = PerformanceAttribution()

    # Strategy capital allocations
    capitals = {
        'trend_xau': 10000,
        'mr_eur': 10000,
        'vol_bo_btc': 10000,
    }

    # Simulate 180 days (6 months) with realistic strategy characteristics
    n_days = 180
    base_date = datetime.now() - timedelta(days=n_days)

    # Strategy parameters (mean daily return, std, trade frequency)
    strategies = {
        'trend_xau': {'mean': 0.0009, 'std': 0.012, 'trades_per_month': 22},
        'mr_eur': {'mean': 0.0006, 'std': 0.008, 'trades_per_month': 31},
        'vol_bo_btc': {'mean': -0.0001, 'std': 0.018, 'trades_per_month': 18},
    }

    print(f"\n1. Recording {n_days} days of returns + trades:")

    for strat_name, params in strategies.items():
        # Generate daily returns
        returns = np.random.normal(params['mean'], params['std'], n_days)
        # Add some serial correlation for realism
        for i in range(1, n_days):
            returns[i] += 0.1 * returns[i-1]

        for i, ret in enumerate(returns):
            attr.record_daily_return(
                strategy_name=strat_name,
                date=base_date + timedelta(days=i),
                return_pct=ret,
            )

        # Generate trade records (matched to returns)
        n_trades = int(params['trades_per_month'] * 6)  # 6 months
        for _ in range(n_trades):
            # Win rate ~50%
            is_winner = np.random.random() > 0.5
            if is_winner:
                pnl = np.random.uniform(40, 200)
            else:
                pnl = -np.random.uniform(30, 150)

            attr.record_trade(strat_name, {
                'pnl': pnl,
                'date': base_date + timedelta(days=np.random.randint(0, n_days)),
            })

        print(f"   {strat_name:15s} {n_trades} trades recorded")

    # Generate full attribution report
    print("\n2. Full attribution report (6-month period):")
    print(attr.generate_attribution_report(capitals))

    # Per-strategy detailed analysis
    print("\n\n3. Per-strategy detailed breakdown:")
    for strat_name in capitals:
        perf = attr.compute_strategy_performance(
            strategy_name=strat_name,
            starting_capital=capitals[strat_name],
        )
        if perf:
            print(f"\n   {strat_name}:")
            print(f"     Period:         {perf.period_start.date()} to {perf.period_end.date()}")
            print(f"     Days:           {perf.days_in_period}")
            print(f"     Starting cap:   ${perf.starting_capital:,.2f}")
            print(f"     Ending cap:     ${perf.ending_capital:,.2f}")
            print(f"     Total P&L:      ${perf.total_pnl:+,.2f}")
            print(f"     Total return:   {perf.total_return_pct:+.2f}%")
            print(f"     Sharpe:         {perf.sharpe:.2f}")
            print(f"     Max DD:         {perf.max_drawdown*100:.2f}%")
            print(f"     Trades:         {perf.n_trades} ({perf.n_winning} W, {perf.n_losing} L)")
            print(f"     Win rate:       {perf.win_rate*100:.1f}%")
            print(f"     Avg win:        ${perf.avg_win:+.2f}")
            print(f"     Avg loss:       ${perf.avg_loss:+.2f}")
            print(f"     Profit factor:  {perf.profit_factor:.2f}")

    # Rolling Sharpe analysis
    print("\n\n4. Rolling Sharpe analysis (60-day window):")
    print("   Strategy           Latest    Min       Max       Mean")
    print("   " + "-" * 60)
    for strat_name in capitals:
        rolling = attr.compute_rolling_sharpe(strat_name, window_days=60)
        if rolling is not None and len(rolling.dropna()) > 0:
            valid = rolling.dropna()
            print(f"   {strat_name:18s} "
                  f"{valid.iloc[-1]:>+6.2f}    "
                  f"{valid.min():>+6.2f}    "
                  f"{valid.max():>+6.2f}    "
                  f"{valid.mean():>+6.2f}")

    # Best/worst identification
    print("\n\n5. Best/worst performers (last 30 days):")
    best, worst = attr.identify_best_worst(period_days=30)
    print(f"   Best 30-day:  {best[0] if best else 'N/A'}")
    print(f"   Worst 30-day: {worst[0] if worst else 'N/A'}")

    # Allocation effectiveness analysis
    print("\n\n6. Allocation effectiveness:")
    portfolio_attr = attr.compute_portfolio_attribution(capitals)
    total_capital = portfolio_attr['total_capital']

    print(f"   Total capital:   ${total_capital:,.2f}")
    print(f"   Total P&L:       ${portfolio_attr['total_pnl']:+,.2f}")
    print(f"   Portfolio return: {portfolio_attr['portfolio_return_pct']:+.2f}%")

    print("\n   Contribution analysis:")
    sorted_strats = sorted(
        portfolio_attr['strategy_attributions'].items(),
        key=lambda x: x[1]['pnl'],
        reverse=True,
    )
    for name, attr_data in sorted_strats:
        contribution = attr_data['pnl'] / portfolio_attr['total_pnl'] * 100 \
                       if portfolio_attr['total_pnl'] != 0 else 0
        print(f"     {name:15s} ${attr_data['pnl']:>+8,.0f} "
              f"({contribution:+5.1f}% of portfolio P&L)  "
              f"Sharpe {attr_data['sharpe']:.2f}")

    # Recommendations
    print("\n\n7. Strategic recommendations:")

    # Identify underperformers
    underperformers = []
    for name, perf_data in portfolio_attr['strategy_attributions'].items():
        if perf_data['sharpe'] < 0.5:
            underperformers.append(name)

    if underperformers:
        print(f"   ⚠️  Underperforming strategies: {', '.join(underperformers)}")
        print("       Action items:")
        print("       - Review strategy logic and parameters")
        print("       - Check for market regime changes")
        print("       - Run extended backtest with recent data")
        print("       - Consider reducing allocation 50%")
        print("       - If decay confirmed, retire strategy")

    # Identify outperformers
    outperformers = []
    for name, perf_data in portfolio_attr['strategy_attributions'].items():
        if perf_data['sharpe'] > 1.5:
            outperformers.append(name)

    if outperformers:
        print(f"\n   ✅ Outperforming strategies: {', '.join(outperformers)}")
        print("       Action items:")
        print("       - Verify performance is statistically significant")
        print("       - Consider whether to increase allocation")
        print("       - Watch for mean reversion (regression to mean)")
        print("       - Don't chase recent winners (recency bias)")

    print("\n   General recommendations:")
    print("   - Equal weight rebalancing recommended monthly")
    print("   - Review correlation matrix monthly")
    print("   - Don't make allocation changes more often than quarterly")
    print("   - Document allocation decisions với reasoning")

    print("\n" + "=" * 70)
    print("Performance attribution exercise complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
