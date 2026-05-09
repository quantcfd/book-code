"""
Performance attribution per strategy.

Decompose portfolio P&L by strategy, compute rolling Sharpe,
identify under/over-performing strategies.

QuantCFD Chapter 12 - Capstone deployment
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class StrategyPerformance:
    """Performance summary for single strategy."""
    strategy_name: str
    period_start: datetime
    period_end: datetime
    starting_capital: float
    ending_capital: float
    total_pnl: float
    total_return_pct: float
    n_trades: int
    n_winning: int
    n_losing: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe: float
    max_drawdown: float
    days_in_period: int


class PerformanceAttribution:
    """
    Compute per-strategy performance attribution.

    Tracks daily returns per strategy, computes rolling metrics,
    identifies best/worst contributors.
    """

    def __init__(self):
        # daily_returns[strategy_name] = pd.Series of daily returns
        self.daily_returns: Dict[str, pd.Series] = {}
        # trades log per strategy
        self.trades: Dict[str, List[Dict]] = {}

    def record_daily_return(
        self,
        strategy_name: str,
        date: datetime,
        return_pct: float,
        capital: float = None,
    ):
        """Record daily return for a strategy."""
        if strategy_name not in self.daily_returns:
            self.daily_returns[strategy_name] = pd.Series(dtype=float)

        ts = pd.Timestamp(date.date())
        self.daily_returns[strategy_name].loc[ts] = return_pct

    def record_trade(
        self,
        strategy_name: str,
        trade_data: Dict,
    ):
        """Record completed trade for attribution."""
        if strategy_name not in self.trades:
            self.trades[strategy_name] = []
        self.trades[strategy_name].append(trade_data)

    def compute_rolling_sharpe(
        self,
        strategy_name: str,
        window_days: int = 90,
    ) -> Optional[pd.Series]:
        """Compute rolling Sharpe ratio for strategy."""
        if strategy_name not in self.daily_returns:
            return None

        returns = self.daily_returns[strategy_name].sort_index()
        if len(returns) < window_days:
            return None

        rolling_mean = returns.rolling(window=window_days).mean()
        rolling_std = returns.rolling(window=window_days).std()

        # Annualized Sharpe
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

        return rolling_sharpe

    def compute_strategy_performance(
        self,
        strategy_name: str,
        starting_capital: float,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> Optional[StrategyPerformance]:
        """Compute comprehensive performance metrics for a strategy."""
        if strategy_name not in self.daily_returns:
            return None

        returns = self.daily_returns[strategy_name].sort_index()

        # Filter by period
        if period_start:
            returns = returns[returns.index >= pd.Timestamp(period_start.date())]
        if period_end:
            returns = returns[returns.index <= pd.Timestamp(period_end.date())]

        if len(returns) == 0:
            return None

        # Compute metrics
        total_return = (1 + returns).prod() - 1
        ending_capital = starting_capital * (1 + total_return)
        total_pnl = ending_capital - starting_capital

        # Sharpe (annualized)
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0

        # Max DD
        cum = (1 + returns).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max
        max_dd = drawdown.min()

        # Trade-level stats
        trades_list = self.trades.get(strategy_name, [])
        n_trades = len(trades_list)
        wins = [t['pnl'] for t in trades_list if t.get('pnl', 0) > 0]
        losses = [t['pnl'] for t in trades_list if t.get('pnl', 0) < 0]
        n_winning = len(wins)
        n_losing = len(losses)
        win_rate = n_winning / n_trades if n_trades > 0 else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return StrategyPerformance(
            strategy_name=strategy_name,
            period_start=returns.index[0].to_pydatetime(),
            period_end=returns.index[-1].to_pydatetime(),
            starting_capital=starting_capital,
            ending_capital=ending_capital,
            total_pnl=total_pnl,
            total_return_pct=total_return * 100,
            n_trades=n_trades,
            n_winning=n_winning,
            n_losing=n_losing,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe=sharpe,
            max_drawdown=max_dd,
            days_in_period=len(returns),
        )

    def compute_portfolio_attribution(
        self,
        strategy_capitals: Dict[str, float],
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> Dict:
        """Compute portfolio-level attribution across strategies."""
        attributions = {}
        total_pnl = 0
        total_capital = sum(strategy_capitals.values())

        for strat_name, capital in strategy_capitals.items():
            perf = self.compute_strategy_performance(
                strategy_name=strat_name,
                starting_capital=capital,
                period_start=period_start,
                period_end=period_end,
            )
            if perf:
                attributions[strat_name] = {
                    'pnl': perf.total_pnl,
                    'return_pct': perf.total_return_pct,
                    'sharpe': perf.sharpe,
                    'max_dd': perf.max_drawdown,
                    'n_trades': perf.n_trades,
                    'win_rate': perf.win_rate,
                    'allocation_pct': capital / total_capital * 100
                                       if total_capital > 0 else 0,
                    'pnl_contribution_pct': (perf.total_pnl / total_capital * 100
                                              if total_capital > 0 else 0),
                }
                total_pnl += perf.total_pnl

        # Portfolio-level metrics
        portfolio_return = total_pnl / total_capital if total_capital > 0 else 0

        return {
            'period_start': period_start,
            'period_end': period_end,
            'total_capital': total_capital,
            'total_pnl': total_pnl,
            'portfolio_return_pct': portfolio_return * 100,
            'strategy_attributions': attributions,
        }

    def identify_best_worst(
        self,
        period_days: int = 30,
    ) -> Tuple[List[str], List[str]]:
        """Identify best and worst performing strategies recently."""
        strategy_returns = {}

        cutoff = datetime.now() - timedelta(days=period_days)
        cutoff_ts = pd.Timestamp(cutoff.date())

        for strat_name, returns_series in self.daily_returns.items():
            recent = returns_series[returns_series.index >= cutoff_ts]
            if len(recent) > 0:
                cum_return = (1 + recent).prod() - 1
                strategy_returns[strat_name] = cum_return

        # Sort
        sorted_strats = sorted(strategy_returns.items(), key=lambda x: x[1], reverse=True)
        best = [name for name, _ in sorted_strats[:3]]
        worst = [name for name, _ in sorted_strats[-3:]]

        return best, worst

    def generate_attribution_report(
        self,
        strategy_capitals: Dict[str, float],
    ) -> str:
        """Generate text attribution report."""
        attribution = self.compute_portfolio_attribution(strategy_capitals)

        lines = []
        lines.append("=" * 70)
        lines.append("PERFORMANCE ATTRIBUTION REPORT")
        lines.append("=" * 70)
        lines.append(f"Total capital:    ${attribution['total_capital']:,.2f}")
        lines.append(f"Portfolio P&L:    ${attribution['total_pnl']:,.2f}")
        lines.append(f"Portfolio return: {attribution['portfolio_return_pct']:+.2f}%")
        lines.append("")

        lines.append("STRATEGY ATTRIBUTION:")
        lines.append("-" * 70)
        lines.append(f"{'Strategy':<25} {'PnL':>10} {'Return%':>10} "
                     f"{'Sharpe':>8} {'DD':>7} {'Trades':>7}")
        lines.append("-" * 70)

        for name, attr in attribution['strategy_attributions'].items():
            lines.append(
                f"{name:<25} "
                f"${attr['pnl']:>+9,.0f} "
                f"{attr['return_pct']:>+9.2f}% "
                f"{attr['sharpe']:>8.2f} "
                f"{attr['max_dd']*100:>6.1f}% "
                f"{attr['n_trades']:>7}"
            )

        lines.append("=" * 70)
        return "\n".join(lines)


def demo():
    """Demo performance attribution."""
    print("=" * 70)
    print("DEMO: Performance attribution")
    print("=" * 70)

    np.random.seed(42)

    attr = PerformanceAttribution()

    # Simulate 90 days of returns for 3 strategies
    n_days = 90
    base_date = datetime.now() - timedelta(days=n_days)

    strategies = {
        'trend_xau': {'mean': 0.0008, 'std': 0.012, 'capital': 10000},
        'mr_eur': {'mean': 0.0005, 'std': 0.008, 'capital': 10000},
        'vol_bo_btc': {'mean': -0.0003, 'std': 0.015, 'capital': 10000},  # underperforming
    }

    # Generate daily returns
    print("\n1. Recording 90 days of returns:")
    for strat_name, config in strategies.items():
        returns = np.random.normal(config['mean'], config['std'], n_days)
        for i, ret in enumerate(returns):
            attr.record_daily_return(
                strategy_name=strat_name,
                date=base_date + timedelta(days=i),
                return_pct=ret,
            )

    # Generate trade records
    for strat_name in strategies:
        n_trades = np.random.randint(20, 40)
        for _ in range(n_trades):
            pnl = np.random.normal(50 if 'underperforming' not in strat_name else -10, 100)
            attr.record_trade(strat_name, {
                'pnl': pnl,
                'date': base_date + timedelta(days=np.random.randint(0, n_days)),
            })

    # Generate report
    print("\n2. Attribution report:")
    capitals = {name: config['capital'] for name, config in strategies.items()}
    print(attr.generate_attribution_report(capitals))

    # Best/worst identification
    print("\n3. Best/worst performers (last 30 days):")
    best, worst = attr.identify_best_worst(period_days=30)
    print(f"   Best:  {', '.join(best)}")
    print(f"   Worst: {', '.join(worst)}")

    # Rolling Sharpe
    print("\n4. Rolling Sharpe (90-day) for each strategy:")
    for strat_name in strategies:
        rolling = attr.compute_rolling_sharpe(strat_name, window_days=30)
        if rolling is not None and len(rolling.dropna()) > 0:
            print(f"   {strat_name:15s} latest Sharpe: {rolling.iloc[-1]:.2f}")

    print("\n" + "=" * 70)
    print("Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    demo()
