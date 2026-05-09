"""
Multi-strategy portfolio orchestrator.

Combines Trend (Ch7), Mean Reversion (Ch8), and Volatility Breakout (Ch9)
strategies into single equal-weight portfolio with cross-strategy risk management.

QuantCFD Chapter 12 - Capstone deployment
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class StrategyAllocation:
    """Capital allocation for a single strategy."""
    strategy_name: str
    allocation_pct: float          # Percentage of total portfolio (0-1)
    strategy_capital: float        # Dollar capital allocated
    risk_per_trade_pct: float = 0.01  # 1% per trade default
    max_concurrent_positions: int = 3
    is_active: bool = True

    @property
    def max_risk_per_trade(self) -> float:
        return self.strategy_capital * self.risk_per_trade_pct


@dataclass
class PortfolioState:
    """Current state of multi-strategy portfolio."""
    timestamp: datetime
    total_equity: float
    cash_balance: float
    open_positions_count: int
    total_open_risk: float          # Sum of risk across all open positions
    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float
    allocations: Dict[str, StrategyAllocation] = field(default_factory=dict)


class PortfolioOrchestrator:
    """
    Coordinate multi-strategy portfolio with equal-weight allocation.

    Responsibilities:
    - Capital allocation across strategies
    - Cross-strategy risk overlay
    - Correlation monitoring
    - Monthly rebalancing
    - Performance attribution
    """

    def __init__(
        self,
        total_capital: float,
        strategy_names: List[str] = None,
        allocation_method: str = 'equal_weight',
        rebalance_frequency: str = 'monthly',
    ):
        if strategy_names is None:
            strategy_names = ['trend', 'mean_reversion', 'vol_breakout']

        self.total_capital = total_capital
        self.strategy_names = strategy_names
        self.allocation_method = allocation_method
        self.rebalance_frequency = rebalance_frequency

        # Initialize allocations
        self.allocations = self._compute_initial_allocations()

        # Portfolio-level limits
        self.daily_loss_limit_pct = 0.03    # -3% daily portfolio limit
        self.weekly_loss_limit_pct = 0.06   # -6% weekly
        self.monthly_loss_limit_pct = 0.10  # -10% monthly

        # State tracking
        self.position_history: List[Dict] = []
        self.daily_pnl_history: List[float] = []
        self.last_rebalance_date: Optional[datetime] = None

    def _compute_initial_allocations(self) -> Dict[str, StrategyAllocation]:
        """Equal weight allocation by default."""
        n = len(self.strategy_names)
        if self.allocation_method == 'equal_weight':
            pct = 1.0 / n
        else:
            pct = 1.0 / n  # Fallback

        allocations = {}
        for name in self.strategy_names:
            allocations[name] = StrategyAllocation(
                strategy_name=name,
                allocation_pct=pct,
                strategy_capital=self.total_capital * pct,
            )
        return allocations

    def validate_new_trade(
        self,
        strategy_name: str,
        trade_risk: float,
        current_equity: float,
        current_open_risk: float,
    ) -> Dict:
        """
        Validate proposed trade against portfolio-level constraints.

        Returns dict with 'approved' flag and reasoning.
        """
        result = {
            'approved': True,
            'reasons': [],
            'warnings': [],
        }

        # Check 1: Strategy must be active
        if strategy_name not in self.allocations:
            result['approved'] = False
            result['reasons'].append(f'Unknown strategy: {strategy_name}')
            return result

        alloc = self.allocations[strategy_name]
        if not alloc.is_active:
            result['approved'] = False
            result['reasons'].append(f'Strategy {strategy_name} is inactive')
            return result

        # Check 2: Per-trade risk within strategy limit
        if trade_risk > alloc.max_risk_per_trade * 1.05:  # 5% tolerance
            result['approved'] = False
            result['reasons'].append(
                f'Trade risk ${trade_risk:.2f} exceeds strategy limit '
                f'${alloc.max_risk_per_trade:.2f}'
            )
            return result

        # Check 3: Total portfolio open risk
        total_max_risk = current_equity * 0.03  # 3% portfolio max
        new_total_risk = current_open_risk + trade_risk
        if new_total_risk > total_max_risk:
            result['approved'] = False
            result['reasons'].append(
                f'Adding trade would push total risk to ${new_total_risk:.2f}, '
                f'exceeding portfolio limit ${total_max_risk:.2f}'
            )
            return result

        # Warning: approaching limit
        if new_total_risk > total_max_risk * 0.8:
            result['warnings'].append(
                f'Total open risk would reach {new_total_risk/total_max_risk*100:.0f}% '
                f'of portfolio limit'
            )

        return result

    def compute_correlation_matrix(
        self,
        returns_dict: Dict[str, pd.Series],
    ) -> pd.DataFrame:
        """Compute pairwise correlation between strategy returns."""
        df = pd.DataFrame(returns_dict)
        return df.corr()

    def check_correlation_health(
        self,
        corr_matrix: pd.DataFrame,
        warning_threshold: float = 0.5,
        critical_threshold: float = 0.7,
    ) -> Dict:
        """Identify high correlations between strategies."""
        result = {
            'health': 'good',
            'warnings': [],
            'critical': [],
        }

        n = len(corr_matrix)
        for i in range(n):
            for j in range(i + 1, n):
                pair = (corr_matrix.index[i], corr_matrix.columns[j])
                corr_val = corr_matrix.iloc[i, j]

                if abs(corr_val) > critical_threshold:
                    result['critical'].append({
                        'pair': pair,
                        'correlation': corr_val,
                    })
                    result['health'] = 'critical'
                elif abs(corr_val) > warning_threshold:
                    result['warnings'].append({
                        'pair': pair,
                        'correlation': corr_val,
                    })
                    if result['health'] == 'good':
                        result['health'] = 'warning'

        return result

    def should_rebalance(self, current_date: datetime) -> bool:
        """Determine if rebalance is due."""
        if self.last_rebalance_date is None:
            return True

        days_since = (current_date - self.last_rebalance_date).days

        if self.rebalance_frequency == 'monthly':
            return days_since >= 30
        elif self.rebalance_frequency == 'quarterly':
            return days_since >= 90
        elif self.rebalance_frequency == 'weekly':
            return days_since >= 7
        return False

    def execute_rebalance(
        self,
        current_equity: float,
        current_strategy_equities: Dict[str, float],
        current_date: datetime,
    ) -> List[Dict]:
        """
        Compute transfers needed to rebalance to target allocation.

        Returns list of transfer instructions.
        """
        transfers = []
        target_per_strategy = current_equity / len(self.strategy_names)

        for name in self.strategy_names:
            current = current_strategy_equities.get(name, 0)
            transfer = target_per_strategy - current

            if abs(transfer) > 50:  # Skip transfers under $50
                transfers.append({
                    'strategy': name,
                    'current_equity': current,
                    'target_equity': target_per_strategy,
                    'transfer_amount': transfer,
                    'direction': 'in' if transfer > 0 else 'out',
                })

            # Update internal state
            self.allocations[name].strategy_capital = target_per_strategy

        self.total_capital = current_equity
        self.last_rebalance_date = current_date

        return transfers

    def get_portfolio_summary(
        self,
        current_equity: float,
        strategy_pnls: Dict[str, float],
    ) -> Dict:
        """Generate portfolio summary."""
        total_pnl = sum(strategy_pnls.values())

        attribution = {}
        for name, pnl in strategy_pnls.items():
            alloc = self.allocations.get(name)
            if alloc:
                attribution[name] = {
                    'pnl': pnl,
                    'pnl_pct_strategy': pnl / alloc.strategy_capital * 100
                                        if alloc.strategy_capital > 0 else 0,
                    'pnl_pct_portfolio': pnl / current_equity * 100
                                         if current_equity > 0 else 0,
                    'allocation_pct': alloc.allocation_pct * 100,
                    'is_active': alloc.is_active,
                }

        return {
            'total_equity': current_equity,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl / current_equity * 100 if current_equity > 0 else 0,
            'strategy_attribution': attribution,
            'last_rebalance': self.last_rebalance_date,
        }


def demo():
    """Demo portfolio orchestration with synthetic strategy returns."""
    print("=" * 60)
    print("DEMO: Multi-strategy portfolio orchestrator")
    print("=" * 60)

    np.random.seed(42)

    # Initialize portfolio with $30k across 3 strategies
    portfolio = PortfolioOrchestrator(
        total_capital=30000,
        strategy_names=['trend_xau', 'mr_eur', 'vol_bo_btc'],
    )

    print("\n1. Initial allocations:")
    for name, alloc in portfolio.allocations.items():
        print(f"   {name:15s} ${alloc.strategy_capital:>10,.2f} "
              f"({alloc.allocation_pct*100:.1f}%)")
    print(f"   Total: ${portfolio.total_capital:,.2f}")

    # Test trade validation
    print("\n2. Trade validation tests:")

    # Acceptable trade
    result = portfolio.validate_new_trade(
        strategy_name='trend_xau',
        trade_risk=80,
        current_equity=30000,
        current_open_risk=200,
    )
    print(f"   Trade $80 risk: {'APPROVED' if result['approved'] else 'REJECTED'}")
    if result['reasons']:
        print(f"     Reasons: {result['reasons']}")

    # Too much risk
    result = portfolio.validate_new_trade(
        strategy_name='trend_xau',
        trade_risk=200,
        current_equity=30000,
        current_open_risk=200,
    )
    print(f"   Trade $200 risk: {'APPROVED' if result['approved'] else 'REJECTED'}")
    if result['reasons']:
        print(f"     Reasons: {result['reasons']}")

    # Compute correlation matrix from synthetic returns
    print("\n3. Correlation matrix (synthetic):")
    n_days = 252  # 1 year
    returns = {
        'trend_xau': np.random.normal(0.0008, 0.012, n_days),
        'mr_eur': np.random.normal(0.0005, 0.008, n_days),
        'vol_bo_btc': np.random.normal(0.0010, 0.020, n_days),
    }
    returns_dict = {k: pd.Series(v) for k, v in returns.items()}

    corr_matrix = portfolio.compute_correlation_matrix(returns_dict)
    print(corr_matrix.round(3))

    # Check correlation health
    health = portfolio.check_correlation_health(corr_matrix)
    print(f"\n   Correlation health: {health['health'].upper()}")

    # Performance attribution
    print("\n4. Performance attribution:")
    strategy_pnls = {
        'trend_xau': 850,
        'mr_eur': 420,
        'vol_bo_btc': -180,
    }
    summary = portfolio.get_portfolio_summary(
        current_equity=31090,
        strategy_pnls=strategy_pnls,
    )
    print(f"   Total equity: ${summary['total_equity']:,.2f}")
    print(f"   Total P&L: ${summary['total_pnl']:,.2f} "
          f"({summary['total_pnl_pct']:+.2f}%)")
    print("\n   Per-strategy:")
    for name, attr in summary['strategy_attribution'].items():
        print(f"     {name:15s} ${attr['pnl']:>+8,.2f}  "
              f"({attr['pnl_pct_strategy']:+.2f}% strategy)")

    # Test rebalancing
    print("\n5. Rebalancing simulation:")
    current_strategy_equities = {
        'trend_xau': 10850,
        'mr_eur': 10420,
        'vol_bo_btc': 9820,
    }
    transfers = portfolio.execute_rebalance(
        current_equity=31090,
        current_strategy_equities=current_strategy_equities,
        current_date=datetime.now(),
    )
    print(f"   Transfers needed: {len(transfers)}")
    for t in transfers:
        print(f"     {t['strategy']:15s} {t['direction']:>4s} "
              f"${abs(t['transfer_amount']):>8,.2f}")

    print("\n" + "=" * 60)
    print("Demo complete. Portfolio orchestration validated.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
