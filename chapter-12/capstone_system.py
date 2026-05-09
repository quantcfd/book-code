"""
Capstone integrated trading system.

Main entry point that combines all subsystems from Ch7-12 into
production-ready integrated trading platform.

QuantCFD Chapter 12 - Capstone deployment
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from portfolio_orchestrator import PortfolioOrchestrator
from pre_launch_checklist import PreLaunchChecklist
from deployment_phases import PhasedDeploymentManager, DeploymentPhase
from live_monitoring import LiveMonitor, AlertPriority, AlertCategory
from performance_attribution import PerformanceAttribution
from scaling_engine import ScalingEngine, ScalingCriteria


class SystemState(Enum):
    INITIALIZING = "initializing"
    PRE_LAUNCH = "pre_launch"
    PHASE_VALIDATION = "phase_validation"
    LIVE_TRADING = "live_trading"
    DRAWDOWN_RECOVERY = "drawdown_recovery"
    HALTED = "halted"
    SHUTDOWN = "shutdown"


@dataclass
class SystemConfig:
    """Configuration for capstone system."""
    trader_name: str
    starting_capital: float
    strategy_names: List[str] = field(default_factory=lambda: [
        'trend_xau', 'mr_eur', 'vol_bo_btc'
    ])
    risk_per_trade_pct: float = 0.01
    daily_loss_limit_pct: float = 0.03
    weekly_loss_limit_pct: float = 0.06
    monthly_loss_limit_pct: float = 0.10
    discord_webhook: Optional[str] = None
    telegram_token: Optional[str] = None


class CapstoneSystem:
    """
    Integrated trading system combining all chapters.

    State machine:
    initializing → pre_launch (checklist) → phase_validation → live_trading
                                                              ↓
                                                       drawdown_recovery
                                                              ↓
                                                          live_trading

    Components:
    - PortfolioOrchestrator: multi-strategy capital allocation
    - PreLaunchChecklist: 78-item validation
    - PhasedDeploymentManager: phase progression tracking
    - LiveMonitor: real-time monitoring + alerts
    - PerformanceAttribution: per-strategy P&L decomposition
    - ScalingEngine: capital scaling decisions
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.state = SystemState.INITIALIZING

        # Initialize subsystems
        self.portfolio = PortfolioOrchestrator(
            total_capital=config.starting_capital,
            strategy_names=config.strategy_names,
        )
        self.checklist = PreLaunchChecklist()
        self.phase_manager = PhasedDeploymentManager()
        self.monitor = LiveMonitor(
            discord_webhook_url=config.discord_webhook,
            telegram_bot_token=config.telegram_token,
        )
        self.attribution = PerformanceAttribution()
        self.scaling = ScalingEngine(current_capital=config.starting_capital)

        # System state
        self.start_date = datetime.now()
        self.last_state_change = datetime.now()
        self.state_history: List[Dict] = []

    def transition_state(self, new_state: SystemState, reason: str = ""):
        """Change system state with logging."""
        old_state = self.state
        self.state_history.append({
            'timestamp': datetime.now(),
            'from': old_state,
            'to': new_state,
            'reason': reason,
        })
        self.state = new_state
        self.last_state_change = datetime.now()

        self.monitor.create_alert(
            priority=AlertPriority.MEDIUM,
            category=AlertCategory.INFORMATIONAL,
            title=f"State change: {old_state.value} → {new_state.value}",
            message=reason or "System state transition",
        )

    def begin_pre_launch(self):
        """Start pre-launch checklist phase."""
        self.transition_state(
            SystemState.PRE_LAUNCH,
            reason="Beginning 78-item pre-launch validation"
        )

    def validate_for_live(self) -> Dict:
        """Validate system ready for Phase 0 paper trading."""
        deployment = self.checklist.can_deploy_live()

        if deployment['authorized']:
            # Initialize Phase 0
            self.phase_manager.initialize_at_phase(
                phase=DeploymentPhase.PHASE_0_PAPER,
                starting_capital=0,  # Paper
            )
            self.transition_state(
                SystemState.PHASE_VALIDATION,
                reason="All 78 checklist items passed. Ready for Phase 0 paper trading."
            )
            return {
                'authorized': True,
                'message': 'System validated for Phase 0 paper trading',
            }
        else:
            return {
                'authorized': False,
                'message': deployment['reason'],
                'incomplete_items': deployment.get('incomplete_items', []),
            }

    def graduate_to_live(self):
        """After paper validation, begin live trading."""
        if self.state != SystemState.PHASE_VALIDATION:
            return {
                'success': False,
                'reason': f'Cannot graduate from {self.state.value}',
            }

        self.transition_state(
            SystemState.LIVE_TRADING,
            reason="Graduated from paper to live trading Phase 1"
        )
        return {'success': True}

    def evaluate_trade_proposal(
        self,
        strategy_name: str,
        trade_risk: float,
        current_equity: float,
        current_open_risk: float,
        psychology_passed: bool = True,
    ) -> Dict:
        """Evaluate whether a proposed trade should be executed."""

        # State check
        if self.state not in [SystemState.PHASE_VALIDATION, SystemState.LIVE_TRADING]:
            return {
                'approved': False,
                'reason': f'System not in trading state: {self.state.value}',
            }

        # Portfolio-level check
        portfolio_check = self.portfolio.validate_new_trade(
            strategy_name=strategy_name,
            trade_risk=trade_risk,
            current_equity=current_equity,
            current_open_risk=current_open_risk,
        )

        if not portfolio_check['approved']:
            return {
                'approved': False,
                'reason': f"Portfolio check failed: {portfolio_check['reasons']}",
                'portfolio_check': portfolio_check,
            }

        # Psychology gate
        if not psychology_passed:
            return {
                'approved': False,
                'reason': 'Pre-trade psychology checklist not passed',
            }

        return {
            'approved': True,
            'portfolio_check': portfolio_check,
        }

    def update_daily_state(
        self,
        equity: float,
        cash: float,
        open_positions: int,
        today_pnl: float,
        week_pnl: float,
        month_pnl: float,
        max_equity_30d: float = None,
        rule_adherence: float = 1.0,
        var_95: float = 0,
    ):
        """Daily state update across all subsystems."""

        # Update monitor
        self.monitor.update_dashboard(
            account_equity=equity,
            cash_balance=cash,
            open_positions=open_positions,
            today_pnl=today_pnl,
            week_pnl=week_pnl,
            month_pnl=month_pnl,
            max_equity_30d=max_equity_30d,
            var_95=var_95,
            rule_adherence_today=rule_adherence,
        )

        # Check for kill switch conditions
        if equity > 0:
            today_pnl_pct = today_pnl / equity
            if today_pnl_pct < -self.config.daily_loss_limit_pct:
                self.transition_state(
                    SystemState.HALTED,
                    reason=f"Daily loss limit hit: {today_pnl_pct*100:.2f}%"
                )

        # Check for DD recovery state
        if max_equity_30d and equity > 0:
            current_dd = (equity - max_equity_30d) / max_equity_30d
            if current_dd < -0.10 and self.state == SystemState.LIVE_TRADING:
                self.transition_state(
                    SystemState.DRAWDOWN_RECOVERY,
                    reason=f"DD reached {current_dd*100:.1f}% - reducing position sizes"
                )
            elif current_dd > -0.05 and self.state == SystemState.DRAWDOWN_RECOVERY:
                self.transition_state(
                    SystemState.LIVE_TRADING,
                    reason="DD recovered to within -5%, resuming normal sizing"
                )

    def evaluate_phase_advancement(
        self,
        sharpe_3month: float,
        rule_adherence: float,
        max_dd_at_phase: float,
        mentor_approval: bool = False,
    ) -> Dict:
        """Check if ready to advance phase."""
        # Update phase stats
        if self.phase_manager.current_stats:
            self.phase_manager.update_stats(
                current_date=datetime.now(),
                rolling_sharpe=sharpe_3month,
                rule_adherence_pct=rule_adherence,
                max_dd_experienced=max_dd_at_phase,
                mentor_approval=mentor_approval,
            )

        return self.phase_manager.check_exit_criteria()

    def get_system_status(self) -> Dict:
        """Generate comprehensive system status."""
        days_running = (datetime.now() - self.start_date).days

        status = {
            'state': self.state.value,
            'days_running': days_running,
            'last_state_change': self.last_state_change,
            'config': {
                'trader_name': self.config.trader_name,
                'starting_capital': self.config.starting_capital,
                'strategies': self.config.strategy_names,
            },
        }

        if self.phase_manager.current_phase:
            status['current_phase'] = self.phase_manager.current_phase.name

        if self.monitor.last_snapshot:
            s = self.monitor.last_snapshot
            status['equity'] = s.account_equity
            status['today_pnl'] = s.today_pnl
            status['health'] = s.system_health

        if self.checklist:
            cl_status = self.checklist.get_status()
            status['checklist_completion'] = f"{cl_status['completed_items']}/" \
                                             f"{cl_status['total_items']}"

        return status

    def generate_full_report(self) -> str:
        """Generate comprehensive system report."""
        lines = []
        lines.append("╔" + "═" * 68 + "╗")
        lines.append("║" + " " * 20 + "QUANTCFD CAPSTONE SYSTEM" + " " * 24 + "║")
        lines.append("╚" + "═" * 68 + "╝")

        # Status
        lines.append("")
        status = self.get_system_status()
        lines.append(f"Trader:           {status['config']['trader_name']}")
        lines.append(f"State:            {status['state'].upper()}")
        lines.append(f"Days running:     {status['days_running']}")
        if 'current_phase' in status:
            lines.append(f"Current phase:    {status['current_phase']}")

        # Subsystem status
        lines.append("")
        lines.append("─" * 70)

        # Checklist
        if self.checklist:
            cl_status = self.checklist.get_status()
            lines.append(f"Pre-launch:       {cl_status['completed_items']}/"
                         f"{cl_status['total_items']} items "
                         f"({cl_status['completion_pct']:.0f}%)")

        # Monitoring
        if self.monitor.last_snapshot:
            s = self.monitor.last_snapshot
            lines.append(f"Account equity:   ${s.account_equity:,.2f}")
            lines.append(f"Today P&L:        ${s.today_pnl:+,.2f} "
                         f"({s.today_pnl_pct:+.2f}%)")
            lines.append(f"Open positions:   {s.open_positions}")
            lines.append(f"System health:    {s.system_health.upper()}")

        # Recent state changes
        if self.state_history:
            lines.append("")
            lines.append("Recent state changes:")
            for h in self.state_history[-5:]:
                lines.append(f"  {h['timestamp'].strftime('%Y-%m-%d %H:%M')} "
                             f"{h['from'].value} → {h['to'].value}")

        lines.append("=" * 70)
        return "\n".join(lines)


def demo():
    """Demo capstone system end-to-end."""
    print("=" * 70)
    print("DEMO: Capstone integrated system")
    print("=" * 70)

    # Initialize system
    config = SystemConfig(
        trader_name="Trí",
        starting_capital=20000,
        strategy_names=['trend_xau', 'mr_eur', 'vol_bo_btc'],
    )

    system = CapstoneSystem(config)

    print("\n1. Initial state:")
    print(system.generate_full_report())

    # Begin pre-launch
    print("\n\n2. Beginning pre-launch checklist:")
    system.begin_pre_launch()

    # Complete all checklist items
    for item in system.checklist.items:
        system.checklist.mark_completed(item.item_id)

    # Validate for live
    print("\n3. Validating for Phase 0:")
    validation = system.validate_for_live()
    print(f"   Authorized: {validation['authorized']}")
    print(f"   Message: {validation['message']}")

    # Graduate to live (after paper validation in real life)
    print("\n4. Graduating to live trading:")
    grad_result = system.graduate_to_live()
    print(f"   Success: {grad_result['success']}")

    # Evaluate trade proposals
    print("\n5. Trade proposal evaluation:")
    trade_eval = system.evaluate_trade_proposal(
        strategy_name='trend_xau',
        trade_risk=60,
        current_equity=20000,
        current_open_risk=120,
        psychology_passed=True,
    )
    print(f"   Approved: {trade_eval['approved']}")

    # Update daily state - normal day
    print("\n6. Updating daily state (normal day):")
    system.update_daily_state(
        equity=20300,
        cash=8000,
        open_positions=4,
        today_pnl=300,
        week_pnl=850,
        month_pnl=2100,
        max_equity_30d=20300,
        rule_adherence=0.95,
        var_95=400,
    )

    # Update with bad day - trigger DD recovery
    print("\n7. Updating daily state (DD scenario):")
    system.update_daily_state(
        equity=18000,
        cash=8000,
        open_positions=2,
        today_pnl=-200,
        week_pnl=-450,
        month_pnl=-100,
        max_equity_30d=20300,
        rule_adherence=0.93,
        var_95=400,
    )

    # Final report
    print("\n8. Final system report:")
    print(system.generate_full_report())

    print("\n" + "=" * 70)
    print("Demo complete. Capstone system integration validated.")
    print("=" * 70)


if __name__ == "__main__":
    demo()
