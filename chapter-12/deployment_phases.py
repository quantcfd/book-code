"""
Phased deployment manager.

Tracks current phase (0-5), validates phase exit criteria,
recommends phase transitions for capital scaling.

QuantCFD Chapter 12 - Capstone deployment
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional


class DeploymentPhase(Enum):
    PHASE_0_PAPER = 0
    PHASE_1_TINY = 1        # $500-1k
    PHASE_2_SMALL = 2       # $5k
    PHASE_3_MEDIUM = 3      # $20k
    PHASE_4_SIGNIFICANT = 4  # $100k
    PHASE_5_INSTITUTIONAL = 5  # $500k+


@dataclass
class PhaseRequirements:
    """Exit criteria for a phase."""
    phase: DeploymentPhase
    capital_min: float
    capital_max: float
    duration_min_days: int
    min_trades: int
    min_sharpe: float
    max_dd_acceptable: float    # Maximum acceptable DD as decimal (-0.15 = -15%)
    rule_adherence_min: float   # Minimum rule adherence (0.90 = 90%)

    @classmethod
    def for_phase(cls, phase: DeploymentPhase) -> 'PhaseRequirements':
        """Get requirements for specified phase."""
        configs = {
            DeploymentPhase.PHASE_0_PAPER: cls(
                phase=DeploymentPhase.PHASE_0_PAPER,
                capital_min=0,
                capital_max=0,
                duration_min_days=28,
                min_trades=50,
                min_sharpe=0.8,
                max_dd_acceptable=-0.20,
                rule_adherence_min=0.85,
            ),
            DeploymentPhase.PHASE_1_TINY: cls(
                phase=DeploymentPhase.PHASE_1_TINY,
                capital_min=500,
                capital_max=1000,
                duration_min_days=28,
                min_trades=30,
                min_sharpe=0.5,
                max_dd_acceptable=-0.20,
                rule_adherence_min=0.90,
            ),
            DeploymentPhase.PHASE_2_SMALL: cls(
                phase=DeploymentPhase.PHASE_2_SMALL,
                capital_min=5000,
                capital_max=10000,
                duration_min_days=90,
                min_trades=100,
                min_sharpe=0.8,
                max_dd_acceptable=-0.15,
                rule_adherence_min=0.90,
            ),
            DeploymentPhase.PHASE_3_MEDIUM: cls(
                phase=DeploymentPhase.PHASE_3_MEDIUM,
                capital_min=20000,
                capital_max=50000,
                duration_min_days=180,
                min_trades=200,
                min_sharpe=1.0,
                max_dd_acceptable=-0.12,
                rule_adherence_min=0.92,
            ),
            DeploymentPhase.PHASE_4_SIGNIFICANT: cls(
                phase=DeploymentPhase.PHASE_4_SIGNIFICANT,
                capital_min=100000,
                capital_max=300000,
                duration_min_days=365,
                min_trades=500,
                min_sharpe=1.0,
                max_dd_acceptable=-0.12,
                rule_adherence_min=0.93,
            ),
            DeploymentPhase.PHASE_5_INSTITUTIONAL: cls(
                phase=DeploymentPhase.PHASE_5_INSTITUTIONAL,
                capital_min=500000,
                capital_max=10000000,
                duration_min_days=730,
                min_trades=1000,
                min_sharpe=1.0,
                max_dd_acceptable=-0.10,
                rule_adherence_min=0.95,
            ),
        }
        return configs[phase]


@dataclass
class PhaseStats:
    """Statistics for current phase."""
    phase: DeploymentPhase
    start_date: datetime
    current_date: datetime
    capital_at_start: float
    current_capital: float
    total_trades: int
    closed_trades: int
    realized_pnl: float
    rolling_sharpe: float
    max_dd_experienced: float
    rule_adherence_pct: float
    mentor_approval: bool = False
    family_supportive: bool = True

    @property
    def days_in_phase(self) -> int:
        return (self.current_date - self.start_date).days

    @property
    def total_return_pct(self) -> float:
        if self.capital_at_start > 0:
            return self.realized_pnl / self.capital_at_start * 100
        return 0


class PhasedDeploymentManager:
    """
    Manage gradual capital deployment across 6 phases (0-5).

    Validates exit criteria, recommends transitions, prevents premature scaling.
    """

    def __init__(self):
        self.current_phase: Optional[DeploymentPhase] = None
        self.phase_history: List[Dict] = []
        self.current_stats: Optional[PhaseStats] = None

    def initialize_at_phase(
        self,
        phase: DeploymentPhase,
        starting_capital: float,
        start_date: Optional[datetime] = None,
    ):
        """Begin tracking at specified phase."""
        if start_date is None:
            start_date = datetime.now()

        self.current_phase = phase
        self.current_stats = PhaseStats(
            phase=phase,
            start_date=start_date,
            current_date=start_date,
            capital_at_start=starting_capital,
            current_capital=starting_capital,
            total_trades=0,
            closed_trades=0,
            realized_pnl=0,
            rolling_sharpe=0,
            max_dd_experienced=0,
            rule_adherence_pct=1.0,
        )

    def update_stats(self, **kwargs):
        """Update current phase statistics."""
        if self.current_stats is None:
            raise ValueError("Phase not initialized. Call initialize_at_phase first.")

        for key, value in kwargs.items():
            if hasattr(self.current_stats, key):
                setattr(self.current_stats, key, value)

    def check_exit_criteria(self) -> Dict:
        """Check if current phase exit criteria met."""
        if self.current_stats is None or self.current_phase is None:
            return {'ready_to_advance': False, 'reason': 'Not initialized'}

        requirements = PhaseRequirements.for_phase(self.current_phase)

        criteria_results = {}

        # Duration
        criteria_results['duration'] = {
            'pass': self.current_stats.days_in_phase >= requirements.duration_min_days,
            'actual': self.current_stats.days_in_phase,
            'required': requirements.duration_min_days,
        }

        # Trade count
        criteria_results['trade_count'] = {
            'pass': self.current_stats.closed_trades >= requirements.min_trades,
            'actual': self.current_stats.closed_trades,
            'required': requirements.min_trades,
        }

        # Sharpe
        criteria_results['sharpe'] = {
            'pass': self.current_stats.rolling_sharpe >= requirements.min_sharpe,
            'actual': self.current_stats.rolling_sharpe,
            'required': requirements.min_sharpe,
        }

        # Max DD
        criteria_results['max_dd'] = {
            'pass': self.current_stats.max_dd_experienced >= requirements.max_dd_acceptable,
            'actual': self.current_stats.max_dd_experienced,
            'required': requirements.max_dd_acceptable,
        }

        # Rule adherence
        criteria_results['rule_adherence'] = {
            'pass': self.current_stats.rule_adherence_pct >= requirements.rule_adherence_min,
            'actual': self.current_stats.rule_adherence_pct,
            'required': requirements.rule_adherence_min,
        }

        # Mentor + family
        criteria_results['mentor_approval'] = {
            'pass': self.current_stats.mentor_approval,
            'actual': self.current_stats.mentor_approval,
            'required': True,
        }

        criteria_results['family_supportive'] = {
            'pass': self.current_stats.family_supportive,
            'actual': self.current_stats.family_supportive,
            'required': True,
        }

        all_pass = all(c['pass'] for c in criteria_results.values())

        return {
            'ready_to_advance': all_pass,
            'criteria': criteria_results,
            'failing_criteria': [k for k, v in criteria_results.items() if not v['pass']],
        }

    def get_next_phase(self) -> Optional[DeploymentPhase]:
        """Get the next phase after current."""
        if self.current_phase is None:
            return None

        next_phase_value = self.current_phase.value + 1
        if next_phase_value > 5:
            return None  # Already at top

        return DeploymentPhase(next_phase_value)

    def advance_to_next_phase(self, new_capital: float) -> Dict:
        """Advance to next phase if criteria met."""
        criteria = self.check_exit_criteria()

        if not criteria['ready_to_advance']:
            return {
                'success': False,
                'reason': f"Criteria not met: {criteria['failing_criteria']}",
            }

        next_phase = self.get_next_phase()
        if next_phase is None:
            return {
                'success': False,
                'reason': "Already at maximum phase",
            }

        # Validate capital matches new phase
        next_req = PhaseRequirements.for_phase(next_phase)
        if not (next_req.capital_min <= new_capital <= next_req.capital_max * 1.2):
            return {
                'success': False,
                'reason': f"Capital ${new_capital:,.0f} outside phase {next_phase.value} "
                          f"range (${next_req.capital_min:,}-${next_req.capital_max:,})",
            }

        # Archive current phase
        self.phase_history.append({
            'phase': self.current_phase,
            'duration_days': self.current_stats.days_in_phase,
            'final_capital': self.current_stats.current_capital,
            'total_pnl': self.current_stats.realized_pnl,
            'sharpe': self.current_stats.rolling_sharpe,
            'trades': self.current_stats.closed_trades,
            'rule_adherence': self.current_stats.rule_adherence_pct,
        })

        # Initialize new phase
        old_phase = self.current_phase
        self.initialize_at_phase(
            phase=next_phase,
            starting_capital=new_capital,
            start_date=datetime.now(),
        )

        return {
            'success': True,
            'old_phase': old_phase,
            'new_phase': next_phase,
            'new_capital': new_capital,
        }

    def generate_status_report(self) -> str:
        """Generate text status report."""
        if self.current_stats is None or self.current_phase is None:
            return "Phase manager not initialized."

        lines = []
        lines.append("=" * 60)
        lines.append(f"DEPLOYMENT STATUS: {self.current_phase.name}")
        lines.append("=" * 60)
        lines.append(f"Days in phase:   {self.current_stats.days_in_phase}")
        lines.append(f"Capital:         ${self.current_stats.current_capital:,.2f}")
        lines.append(f"Total P&L:       ${self.current_stats.realized_pnl:,.2f} "
                     f"({self.current_stats.total_return_pct:+.2f}%)")
        lines.append(f"Closed trades:   {self.current_stats.closed_trades}")
        lines.append(f"Rolling Sharpe:  {self.current_stats.rolling_sharpe:.2f}")
        lines.append(f"Max DD:          {self.current_stats.max_dd_experienced:.1%}")
        lines.append(f"Rule adherence:  {self.current_stats.rule_adherence_pct:.1%}")
        lines.append("")

        lines.append("Exit criteria status:")
        criteria_check = self.check_exit_criteria()
        for criterion, result in criteria_check['criteria'].items():
            status = "✓" if result['pass'] else "✗"
            actual = result['actual']
            req = result['required']
            lines.append(f"  {status} {criterion:20s} actual={actual} required={req}")

        lines.append("")
        if criteria_check['ready_to_advance']:
            next_phase = self.get_next_phase()
            if next_phase:
                lines.append(f"✅ READY TO ADVANCE to {next_phase.name}")
            else:
                lines.append("✅ At maximum phase")
        else:
            failing = criteria_check['failing_criteria']
            lines.append(f"⏳ NOT READY - failing: {', '.join(failing)}")

        if self.phase_history:
            lines.append("")
            lines.append("Phase history:")
            for h in self.phase_history:
                lines.append(f"  {h['phase'].name:30s} "
                             f"{h['duration_days']:>4}d "
                             f"${h['final_capital']:>10,.0f} "
                             f"Sharpe={h['sharpe']:.2f}")

        lines.append("=" * 60)
        return "\n".join(lines)


def demo():
    """Demo phased deployment manager."""
    print("=" * 60)
    print("DEMO: Phased deployment manager")
    print("=" * 60)

    manager = PhasedDeploymentManager()

    # Start at Phase 0
    print("\n1. Initializing Phase 0 (paper trading):")
    manager.initialize_at_phase(
        phase=DeploymentPhase.PHASE_0_PAPER,
        starting_capital=0,
        start_date=datetime.now() - timedelta(days=45),
    )

    # Update stats - simulating 45 days of paper trading
    manager.update_stats(
        current_date=datetime.now(),
        current_capital=24500,
        closed_trades=58,
        realized_pnl=4500,
        rolling_sharpe=0.95,
        max_dd_experienced=-0.08,
        rule_adherence_pct=0.92,
        mentor_approval=True,
        family_supportive=True,
    )

    print(manager.generate_status_report())

    # Try to advance
    print("\n\n2. Attempting to advance to Phase 1 ($1k):")
    result = manager.advance_to_next_phase(new_capital=1000)
    print(f"   Success: {result['success']}")
    if result['success']:
        print(f"   New phase: {result['new_phase'].name}")
        print(f"   New capital: ${result['new_capital']:,.2f}")

    # Update Phase 1 stats - 50 days in
    print("\n\n3. After 50 days in Phase 1:")
    manager.update_stats(
        current_date=datetime.now() + timedelta(days=50),
        current_capital=1180,
        closed_trades=35,
        realized_pnl=180,
        rolling_sharpe=0.65,
        max_dd_experienced=-0.05,
        rule_adherence_pct=0.94,
        mentor_approval=True,
        family_supportive=True,
    )
    print(manager.generate_status_report())

    # Try to advance to Phase 2
    print("\n\n4. Attempting to advance to Phase 2 ($5k):")
    result = manager.advance_to_next_phase(new_capital=5000)
    print(f"   Success: {result['success']}")
    if result['success']:
        print(f"   New phase: {result['new_phase'].name}")

    # Try to advance with insufficient stats
    print("\n\n5. Trying to advance Phase 2 with insufficient time:")
    manager.update_stats(
        current_date=datetime.now() + timedelta(days=70),
        closed_trades=80,
        rolling_sharpe=0.7,
    )
    result = manager.advance_to_next_phase(new_capital=20000)
    print(f"   Success: {result['success']}")
    print(f"   Reason: {result.get('reason', 'N/A')}")

    print("\n" + "=" * 60)
    print("Demo complete. Phase progression validated.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
