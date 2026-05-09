"""
Scaling engine.

Validates 8-criteria checklist for capital scaling decisions,
implements 50% rule for DD-based position size adjustment.

QuantCFD Chapter 12 - Capstone deployment
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional


@dataclass
class ScalingCriteria:
    """8 criteria that must ALL pass for capital scaling."""
    # Required fields (no defaults)
    months_at_phase: float
    closed_trades: int
    sharpe_3month: float
    max_dd_at_size: float        # As decimal (-0.12 = -12%)
    rule_adherence_pct: float
    emotional_state_stable: bool
    mentor_approval: bool
    personal_life_stable: bool

    # Optional thresholds (with defaults)
    months_required: float = 6.0
    trades_required: int = 100
    sharpe_required: float = 1.0
    max_dd_acceptable: float = -0.15
    adherence_required: float = 0.90

    def check_all(self) -> Dict:
        """Check all criteria."""
        criteria_results = {
            'time_at_phase': self.months_at_phase >= self.months_required,
            'trade_count': self.closed_trades >= self.trades_required,
            'sharpe_sustained': self.sharpe_3month >= self.sharpe_required,
            'dd_acceptable': self.max_dd_at_size >= self.max_dd_acceptable,
            'rule_adherence': self.rule_adherence_pct >= self.adherence_required,
            'emotional_stable': self.emotional_state_stable,
            'mentor_approval': self.mentor_approval,
            'personal_life_stable': self.personal_life_stable,
        }

        passing = sum(1 for v in criteria_results.values() if v)
        all_pass = all(criteria_results.values())

        return {
            'all_pass': all_pass,
            'passing': passing,
            'total': 8,
            'criteria': criteria_results,
            'failing': [k for k, v in criteria_results.items() if not v],
        }


class ScalingEngine:
    """
    Capital scaling decision engine.

    Implements:
    - 8-criteria validation
    - Gradual scaling (25% increments)
    - 50% rule for DD-based position size adjustment
    """

    def __init__(self, current_capital: float):
        self.current_capital = current_capital
        self.scaling_history: List[Dict] = []

        # 50% rule thresholds (DD-based position sizing)
        self.dd_size_table = [
            (-0.05, 1.00),   # 0-5% DD: 100% normal size
            (-0.10, 0.75),   # 5-10% DD: 75% size
            (-0.15, 0.50),   # 10-15% DD: 50% size
            (-0.20, 0.25),   # 15-20% DD: 25% size
            (-1.00, 0.00),   # 20%+ DD: HALT
        ]

    def can_scale_up(
        self,
        target_capital: float,
        criteria: ScalingCriteria,
    ) -> Dict:
        """
        Determine if scaling up to target capital is approved.

        Returns dict with approval and reasoning.
        """
        if target_capital <= self.current_capital:
            return {
                'approved': False,
                'reason': 'Target not greater than current',
            }

        # Check 8 criteria
        criteria_check = criteria.check_all()

        if not criteria_check['all_pass']:
            return {
                'approved': False,
                'reason': f"Failed criteria: {criteria_check['failing']}",
                'criteria_check': criteria_check,
            }

        # Check increment (25% max recommended)
        increment_pct = (target_capital - self.current_capital) / self.current_capital
        if increment_pct > 0.30:  # 30% allows 25% + buffer
            return {
                'approved': False,
                'reason': f'Increment {increment_pct*100:.0f}% too aggressive. '
                          f'Max recommended 25% per scale-up.',
                'recommended_target': self.current_capital * 1.25,
            }

        return {
            'approved': True,
            'old_capital': self.current_capital,
            'new_capital': target_capital,
            'increment_pct': increment_pct * 100,
            'criteria_check': criteria_check,
        }

    def execute_scale_up(
        self,
        new_capital: float,
        criteria: ScalingCriteria,
    ) -> Dict:
        """Execute approved scale-up."""
        result = self.can_scale_up(new_capital, criteria)

        if not result['approved']:
            return result

        # Record history
        self.scaling_history.append({
            'date': datetime.now(),
            'action': 'scale_up',
            'old_capital': self.current_capital,
            'new_capital': new_capital,
            'increment_pct': result['increment_pct'],
        })

        self.current_capital = new_capital
        return result

    def execute_scale_down(
        self,
        new_capital: float,
        reason: str,
    ) -> Dict:
        """Execute scale-down (always allowed for risk reduction)."""
        if new_capital >= self.current_capital:
            return {
                'success': False,
                'reason': 'Scale-down requires lower capital',
            }

        self.scaling_history.append({
            'date': datetime.now(),
            'action': 'scale_down',
            'old_capital': self.current_capital,
            'new_capital': new_capital,
            'reason': reason,
        })

        self.current_capital = new_capital
        return {
            'success': True,
            'old_capital': self.scaling_history[-1]['old_capital'],
            'new_capital': new_capital,
            'reason': reason,
        }

    def compute_position_size_factor(
        self,
        current_dd_pct: float,
    ) -> float:
        """
        Apply 50% rule: as DD grows, reduce position size.

        Returns factor 0-1 to multiply normal position size.
        """
        for threshold, factor in self.dd_size_table:
            if current_dd_pct >= threshold:
                return factor
        return 0.0  # Below all thresholds = halt

    def evaluate_scale_down_triggers(
        self,
        current_dd_pct: float,
        consecutive_loss_months: int = 0,
        personal_crisis: bool = False,
        system_bug_detected: bool = False,
    ) -> Optional[Dict]:
        """Check for triggers requiring scale-down."""
        triggers = []

        if current_dd_pct < -0.15:
            triggers.append({
                'type': 'critical_drawdown',
                'severity': 'critical',
                'recommended_action': 'reduce_capital_50pct',
            })

        if consecutive_loss_months >= 3:
            triggers.append({
                'type': 'sustained_underperformance',
                'severity': 'high',
                'recommended_action': 'reduce_capital_25pct',
            })

        if personal_crisis:
            triggers.append({
                'type': 'personal_crisis',
                'severity': 'high',
                'recommended_action': 'reduce_capital_or_halt',
            })

        if system_bug_detected:
            triggers.append({
                'type': 'system_integrity',
                'severity': 'critical',
                'recommended_action': 'halt_immediately',
            })

        if triggers:
            return {
                'should_scale_down': True,
                'triggers': triggers,
                'most_severe': max(triggers, key=lambda t:
                                   ['low', 'medium', 'high', 'critical'].index(t['severity'])),
            }
        return None

    def get_recovery_threshold(self, peak_capital: float) -> float:
        """Compute capital level at which to resume full size."""
        return peak_capital * 0.95  # Recovery to within 5% of peak

    def generate_scaling_report(
        self,
        criteria: ScalingCriteria,
        proposed_capital: float = None,
    ) -> str:
        """Generate scaling decision report."""
        lines = []
        lines.append("=" * 60)
        lines.append("SCALING DECISION REPORT")
        lines.append("=" * 60)
        lines.append(f"Current capital:  ${self.current_capital:,.2f}")
        if proposed_capital:
            lines.append(f"Proposed capital: ${proposed_capital:,.2f}")
            increment = (proposed_capital - self.current_capital) / self.current_capital
            lines.append(f"Increment:        {increment*100:+.1f}%")
        lines.append("")

        lines.append("8-CRITERIA CHECK:")
        check = criteria.check_all()
        criteria_names = {
            'time_at_phase': 'Time at phase',
            'trade_count': 'Trade count',
            'sharpe_sustained': 'Sharpe sustained',
            'dd_acceptable': 'Max DD acceptable',
            'rule_adherence': 'Rule adherence',
            'emotional_stable': 'Emotional state',
            'mentor_approval': 'Mentor approval',
            'personal_life_stable': 'Personal life',
        }
        for key, name in criteria_names.items():
            status = "✓" if check['criteria'][key] else "✗"
            lines.append(f"  {status} {name}")

        lines.append("")
        lines.append(f"Passing: {check['passing']}/8")

        if proposed_capital:
            result = self.can_scale_up(proposed_capital, criteria)
            if result['approved']:
                lines.append(f"\n✅ SCALE-UP APPROVED")
            else:
                lines.append(f"\n❌ SCALE-UP DENIED")
                lines.append(f"   Reason: {result['reason']}")

        lines.append("=" * 60)
        return "\n".join(lines)


def demo():
    """Demo scaling engine."""
    print("=" * 60)
    print("DEMO: Scaling engine")
    print("=" * 60)

    engine = ScalingEngine(current_capital=20000)

    # Scenario 1: All criteria passing
    print("\n1. Scenario: All criteria passing, scale Phase 3 → Phase 4")
    criteria_pass = ScalingCriteria(
        months_at_phase=8,
        closed_trades=180,
        sharpe_3month=1.15,
        max_dd_at_size=-0.10,
        rule_adherence_pct=0.94,
        emotional_state_stable=True,
        mentor_approval=True,
        personal_life_stable=True,
    )
    print(engine.generate_scaling_report(criteria_pass, proposed_capital=25000))

    # Execute
    result = engine.execute_scale_up(new_capital=25000, criteria=criteria_pass)
    print(f"\nExecution result: {'SUCCESS' if result['approved'] else 'FAILED'}")
    print(f"New capital: ${engine.current_capital:,.2f}")

    # Scenario 2: Some criteria failing
    print("\n\n2. Scenario: Some criteria failing")
    criteria_fail = ScalingCriteria(
        months_at_phase=4,    # Fail: < 6 months
        closed_trades=80,     # Fail: < 100 trades
        sharpe_3month=1.05,
        max_dd_at_size=-0.08,
        rule_adherence_pct=0.92,
        emotional_state_stable=True,
        mentor_approval=True,
        personal_life_stable=True,
    )
    print(engine.generate_scaling_report(criteria_fail, proposed_capital=35000))

    # Scenario 3: 50% rule based on DD
    print("\n\n3. 50% rule - position size factor at different DDs:")
    for dd in [-0.02, -0.07, -0.12, -0.18, -0.25]:
        factor = engine.compute_position_size_factor(dd)
        bar = '█' * int(factor * 20) + '░' * (20 - int(factor * 20))
        print(f"   DD = {dd*100:>5.1f}% → factor = {factor:.2f} [{bar}]")

    # Scenario 4: Scale-down triggers
    print("\n\n4. Scale-down trigger evaluation:")
    triggers = engine.evaluate_scale_down_triggers(
        current_dd_pct=-0.18,
        consecutive_loss_months=2,
    )
    if triggers and triggers['should_scale_down']:
        print(f"   ⚠️  Scale-down triggered:")
        for t in triggers['triggers']:
            print(f"      - {t['type']}: {t['recommended_action']}")
        print(f"   Most severe: {triggers['most_severe']['type']}")

    # Execute scale-down
    print("\n5. Executing scale-down:")
    result = engine.execute_scale_down(
        new_capital=engine.current_capital * 0.5,
        reason="Critical drawdown breach",
    )
    print(f"   Old: ${result['old_capital']:,.2f}")
    print(f"   New: ${result['new_capital']:,.2f}")
    print(f"   Reason: {result['reason']}")

    # Scaling history
    print("\n6. Scaling history:")
    for entry in engine.scaling_history:
        print(f"   {entry['date'].strftime('%Y-%m-%d')} {entry['action']:12s} "
              f"${entry['old_capital']:>10,.0f} → ${entry['new_capital']:>10,.0f}")

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
