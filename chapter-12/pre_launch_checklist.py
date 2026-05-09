"""
Pre-launch checklist enforcer.

Validates 78-item checklist before live deployment is allowed.
Categorized by infrastructure section A-F.

QuantCFD Chapter 12 - Capstone deployment
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class ChecklistSection(Enum):
    DATA = "A_data_infrastructure"
    STRATEGY = "B_strategy_validation"
    RISK = "C_risk_infrastructure"
    PSYCHOLOGY = "D_psychology_infrastructure"
    EXECUTION = "E_execution_infrastructure"
    OPERATIONS = "F_operations_monitoring"


@dataclass
class ChecklistItem:
    """Single item in pre-launch checklist."""
    item_id: str
    section: ChecklistSection
    description: str
    is_critical: bool = True
    is_completed: bool = False
    completed_date: Optional[datetime] = None
    notes: str = ""


class PreLaunchChecklist:
    """
    Manage 78-item pre-launch checklist.

    All items must be completed before live deployment authorized.
    """

    def __init__(self):
        self.items: List[ChecklistItem] = []
        self._initialize_items()

    def _initialize_items(self):
        """Define all 78 checklist items."""
        # Section A: Data infrastructure (12 items)
        a_items = [
            ('A1', 'Real-time price feed working (5 min uptime test)'),
            ('A2', 'Historical data 5+ years stored'),
            ('A3', 'Data cleaning pipeline tested'),
            ('A4', 'Timezone handling correct (Asia/Ho_Chi_Minh)'),
            ('A5', 'Gap detection automated'),
            ('A6', 'Multi-broker price comparison capability'),
            ('A7', 'News and economic calendar integration'),
            ('A8', 'Data backup automated daily'),
            ('A9', 'Fallback data source identified'),
            ('A10', 'Data quality monitoring alerts'),
            ('A11', 'Storage capacity for 5 years forward'),
            ('A12', 'Recovery procedure documented if data corrupted'),
        ]
        for item_id, desc in a_items:
            self.items.append(ChecklistItem(
                item_id=item_id, section=ChecklistSection.DATA, description=desc
            ))

        # Section B: Strategy validation (15 items)
        b_items = [
            ('B1', 'Each strategy backtest 2015-2024 done'),
            ('B2', 'Walk-forward 2015-2018 train, 2019-2024 test'),
            ('B3', 'Out-of-sample Sharpe > 0.8 each strategy'),
            ('B4', 'Max DD acceptable (< -20% each strategy)'),
            ('B5', 'Statistical significance p < 0.05 (vs random)'),
            ('B6', 'Multi-asset robustness check'),
            ('B7', 'Multi-timeframe robustness check'),
            ('B8', 'Parameter sensitivity analysis done'),
            ('B9', 'Slippage assumptions realistic'),
            ('B10', 'Commission costs included'),
            ('B11', 'Crisis period stress tested (2020 COVID)'),
            ('B12', 'Strategy correlation matrix computed'),
            ('B13', 'Combined portfolio Sharpe > 1.2'),
            ('B14', 'Strategy code unit tested'),
            ('B15', 'Strategy code peer reviewed'),
        ]
        for item_id, desc in b_items:
            self.items.append(ChecklistItem(
                item_id=item_id, section=ChecklistSection.STRATEGY, description=desc
            ))

        # Section C: Risk infrastructure (12 items)
        c_items = [
            ('C1', 'Position sizing function tested'),
            ('C2', 'Stop loss orders programmable on broker'),
            ('C3', 'Daily loss limit enforced via code'),
            ('C4', 'Weekly loss limit enforced'),
            ('C5', 'Monthly loss limit enforced'),
            ('C6', 'Drawdown stages defined (5/10/15/20%)'),
            ('C7', 'Kill switch tested in simulation'),
            ('C8', 'Maximum concurrent positions limit set'),
            ('C9', 'VaR computed daily'),
            ('C10', 'Correlation breach detection'),
            ('C11', 'Risk dashboard real-time'),
            ('C12', 'Manual override procedure documented'),
        ]
        for item_id, desc in c_items:
            self.items.append(ChecklistItem(
                item_id=item_id, section=ChecklistSection.RISK, description=desc
            ))

        # Section D: Psychology infrastructure (12 items)
        d_items = [
            ('D1', 'Trade journal database setup'),
            ('D2', 'Pre-trade checklist hardcoded'),
            ('D3', 'Post-trade review automated'),
            ('D4', 'Emotion tracker logging 3x/day'),
            ('D5', 'Bias detector running weekly'),
            ('D6', 'Habit tracker setup (5 critical habits)'),
            ('D7', 'Mental rehearsal scenarios scheduled'),
            ('D8', 'Mentor relationship established'),
            ('D9', 'Family awareness and support'),
            ('D10', 'Lifestyle foundations (sleep, exercise)'),
            ('D11', 'Identity work (process executor mindset)'),
            ('D12', 'Drawdown psychology rehearsed'),
        ]
        for item_id, desc in d_items:
            self.items.append(ChecklistItem(
                item_id=item_id, section=ChecklistSection.PSYCHOLOGY, description=desc
            ))

        # Section E: Execution infrastructure (15 items)
        e_items = [
            ('E1', 'Broker account opened and funded paper'),
            ('E2', 'Broker API credentials secured'),
            ('E3', 'Order placement function tested 100x'),
            ('E4', 'Order modification (stop, target) tested'),
            ('E5', 'Order cancellation tested'),
            ('E6', 'Slippage measurement live'),
            ('E7', 'Latency measurement (typically 50-200ms)'),
            ('E8', 'Position reconciliation (broker vs internal)'),
            ('E9', 'Trade history download automated'),
            ('E10', 'P&L calculation matches broker'),
            ('E11', 'Multi-asset trading enabled'),
            ('E12', 'After-hours order handling'),
            ('E13', 'Weekend gap handling'),
            ('E14', 'Failover broker identified'),
            ('E15', 'Recovery procedure if broker outage'),
        ]
        for item_id, desc in e_items:
            self.items.append(ChecklistItem(
                item_id=item_id, section=ChecklistSection.EXECUTION, description=desc
            ))

        # Section F: Operations + monitoring (12 items)
        f_items = [
            ('F1', 'System startup automated'),
            ('F2', 'Daily morning report generated'),
            ('F3', 'End-of-day report generated'),
            ('F4', 'Weekly review automated'),
            ('F5', 'Monthly review template ready'),
            ('F6', 'Discord/Telegram alerts working'),
            ('F7', 'Critical alerts (kill switch, data loss)'),
            ('F8', 'Backup procedures (daily, weekly, monthly)'),
            ('F9', 'Code version control (Git) committed'),
            ('F10', 'Documentation up to date'),
            ('F11', 'Disaster recovery tested'),
            ('F12', 'Personal financial buffer (6 months expenses)'),
        ]
        for item_id, desc in f_items:
            self.items.append(ChecklistItem(
                item_id=item_id, section=ChecklistSection.OPERATIONS, description=desc
            ))

    def mark_completed(self, item_id: str, notes: str = "") -> bool:
        """Mark a checklist item as completed."""
        for item in self.items:
            if item.item_id == item_id:
                item.is_completed = True
                item.completed_date = datetime.now()
                item.notes = notes
                return True
        return False

    def mark_incomplete(self, item_id: str) -> bool:
        """Mark a checklist item as incomplete (revert)."""
        for item in self.items:
            if item.item_id == item_id:
                item.is_completed = False
                item.completed_date = None
                return True
        return False

    def get_status(self) -> Dict:
        """Compute overall checklist status."""
        total = len(self.items)
        completed = sum(1 for item in self.items if item.is_completed)

        section_status = {}
        for section in ChecklistSection:
            section_items = [item for item in self.items if item.section == section]
            section_completed = sum(1 for item in section_items if item.is_completed)
            section_status[section.value] = {
                'total': len(section_items),
                'completed': section_completed,
                'pct': section_completed / len(section_items) * 100
                       if section_items else 0,
            }

        return {
            'total_items': total,
            'completed_items': completed,
            'completion_pct': completed / total * 100 if total > 0 else 0,
            'sections': section_status,
            'is_ready_for_live': completed == total,
        }

    def get_incomplete_items(self) -> List[ChecklistItem]:
        """Return list of items not yet completed."""
        return [item for item in self.items if not item.is_completed]

    def can_deploy_live(self) -> Dict:
        """Decision: is system ready for live deployment?"""
        incomplete = self.get_incomplete_items()

        if not incomplete:
            return {
                'authorized': True,
                'reason': 'All 78 checklist items completed',
                'incomplete_count': 0,
            }
        else:
            return {
                'authorized': False,
                'reason': f'{len(incomplete)} items incomplete - cannot deploy live',
                'incomplete_count': len(incomplete),
                'incomplete_items': [item.item_id for item in incomplete],
            }

    def generate_report(self) -> str:
        """Generate text report of checklist status."""
        status = self.get_status()

        lines = []
        lines.append("=" * 60)
        lines.append("PRE-LAUNCH CHECKLIST STATUS")
        lines.append("=" * 60)
        lines.append(f"Total items:     {status['total_items']}")
        lines.append(f"Completed:       {status['completed_items']}")
        lines.append(f"Completion:      {status['completion_pct']:.1f}%")
        lines.append("")

        lines.append("By section:")
        section_names = {
            'A_data_infrastructure': 'Data Infrastructure',
            'B_strategy_validation': 'Strategy Validation',
            'C_risk_infrastructure': 'Risk Infrastructure',
            'D_psychology_infrastructure': 'Psychology Infrastructure',
            'E_execution_infrastructure': 'Execution Infrastructure',
            'F_operations_monitoring': 'Operations + Monitoring',
        }
        for key, name in section_names.items():
            s = status['sections'][key]
            bar = '█' * int(s['pct'] / 5) + '░' * (20 - int(s['pct'] / 5))
            lines.append(f"  {name:32s} [{bar}] {s['completed']}/{s['total']} ({s['pct']:.0f}%)")

        lines.append("")
        deployment = self.can_deploy_live()
        if deployment['authorized']:
            lines.append("✅ AUTHORIZED FOR LIVE DEPLOYMENT")
        else:
            lines.append(f"❌ NOT AUTHORIZED - {deployment['incomplete_count']} items incomplete")
            if deployment.get('incomplete_items'):
                items_str = ", ".join(deployment['incomplete_items'][:10])
                if len(deployment['incomplete_items']) > 10:
                    items_str += f" ... (+{len(deployment['incomplete_items'])-10} more)"
                lines.append(f"   Missing: {items_str}")

        lines.append("=" * 60)
        return "\n".join(lines)


def demo():
    """Demo pre-launch checklist."""
    print("=" * 60)
    print("DEMO: Pre-launch checklist")
    print("=" * 60)

    checklist = PreLaunchChecklist()

    # Initial state
    print("\n1. Initial state (empty checklist):")
    print(checklist.generate_report())

    # Mark some items completed
    print("\n\n2. Marking subset of items as completed:")

    # Section A: complete most
    for item_id in ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']:
        checklist.mark_completed(item_id, notes="Verified in dev environment")

    # Section B: complete most
    for item_id in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9',
                    'B10', 'B11', 'B12', 'B13', 'B14']:
        checklist.mark_completed(item_id)

    # Section C: complete all
    for item_id in [f'C{i}' for i in range(1, 13)]:
        checklist.mark_completed(item_id)

    # Section D: half
    for item_id in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
        checklist.mark_completed(item_id)

    # Section E: most
    for item_id in [f'E{i}' for i in range(1, 12)]:
        checklist.mark_completed(item_id)

    # Section F: half
    for item_id in ['F1', 'F2', 'F3', 'F4', 'F5', 'F6']:
        checklist.mark_completed(item_id)

    print(checklist.generate_report())

    # Try to deploy
    print("\n\n3. Deployment authorization check:")
    deployment = checklist.can_deploy_live()
    print(f"   Authorized: {deployment['authorized']}")
    print(f"   Reason: {deployment['reason']}")

    # Complete remaining
    print("\n\n4. Completing all remaining items:")
    incomplete = checklist.get_incomplete_items()
    for item in incomplete:
        checklist.mark_completed(item.item_id)

    print(checklist.generate_report())

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
