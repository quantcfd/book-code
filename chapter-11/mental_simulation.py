"""
QuantCFD - Chapter 11
mental_simulation.py - Mental rehearsal of high-stress trading scenarios

7 scenarios to rehearse weekly:
1. Major news event hitting positions
2. Big winning streak
3. Big losing streak
4. Daily loss limit hit
5. Friend bragging about gains
6. Position approaching stop
7. Crisis market environment
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional
import random


@dataclass
class Scenario:
    """High-stress trading scenario for mental rehearsal."""
    name: str
    description: str
    typical_emotional_response: str
    correct_action: str
    common_mistakes: List[str]
    physiological_signals: List[str]


# 7 standard scenarios from Ch11.14
STANDARD_SCENARIOS = [
    Scenario(
        "major_news_event",
        "Fed decision creates 200-pip spike. Position deeply underwater.",
        "Panic, urge to immediately close at loss",
        "Follow stop loss as planned. If hit, accept. If not, hold to original target.",
        [
            "Close position pre-stop in panic",
            "Move stop further to avoid loss",
            "Add to losing position to average down",
        ],
        [
            "Heart rate spike",
            "Tunnel vision on chart",
            "Difficulty breathing",
            "Sweating",
        ],
    ),
    Scenario(
        "winning_streak",
        "10 wins in row. Account up 15% this month. Euphoric.",
        "Confidence high, urge to size up 'while hot'",
        "Maintain position size formula. Journal the urge. Do NOT increase size.",
        [
            "Increase position size 2-3x",
            "Skip pre-trade checklist (too confident)",
            "Take lower-quality setups",
            "Brag publicly (creates pressure)",
        ],
        [
            "Elevated mood",
            "Restless energy",
            "Strong urge to act",
            "Skip mundane checks",
        ],
    ),
    Scenario(
        "losing_streak",
        "5 losses in row. Account down 8% this week. Frustrated.",
        "Anger, urge to revenge trade and recover",
        "Trigger halt rule. Walk away. Exercise. No trades for cooling-off period.",
        [
            "Take revenge trade with bigger size",
            "Remove stop loss",
            "Open multiple positions simultaneously",
            "Skip checklist (urgency)",
        ],
        [
            "Heart rate elevated baseline",
            "Tense shoulders",
            "Aggressive thoughts",
            "Difficulty focusing on anything else",
        ],
    ),
    Scenario(
        "daily_loss_limit_hit",
        "Account down -3% intraday. Daily loss limit triggered.",
        "Frustration, urge to 'just one more trade' to recover",
        "Close trading platform immediately. No exceptions. Walk away.",
        [
            "Override halt rule with 'just one more'",
            "Switch to demo account để 'recover psychologically'",
            "Watch market obsessively for 'opportunities'",
            "Open new position 'just to break even'",
        ],
        [
            "Restless, cannot sit still",
            "Urge to constantly check phone",
            "Physical pull toward computer",
            "Inability to enjoy other activities",
        ],
    ),
    Scenario(
        "friend_bragging",
        "Discord friend posts +50% screenshot from one trade.",
        "FOMO, urge to enter market immediately",
        "Congratulate genuinely. Return to own system. Resist all urges to trade based on this.",
        [
            "Open random trade to 'participate'",
            "Increase size to 'catch up'",
            "Skip own system signals chasing similar",
            "Spend hours analyzing why you 'missed' it",
        ],
        [
            "Anxiety in chest",
            "Comparison thoughts running",
            "Impulse to act immediately",
            "Difficulty with own analysis",
        ],
    ),
    Scenario(
        "approaching_stop",
        "Trade green earlier, now -50% of risk. Stop loss 20 pips away.",
        "Anxiety about loss, urge to protect 'profits' or move stop",
        "Let stop hit if planned. Do NOT move stop further. Do NOT take partial.",
        [
            "Move stop further to avoid hit",
            "Move stop tighter to 'lock in less loss'",
            "Take partial profit early",
            "Add to position",
        ],
        [
            "Constantly checking chart",
            "Mental calculations of P&L scenarios",
            "Physical tension",
            "Breath shallow",
        ],
    ),
    Scenario(
        "crisis_environment",
        "VIX > 50. All positions volatile. Multiple drawdowns simultaneously.",
        "Fear, urge to liquidate everything or panic-buy hedges",
        "Execute crisis playbook: halt new entries, reduce size 50%, monitor only.",
        [
            "Liquidate everything in panic",
            "Buy expensive hedges at peak",
            "Switch strategies frantically",
            "Watch market 24/7 obsessively",
        ],
        [
            "Sustained elevated heart rate",
            "Sleep disruption",
            "Constant phone checking",
            "Family relationships strained",
        ],
    ),
]


@dataclass
class RehearsalLog:
    """Log of a mental rehearsal session."""
    timestamp: datetime
    scenario_name: str
    duration_minutes: int
    emotional_response_intensity: int  # 1-10 how strong was emotion during rehearsal
    correct_action_clarity: int  # 1-10 how clear was correct response
    notes: str = ""


class MentalSimulation:
    """
    Mental rehearsal system.

    Rehearse 1 scenario per day, rotate through 7 scenarios weekly.
    Track over time để build automatic responses.
    """

    def __init__(self, scenarios: Optional[List[Scenario]] = None):
        self.scenarios = scenarios or STANDARD_SCENARIOS
        self.scenarios_by_name = {s.name: s for s in self.scenarios}
        self.logs: List[RehearsalLog] = []

    def get_scenario(self, name: str) -> Scenario:
        """Get scenario by name."""
        if name not in self.scenarios_by_name:
            raise ValueError(f"Unknown scenario: {name}")
        return self.scenarios_by_name[name]

    def random_scenario(self, seed: Optional[int] = None) -> Scenario:
        """Get random scenario for daily rehearsal."""
        if seed is not None:
            random.seed(seed)
        return random.choice(self.scenarios)

    def rehearsal_for_today(self, day_of_week: Optional[int] = None) -> Scenario:
        """
        Get recommended scenario for today.

        Default: rotate through 7 scenarios by day of week.
        """
        if day_of_week is None:
            day_of_week = datetime.now().weekday()
        return self.scenarios[day_of_week % len(self.scenarios)]

    def conduct_rehearsal(
        self,
        scenario_name: str,
        duration_minutes: int = 5,
        emotional_response_intensity: int = 5,
        correct_action_clarity: int = 7,
        notes: str = "",
    ) -> RehearsalLog:
        """Log a rehearsal session."""
        log = RehearsalLog(
            timestamp=datetime.now(),
            scenario_name=scenario_name,
            duration_minutes=duration_minutes,
            emotional_response_intensity=emotional_response_intensity,
            correct_action_clarity=correct_action_clarity,
            notes=notes,
        )
        self.logs.append(log)
        return log

    def rehearsal_stats(self) -> Dict[str, float]:
        """Compute statistics on rehearsal practice."""
        if not self.logs:
            return {}

        total_minutes = sum(l.duration_minutes for l in self.logs)
        avg_emotional_response = sum(l.emotional_response_intensity for l in self.logs) / len(self.logs)
        avg_clarity = sum(l.correct_action_clarity for l in self.logs) / len(self.logs)

        # Count distinct scenarios rehearsed
        scenarios_practiced = set(l.scenario_name for l in self.logs)

        return {
            "n_rehearsals": len(self.logs),
            "total_minutes": total_minutes,
            "avg_emotional_intensity": avg_emotional_response,
            "avg_action_clarity": avg_clarity,
            "scenarios_practiced": len(scenarios_practiced),
            "total_scenarios": len(self.scenarios),
        }

    def display_scenario(self, name: str) -> str:
        """Format scenario for rehearsal session."""
        s = self.get_scenario(name)
        lines = [
            "=" * 70,
            f"MENTAL REHEARSAL: {s.name.upper().replace('_', ' ')}",
            "=" * 70,
            "",
            f"SCENARIO:",
            f"  {s.description}",
            "",
            f"TYPICAL EMOTIONAL RESPONSE:",
            f"  {s.typical_emotional_response}",
            "",
            f"PHYSIOLOGICAL SIGNALS TO RECOGNIZE:",
        ]
        for sig in s.physiological_signals:
            lines.append(f"  - {sig}")

        lines.extend([
            "",
            f"COMMON MISTAKES TO AVOID:",
        ])
        for m in s.common_mistakes:
            lines.append(f"  ✗ {m}")

        lines.extend([
            "",
            f"CORRECT ACTION:",
            f"  ✓ {s.correct_action}",
            "",
            "REHEARSAL STEPS:",
            "  1. Find quiet space, close eyes",
            "  2. Vividly imagine the scenario (5 senses)",
            "  3. Feel the emotional response intensely",
            "  4. Visualize executing correct action despite emotion",
            "  5. Visualize positive outcome of correct action",
            "  6. Repeat 3-5 minutes",
            "",
            "=" * 70,
        ])
        return "\n".join(lines)

    def report(self) -> str:
        """Generate stats report."""
        lines = ["=" * 70, "MENTAL SIMULATION REPORT", "=" * 70]

        stats = self.rehearsal_stats()
        if not stats:
            return "\n".join(lines + ["\nNo rehearsals logged yet."])

        lines.append(f"\nTotal rehearsals: {stats['n_rehearsals']}")
        lines.append(f"Total time invested: {stats['total_minutes']} minutes")
        lines.append(f"Scenarios practiced: {stats['scenarios_practiced']}/{stats['total_scenarios']}")
        lines.append(f"Average emotional intensity: {stats['avg_emotional_intensity']:.1f}/10")
        lines.append(f"Average action clarity: {stats['avg_action_clarity']:.1f}/10")

        if stats["scenarios_practiced"] < stats["total_scenarios"]:
            untouched = set(s.name for s in self.scenarios) - set(l.scenario_name for l in self.logs)
            lines.append(f"\nScenarios not yet practiced:")
            for s in untouched:
                lines.append(f"  - {s}")

        lines.append("=" * 70)
        return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 70)
    print("Mental Simulation Demo")
    print("=" * 70)

    sim = MentalSimulation()

    # Show all 7 scenarios available
    print(f"\nAvailable scenarios ({len(sim.scenarios)}):")
    for s in sim.scenarios:
        print(f"  - {s.name}")

    # Display today's recommended rehearsal
    today_scenario = sim.rehearsal_for_today()
    print(f"\nToday's scenario: {today_scenario.name}")
    print(sim.display_scenario(today_scenario.name))

    # Simulate 14 days of rehearsal practice
    import random as r2
    r2.seed(42)
    base = datetime(2024, 6, 1)
    for day in range(14):
        scenario = sim.scenarios[day % len(sim.scenarios)]
        sim.conduct_rehearsal(
            scenario_name=scenario.name,
            duration_minutes=r2.randint(3, 10),
            emotional_response_intensity=r2.randint(4, 8),
            correct_action_clarity=r2.randint(5, 9),
        )

    print("\n")
    print(sim.report())
