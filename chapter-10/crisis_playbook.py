"""
QuantCFD — Chương 10.10
Crisis Playbook — state machine

4 phases: detection → stabilization → assessment → reentry.
Pre-defined actions per phase to remove emotional decision-making.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum


class CrisisPhase(Enum):
    NORMAL = "normal"
    DETECTION = "detection"
    STABILIZATION = "stabilization"
    ASSESSMENT = "assessment"
    REENTRY = "reentry"


class CrisisPlaybook:
    """
    Systematic crisis response with 4-phase progression.

    Phase transitions based on quantitative criteria, not gut feel.
    """

    def __init__(self):
        self.phase = CrisisPhase.NORMAL
        self.entry_date = None
        self.history = []

    def check_indicators(self, state: dict) -> dict:
        """
        Evaluate crisis indicators from market state.

        Args:
            state: Dict with vix, avg_correlation, equity_drop_pct,
                   btc_drop_7d_pct.

        Returns:
            Dict with active indicators.
        """
        indicators = {
            "vix_high": state.get("vix", 0) > 30,
            "correlation_spike": state.get("avg_correlation", 0) > 0.6,
            "equity_dropped_5pct": state.get("equity_drop_pct", 0) < -0.05,
            "btc_dropped_15pct": state.get("btc_drop_7d_pct", 0) < -0.15,
            "multiple_strategies_dd": state.get("strategies_in_dd", 0) >= 2,
        }
        indicators["count"] = sum(1 for k, v in indicators.items() if k != "count" and v)
        indicators["any_active"] = indicators["count"] >= 1
        indicators["severe"] = indicators["count"] >= 3
        return indicators

    def transition(self, state: dict) -> dict:
        """
        Evaluate phase transition based on current state.

        Returns:
            Dict with new_phase, action_plan, reasoning.
        """
        indicators = self.check_indicators(state)
        old_phase = self.phase
        action_plan = []
        reasoning = ""

        if self.phase == CrisisPhase.NORMAL:
            if indicators["any_active"]:
                self.phase = CrisisPhase.DETECTION
                self.entry_date = datetime.now()
                action_plan = [
                    "HALT new entries across all strategies",
                    "Reduce all open positions 50%",
                    "Cancel all pending orders",
                    "Update risk dashboard hourly",
                    "Check correlations every 4 hours",
                ]
                reasoning = (
                    f"Crisis indicator(s) detected: {indicators['count']} active"
                )

        elif self.phase == CrisisPhase.DETECTION:
            days_in_phase = (
                (datetime.now() - self.entry_date).days
                if self.entry_date else 0
            )
            if days_in_phase >= 3:
                self.phase = CrisisPhase.STABILIZATION
                self.entry_date = datetime.now()
                action_plan = [
                    "Move to 70% cash, 30% reduced positions",
                    "Cancel all pending orders",
                    "Watch VIX, correlations daily",
                    "Document lessons (what worked, what failed)",
                    "Continue daily check-ins, no trading impulse",
                ]
                reasoning = "Crisis sustained 3+ days, escalating to stabilization"
            elif not indicators["any_active"]:
                # False alarm, return to normal
                self.phase = CrisisPhase.NORMAL
                action_plan = ["Resume normal operations"]
                reasoning = "Indicators cleared, false alarm"

        elif self.phase == CrisisPhase.STABILIZATION:
            # Stabilization waits for VIX < 25 sustained 1 week + correlations < 0.4
            vix_stable = state.get("vix", 100) < 25
            corr_stable = state.get("avg_correlation", 1.0) < 0.4
            days_stable = state.get("days_since_calm", 0)

            if vix_stable and corr_stable and days_stable >= 7:
                self.phase = CrisisPhase.ASSESSMENT
                self.entry_date = datetime.now()
                action_plan = [
                    "Review strategy performance during crisis",
                    "Identify which strategies blew up",
                    "Decide which to keep, retire, or refactor",
                    "Run stress test on remaining portfolio",
                    "Plan reentry size and timing",
                ]
                reasoning = "Markets stabilized 1+ week, ready for assessment"

        elif self.phase == CrisisPhase.ASSESSMENT:
            # Assessment phase: 2-4 weeks of analysis
            days_in_phase = (
                (datetime.now() - self.entry_date).days
                if self.entry_date else 0
            )
            if days_in_phase >= 14:
                self.phase = CrisisPhase.REENTRY
                self.entry_date = datetime.now()
                action_plan = [
                    "Resume trading with 25% original size",
                    "Only proven strategies, skip experimental ones",
                    "Daily monitoring, weekly review",
                    "Scale to 50% after 2 weeks profitable",
                    "Scale to 100% after 4 weeks profitable",
                ]
                reasoning = "Assessment complete, beginning gradual reentry"

        elif self.phase == CrisisPhase.REENTRY:
            # Reentry: scale up gradually based on performance
            days_in_phase = (
                (datetime.now() - self.entry_date).days
                if self.entry_date else 0
            )
            recent_pnl = state.get("recent_pnl_pct", 0)
            if days_in_phase >= 28 and recent_pnl > 0.05:
                self.phase = CrisisPhase.NORMAL
                action_plan = [
                    "Resume full normal operations",
                    "Document lessons learned",
                    "Update risk parameters if needed",
                ]
                reasoning = "Reentry successful 4+ weeks, returning to normal"

        # Log transition
        if old_phase != self.phase:
            self.history.append({
                "timestamp": datetime.now(),
                "from_phase": old_phase.value,
                "to_phase": self.phase.value,
                "reasoning": reasoning,
                "action_plan": action_plan,
            })

        return {
            "old_phase": old_phase.value,
            "new_phase": self.phase.value,
            "transitioned": old_phase != self.phase,
            "action_plan": action_plan,
            "reasoning": reasoning,
            "indicators": indicators,
        }

    def status(self) -> dict:
        return {
            "current_phase": self.phase.value,
            "entry_date": self.entry_date,
            "n_transitions": len(self.history),
            "history": self.history[-3:],  # last 3 transitions
        }

    def report(self):
        print("=" * 70)
        print(f"CRISIS PLAYBOOK STATUS")
        print("=" * 70)
        print(f"Current phase: {self.phase.value.upper()}")
        if self.entry_date:
            days = (datetime.now() - self.entry_date).days
            print(f"Days in phase: {days}")
        if self.history:
            print(f"\nTransition history (last 3):")
            for t in self.history[-3:]:
                print(
                    f"  {t['timestamp'].strftime('%Y-%m-%d')}: "
                    f"{t['from_phase']} → {t['to_phase']}"
                )
                print(f"    Reason: {t['reasoning']}")


if __name__ == "__main__":
    print("=" * 70)
    print("Crisis Playbook — Demo (simulating COVID-style crisis)")
    print("=" * 70)

    pb = CrisisPlaybook()
    pb.report()

    # Simulate crisis progression
    scenarios = [
        ("Day 0 — Normal market", {
            "vix": 18, "avg_correlation": 0.2,
            "equity_drop_pct": 0.001, "btc_drop_7d_pct": 0.02,
            "strategies_in_dd": 0,
        }),
        ("Day 1 — Vol spike begins", {
            "vix": 35, "avg_correlation": 0.55,
            "equity_drop_pct": -0.04, "btc_drop_7d_pct": -0.08,
            "strategies_in_dd": 1,
        }),
        ("Day 4 — Sustained crisis", {
            "vix": 60, "avg_correlation": 0.75,
            "equity_drop_pct": -0.10, "btc_drop_7d_pct": -0.30,
            "strategies_in_dd": 3,
        }),
        ("Day 30 — Stabilizing", {
            "vix": 22, "avg_correlation": 0.35,
            "equity_drop_pct": 0.005, "btc_drop_7d_pct": 0.01,
            "strategies_in_dd": 1, "days_since_calm": 8,
        }),
        ("Day 60 — Assessment done", {
            "vix": 18, "avg_correlation": 0.25,
            "equity_drop_pct": 0.003, "btc_drop_7d_pct": 0.02,
            "strategies_in_dd": 0, "recent_pnl_pct": 0.06,
        }),
    ]

    for label, state in scenarios:
        # Hack: modify entry_date to simulate time passage
        if pb.entry_date and "Day" in label:
            day_n = int(label.split("Day ")[1].split(" ")[0])
            from datetime import timedelta
            if day_n >= 4 and pb.phase == CrisisPhase.DETECTION:
                pb.entry_date = datetime.now() - timedelta(days=4)
            elif day_n >= 30 and pb.phase == CrisisPhase.STABILIZATION:
                pb.entry_date = datetime.now() - timedelta(days=20)
            elif day_n >= 60 and pb.phase == CrisisPhase.ASSESSMENT:
                pb.entry_date = datetime.now() - timedelta(days=15)

        print(f"\n--- {label} ---")
        result = pb.transition(state)
        print(f"  Phase: {result['old_phase']} → {result['new_phase']}")
        if result["transitioned"]:
            print(f"  Reasoning: {result['reasoning']}")
            print(f"  Action plan:")
            for action in result["action_plan"]:
                print(f"    • {action}")
        else:
            print(f"  No transition (still in {result['new_phase']})")

    print(f"\n{pb.report() or ''}")
