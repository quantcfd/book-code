"""
QuantCFD - Chapter 11
post_trade_review.py - Post-trade review workflow with mistake/noise classification

Per Annie Duke: separate process quality from outcome quality.
Mistake = process violation. Noise = process correct but unlucky.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class TradeReview:
    """Single post-trade review record."""
    trade_id: str
    review_time: datetime

    # Outcome
    pnl: float
    r_multiple: float
    won: bool

    # Process review
    followed_rules: bool
    rule_violations: List[str] = field(default_factory=list)
    why_violated: str = ""

    # Classification
    is_mistake: bool = False
    is_noise: bool = False

    # Reflection
    what_went_right: str = ""
    what_could_improve: str = ""
    lesson_learned: str = ""
    emotional_state_during: str = ""

    @property
    def review_quality(self) -> str:
        """Classify review based on Annie Duke framework."""
        if self.followed_rules and self.won:
            return "Skill"
        elif self.followed_rules and not self.won:
            return "Noise"  # Bad luck
        elif not self.followed_rules and self.won:
            return "Lucky"  # Bad decision, lucky outcome
        else:
            return "Mistake"  # Bad decision, bad outcome


class PostTradeReview:
    """
    Post-trade review workflow.

    Each trade gets structured review separating process from outcome.
    Tracks patterns over time.
    """

    def __init__(self):
        self.reviews: List[TradeReview] = []

    def review_trade(
        self,
        trade_id: str,
        pnl: float,
        r_multiple: float,
        followed_rules: bool,
        rule_violations: Optional[List[str]] = None,
        what_went_right: str = "",
        what_could_improve: str = "",
        lesson_learned: str = "",
        emotional_state_during: str = "",
        why_violated: str = "",
    ) -> TradeReview:
        """Conduct post-trade review."""
        review = TradeReview(
            trade_id=trade_id,
            review_time=datetime.now(),
            pnl=pnl,
            r_multiple=r_multiple,
            won=pnl > 0,
            followed_rules=followed_rules,
            rule_violations=rule_violations or [],
            why_violated=why_violated,
            is_mistake=not followed_rules,
            is_noise=followed_rules and pnl <= 0,
            what_went_right=what_went_right,
            what_could_improve=what_could_improve,
            lesson_learned=lesson_learned,
            emotional_state_during=emotional_state_during,
        )
        self.reviews.append(review)
        return review

    def mistake_rate(self) -> float:
        """Pct of trades classified as mistakes (process violations)."""
        if not self.reviews:
            return 0.0
        return sum(1 for r in self.reviews if r.is_mistake) / len(self.reviews)

    def noise_rate(self) -> float:
        """Pct of trades classified as noise (process correct, unlucky)."""
        if not self.reviews:
            return 0.0
        return sum(1 for r in self.reviews if r.is_noise) / len(self.reviews)

    def common_violations(self) -> Dict[str, int]:
        """Identify most common rule violations."""
        violation_counts: Dict[str, int] = {}
        for review in self.reviews:
            for v in review.rule_violations:
                violation_counts[v] = violation_counts.get(v, 0) + 1
        # Sort by count
        return dict(sorted(violation_counts.items(), key=lambda x: x[1], reverse=True))

    def violation_cost(self) -> float:
        """Total P&L cost of rule violations (mistakes only)."""
        return sum(r.pnl for r in self.reviews if r.is_mistake and r.pnl < 0)

    def quadrant_summary(self) -> Dict[str, int]:
        """Count reviews in each quadrant."""
        result = {"Skill": 0, "Noise": 0, "Lucky": 0, "Mistake": 0}
        for r in self.reviews:
            result[r.review_quality] += 1
        return result

    def report(self) -> str:
        """Generate post-trade review report."""
        lines = ["=" * 70, "POST-TRADE REVIEW REPORT", "=" * 70]

        n = len(self.reviews)
        if n == 0:
            return "\n".join(lines + ["\nNo reviews recorded."])

        lines.append(f"\nTotal reviews: {n}")
        lines.append(f"Mistake rate: {self.mistake_rate()*100:.1f}%")
        lines.append(f"Noise rate: {self.noise_rate()*100:.1f}%")
        lines.append(f"Cost of mistakes: ${abs(self.violation_cost()):,.2f}")

        # Quadrants
        q = self.quadrant_summary()
        lines.append(f"\n--- Quadrant Summary ---")
        lines.append(f"  Skill (rules+win):     {q['Skill']:3d} ({q['Skill']/n*100:.0f}%)")
        lines.append(f"  Noise (rules+loss):    {q['Noise']:3d} ({q['Noise']/n*100:.0f}%)")
        lines.append(f"  Lucky (no rules+win):  {q['Lucky']:3d} ({q['Lucky']/n*100:.0f}%) ⚠")
        lines.append(f"  Mistake (no rules+loss): {q['Mistake']:3d} ({q['Mistake']/n*100:.0f}%)")

        # Common violations
        violations = self.common_violations()
        if violations:
            lines.append(f"\n--- Most Common Rule Violations ---")
            for v, count in list(violations.items())[:5]:
                lines.append(f"  {v}: {count} times")

        # Recommendations
        lines.append(f"\n--- Recommendations ---")
        if self.mistake_rate() > 0.20:
            lines.append("  ⚠ High mistake rate (>20%). Review rule enforcement.")
        if q.get("Lucky", 0) / n > 0.10:
            lines.append("  ⚠ High 'Lucky' rate. Bad decisions winning. Reinforces poor process.")
        if self.mistake_rate() < 0.10:
            lines.append("  ✓ Mistake rate acceptable.")

        lines.append("=" * 70)
        return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 70)
    print("Post-Trade Review Demo")
    print("=" * 70)

    review = PostTradeReview()

    # Mock 30 trade reviews với mix of outcomes
    import random
    random.seed(42)

    common_violations = [
        "Moved stop to breakeven prematurely",
        "Skipped pre-trade checklist",
        "Increased position size after win",
        "Held loser past stop loss",
        "FOMO entry without setup",
    ]

    for i in range(30):
        followed = random.random() > 0.20  # 80% follow rules
        pnl = random.choice([200, 200, -200]) if followed else random.choice([200, -300, -300])
        rule_violations = (
            [] if followed else [random.choice(common_violations)]
        )

        review.review_trade(
            trade_id=f"T-{i+1:03d}",
            pnl=pnl,
            r_multiple=pnl / 200,
            followed_rules=followed,
            rule_violations=rule_violations,
            what_went_right="Identified setup correctly" if followed else "",
            what_could_improve="" if followed else "Stay disciplined",
            why_violated="Frustration from prior losses" if not followed else "",
        )

    print(review.report())
