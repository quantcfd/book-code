"""
QuantCFD - Chapter 11
decision_quality.py - Annie Duke decision quality framework

Decision quality vs outcome quality (4-quadrant classification).
Resulting fallacy detection. Probabilistic thinking. Calibration tracking.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class Decision:
    """Single decision record with reasoning + outcome."""
    decision_id: str
    timestamp: datetime
    description: str

    # At decision time
    information_available: str
    win_probability_estimate: float  # 0-1
    expected_value_positive: bool
    alternatives_considered: List[str] = field(default_factory=list)
    reasoning: str = ""

    # Process quality (set at decision time, before outcome)
    followed_system: bool = True
    setup_grade: str = "A"

    # Outcome (set after)
    won: Optional[bool] = None
    outcome_pnl: Optional[float] = None
    outcome_recorded_at: Optional[datetime] = None

    # Retrospective evaluation (set after, separate from outcome)
    decision_was_good_in_retrospect: Optional[bool] = None
    would_decide_same_again: Optional[bool] = None

    @property
    def quadrant(self) -> str:
        """Classify into 4-quadrant matrix."""
        if self.decision_was_good_in_retrospect is None or self.won is None:
            return "Pending"

        good_decision = self.decision_was_good_in_retrospect
        good_outcome = self.won

        if good_decision and good_outcome:
            return "Skill"  # Good decision + good outcome
        elif good_decision and not good_outcome:
            return "Bad luck"  # Good decision + bad outcome (acceptable)
        elif not good_decision and good_outcome:
            return "Lucky"  # Bad decision + good outcome (DANGEROUS - reinforces bad)
        else:
            return "Just deserts"  # Bad decision + bad outcome (learn from)


class DecisionQuality:
    """
    Track decisions với separate quality and outcome metrics.

    Implements Annie Duke (Thinking in Bets) framework:
    - Decision quality ≠ outcome quality
    - Calibration of probability estimates
    - Quadrant classification
    """

    def __init__(self):
        self.decisions: List[Decision] = []

    def add_decision(self, decision: Decision) -> None:
        """Record a decision (before outcome known)."""
        self.decisions.append(decision)

    def record_outcome(
        self,
        decision_id: str,
        won: bool,
        outcome_pnl: float,
        decision_was_good_in_retrospect: bool,
    ) -> None:
        """Record outcome and retrospective decision quality."""
        for d in self.decisions:
            if d.decision_id == decision_id:
                d.won = won
                d.outcome_pnl = outcome_pnl
                d.outcome_recorded_at = datetime.now()
                d.decision_was_good_in_retrospect = decision_was_good_in_retrospect
                d.would_decide_same_again = decision_was_good_in_retrospect
                return
        raise ValueError(f"Decision {decision_id} not found")

    def quadrant_distribution(self) -> Dict[str, int]:
        """Count decisions in each quadrant."""
        counts = {"Skill": 0, "Bad luck": 0, "Lucky": 0, "Just deserts": 0, "Pending": 0}
        for d in self.decisions:
            counts[d.quadrant] = counts.get(d.quadrant, 0) + 1
        return counts

    def calibration_check(self) -> pd.DataFrame:
        """
        Check probability calibration.

        For trades you said had X% probability, what was actual win rate?
        Well-calibrated = predicted ≈ actual.
        Overconfident = predicted > actual.
        """
        completed = [d for d in self.decisions if d.won is not None]
        if len(completed) < 10:
            return pd.DataFrame()

        # Bucket probabilities into bins
        df = pd.DataFrame([
            {
                "predicted_prob": d.win_probability_estimate,
                "actual_win": d.won,
            }
            for d in completed
        ])

        # Round to nearest 10% bin
        df["bucket"] = (df["predicted_prob"] * 10).round() / 10

        result = df.groupby("bucket").agg(
            n_trades=("actual_win", "count"),
            actual_win_rate=("actual_win", "mean"),
        )
        result["predicted_prob"] = result.index
        result["calibration_error"] = result["actual_win_rate"] - result["predicted_prob"]
        return result

    def overall_calibration_score(self) -> float:
        """
        Overall calibration score (0-100).

        Penalizes systematic overconfidence or underconfidence.
        100 = perfect calibration.
        """
        cal_df = self.calibration_check()
        if cal_df.empty:
            return 0.0

        # Weighted by sample size
        total_weighted_error = (cal_df["calibration_error"].abs() * cal_df["n_trades"]).sum()
        total_trades = cal_df["n_trades"].sum()

        if total_trades == 0:
            return 0.0

        avg_abs_error = total_weighted_error / total_trades

        # Convert to score: 0 error = 100, 0.5 error = 0
        score = max(0, 100 - 200 * avg_abs_error)
        return float(score)

    def resulting_fallacy_check(self) -> List[str]:
        """
        Detect resulting fallacy patterns.

        Resulting = judging decision purely by outcome.
        Pattern: many "Lucky" outcomes celebrated as skill.
        """
        warnings = []
        dist = self.quadrant_distribution()

        n_decisions = sum(dist[q] for q in ("Skill", "Bad luck", "Lucky", "Just deserts"))
        if n_decisions < 10:
            return ["Insufficient decisions for analysis (need 10+)"]

        # Check 1: too many "Lucky" outcomes
        lucky_pct = dist["Lucky"] / n_decisions
        if lucky_pct > 0.20:
            warnings.append(
                f"⚠ High 'Lucky' rate: {lucky_pct:.0%}. "
                "Bad decisions winning often. Reinforces poor process. "
                "Review: are you overriding system rules and getting away with it?"
            )

        # Check 2: too few "Bad luck" cases
        bad_luck_pct = dist["Bad luck"] / n_decisions
        if bad_luck_pct < 0.10:
            warnings.append(
                f"⚠ Low 'Bad luck' rate: {bad_luck_pct:.0%}. "
                "Possibly attributing all losses to bad decisions. "
                "Some losses are legitimate noise — accept them."
            )

        # Check 3: heavy "Just deserts"
        just_deserts_pct = dist["Just deserts"] / n_decisions
        if just_deserts_pct > 0.30:
            warnings.append(
                f"⚠ High 'Just deserts' rate: {just_deserts_pct:.0%}. "
                "Frequent bad decisions leading to losses. "
                "Review: identify common rule violations."
            )

        if not warnings:
            warnings.append("✓ Decision quality distribution healthy")

        return warnings

    def report(self) -> str:
        """Generate decision quality report."""
        lines = ["=" * 70, "DECISION QUALITY REPORT (Annie Duke Framework)", "=" * 70]

        n_total = len(self.decisions)
        n_completed = sum(1 for d in self.decisions if d.won is not None)
        lines.append(f"\nTotal decisions: {n_total}")
        lines.append(f"With outcomes: {n_completed}")

        if n_completed == 0:
            lines.append("\nNo completed decisions yet.")
            return "\n".join(lines)

        # Quadrant distribution
        dist = self.quadrant_distribution()
        lines.append(f"\n--- Quadrant Distribution ---")
        lines.append(f"  Skill (good decision + good outcome):       {dist['Skill']:3d} ({dist['Skill']/n_completed*100:.1f}%)")
        lines.append(f"  Bad luck (good decision + bad outcome):     {dist['Bad luck']:3d} ({dist['Bad luck']/n_completed*100:.1f}%)")
        lines.append(f"  Lucky (bad decision + good outcome):        {dist['Lucky']:3d} ({dist['Lucky']/n_completed*100:.1f}%) ⚠")
        lines.append(f"  Just deserts (bad decision + bad outcome):  {dist['Just deserts']:3d} ({dist['Just deserts']/n_completed*100:.1f}%)")

        # Decision quality rate (good decisions / total)
        good_decisions = dist["Skill"] + dist["Bad luck"]
        decision_quality_rate = good_decisions / n_completed
        lines.append(f"\nDecision quality rate: {decision_quality_rate:.1%}")
        lines.append(f"  (good decisions independent of outcome)")

        # Calibration
        cal_df = self.calibration_check()
        if not cal_df.empty:
            cal_score = self.overall_calibration_score()
            lines.append(f"\n--- Calibration Analysis ---")
            lines.append(f"Calibration score: {cal_score:.0f}/100")
            lines.append("Predicted vs actual win rate by bucket:")
            for _, row in cal_df.iterrows():
                err = row["calibration_error"]
                err_str = f"{err:+.2%}"
                lines.append(
                    f"  P={row['predicted_prob']:.0%}: predicted, "
                    f"actual {row['actual_win_rate']:.1%} ({row['n_trades']:.0f} trades, error: {err_str})"
                )

        # Resulting fallacy
        warnings = self.resulting_fallacy_check()
        lines.append(f"\n--- Resulting Fallacy Check ---")
        for w in warnings:
            lines.append(f"  {w}")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 70)
    print("Decision Quality Demo (Annie Duke Framework)")
    print("=" * 70)

    dq = DecisionQuality()
    rng = np.random.default_rng(42)

    # Generate 50 trade decisions với varying quality and outcomes
    base_time = datetime(2024, 1, 1)
    for i in range(50):
        prob_estimate = rng.uniform(0.4, 0.7)
        followed_system = rng.random() > 0.20  # 80% follow system

        decision = Decision(
            decision_id=f"D-{i+1:03d}",
            timestamp=base_time + pd.Timedelta(days=i),
            description=f"Trade decision {i+1}",
            information_available="Strategy signal + risk check",
            win_probability_estimate=prob_estimate,
            expected_value_positive=True,
            alternatives_considered=["Skip trade", "Smaller size"],
            reasoning="Setup grade A, all criteria met" if followed_system else "Override system",
            followed_system=followed_system,
            setup_grade="A" if followed_system else "C",
        )

        dq.add_decision(decision)

        # Outcome: actual win probability slightly worse than predicted (overconfidence)
        actual_win_prob = prob_estimate - 0.05
        won = rng.random() < actual_win_prob

        # Decision was good = followed system (independent of outcome)
        decision_was_good = followed_system

        outcome_pnl = 200 if won else -200

        dq.record_outcome(
            decision_id=decision.decision_id,
            won=won,
            outcome_pnl=outcome_pnl,
            decision_was_good_in_retrospect=decision_was_good,
        )

    print(dq.report())
