"""
QuantCFD - Chapter 11
integrated_psychology_system.py - Master orchestrator combining all components

Pre-trade workflow: emotion check -> bias check -> checklist
Post-trade workflow: log -> reflect -> pattern analysis
Weekly review automation. Mentor reporting. Identity tracking.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List

import pandas as pd

from trade_journal import TradeJournal, TradeEntry
from emotion_tracker import EmotionTracker, EmotionReading
from bias_detector import BiasDetector
from pre_trade_checklist import PreTradeChecklist, ChecklistResult
from habit_tracker import HabitTracker
from post_trade_review import PostTradeReview
from decision_quality import DecisionQuality, Decision
from mental_simulation import MentalSimulation
from behavioral_dashboard import BehavioralDashboard


@dataclass
class PreTradeWorkflowResult:
    """Result of integrated pre-trade workflow."""
    can_trade: bool
    emotion_check_passed: bool
    bias_check_passed: bool
    checklist_passed: bool
    blocking_reasons: List[str]
    warnings: List[str]
    timestamp: datetime


class IntegratedPsychologySystem:
    """
    Master orchestrator combining all Ch11 psychology components.

    Provides workflows:
    - Pre-trade: emotion -> bias -> checklist
    - Post-trade: log -> reflect -> pattern
    - Weekly review: aggregate all metrics
    - Mentor report: structured share format
    """

    def __init__(
        self,
        backtest_avg_win: float = 1.5,
        backtest_avg_loss: float = -1.0,
        backtest_win_rate: float = 0.50,
        max_risk_per_trade_pct: float = 0.01,
        daily_loss_limit_pct: float = 0.03,
    ):
        # Initialize all components
        self.trade_journal = TradeJournal()
        self.emotion_tracker = EmotionTracker()
        self.habit_tracker = HabitTracker()
        self.post_trade_review = PostTradeReview()
        self.decision_quality = DecisionQuality()
        self.mental_simulation = MentalSimulation()
        self.checklist = PreTradeChecklist(
            max_risk_per_trade_pct=max_risk_per_trade_pct,
            daily_loss_limit_pct=daily_loss_limit_pct,
        )

        self.backtest_avg_win = backtest_avg_win
        self.backtest_avg_loss = backtest_avg_loss
        self.backtest_win_rate = backtest_win_rate

        # Track recent trades intra-day
        self._daily_pnl_pct: float = 0.0
        self._consecutive_losses_today: int = 0

    def reset_daily_state(self) -> None:
        """Reset intra-day counters (call at market open each day)."""
        self._daily_pnl_pct = 0.0
        self._consecutive_losses_today = 0

    def pre_trade_workflow(
        self,
        # Setup info
        signal_triggered: bool,
        setup_grade: str,
        # Risk info
        position_size: float,
        risk_amount: float,
        current_equity: float,
        # Emotion (current reading)
        stress: int,
        confidence: int,
        patience: int,
        focus: int,
        energy: int,
        # Subjective state
        feeling_fomo: bool = False,
        feeling_revenge: bool = False,
        recent_big_win_or_loss: bool = False,
    ) -> PreTradeWorkflowResult:
        """
        Full pre-trade workflow: log emotion → check biases → run checklist.

        Returns PreTradeWorkflowResult with go/no-go decision.
        """
        blocking_reasons: List[str] = []
        warnings: List[str] = []

        # Step 1: Log current emotion reading
        reading = self.emotion_tracker.log(
            stress=stress,
            confidence=confidence,
            patience=patience,
            focus=focus,
            energy=energy,
            notes="pre-trade reading",
        )

        emotion_check_passed = not (
            reading.is_high_stress
            or reading.is_low_energy
            or reading.is_low_patience
        )

        if not emotion_check_passed:
            if reading.is_high_stress:
                blocking_reasons.append(f"Stress {stress}/10 too high")
            if reading.is_low_energy:
                blocking_reasons.append(f"Energy {energy}/10 too low")
            if reading.is_low_patience:
                blocking_reasons.append(f"Patience {patience}/10 too low")

        # Step 2: Bias check (if enough trade history)
        bias_check_passed = True
        if len(self.trade_journal.entries) >= 10:
            df = self.trade_journal.to_dataframe()
            if not df.empty:
                detector = BiasDetector(
                    df,
                    backtest_avg_win=self.backtest_avg_win,
                    backtest_avg_loss=self.backtest_avg_loss,
                    backtest_win_rate=self.backtest_win_rate,
                )
                findings = detector.detect_all_biases()
                severe = [f for f in findings if f.severity == "SEVERE"]
                moderate = [f for f in findings if f.severity == "MODERATE"]

                if severe:
                    bias_check_passed = False
                    for f in severe:
                        blocking_reasons.append(
                            f"SEVERE bias active: {f.bias_name}"
                        )
                if moderate:
                    for f in moderate:
                        warnings.append(f"Moderate bias: {f.bias_name}")

        # Step 3: Run pre-trade checklist
        checklist_result = self.checklist.check(
            signal_triggered=signal_triggered,
            all_entry_criteria_met=signal_triggered,
            setup_grade=setup_grade,
            position_size=position_size,
            risk_amount=risk_amount,
            current_equity=current_equity,
            daily_loss_so_far_pct=self._daily_pnl_pct,
            stress_level=stress,
            energy_level=energy,
            consecutive_losses_today=self._consecutive_losses_today,
            feeling_fomo=feeling_fomo,
            feeling_revenge=feeling_revenge,
            recent_big_win_or_loss=recent_big_win_or_loss,
        )

        if not checklist_result.passed:
            blocking_reasons.extend(checklist_result.failed_checks)
        warnings.extend(checklist_result.warnings)

        can_trade = (
            emotion_check_passed
            and bias_check_passed
            and checklist_result.passed
        )

        return PreTradeWorkflowResult(
            can_trade=can_trade,
            emotion_check_passed=emotion_check_passed,
            bias_check_passed=bias_check_passed,
            checklist_passed=checklist_result.passed,
            blocking_reasons=blocking_reasons,
            warnings=warnings,
            timestamp=datetime.now(),
        )

    def log_trade(self, entry: TradeEntry) -> None:
        """Log new trade to journal."""
        self.trade_journal.add_trade(entry)

    def post_trade_workflow(
        self,
        trade_id: str,
        exit_price: float,
        followed_rules: bool,
        rule_violations: Optional[List[str]] = None,
        what_went_right: str = "",
        what_could_improve: str = "",
        lesson_learned: str = "",
        emotional_state_during: str = "",
    ) -> Dict:
        """
        Post-trade workflow: close trade -> review -> update daily state.
        """
        # Find and close trade
        target_entry = None
        for e in self.trade_journal.entries:
            if e.trade_id == trade_id:
                target_entry = e
                break

        if target_entry is None:
            raise ValueError(f"Trade {trade_id} not found")

        target_entry.close(exit_price=exit_price)
        target_entry.followed_rules = followed_rules
        target_entry.is_mistake = not followed_rules
        target_entry.is_noise = followed_rules and (target_entry.pnl or 0) <= 0
        target_entry.what_went_right = what_went_right
        target_entry.what_could_improve = what_could_improve
        target_entry.lesson_learned = lesson_learned
        target_entry.emotional_state_during = emotional_state_during

        # Conduct review
        review = self.post_trade_review.review_trade(
            trade_id=trade_id,
            pnl=target_entry.pnl or 0,
            r_multiple=target_entry.r_multiple or 0,
            followed_rules=followed_rules,
            rule_violations=rule_violations,
            what_went_right=what_went_right,
            what_could_improve=what_could_improve,
            lesson_learned=lesson_learned,
            emotional_state_during=emotional_state_during,
        )

        # Update daily state
        if target_entry.pnl is not None and target_entry.pnl <= 0:
            self._consecutive_losses_today += 1
        else:
            self._consecutive_losses_today = 0

        return {
            "trade": target_entry,
            "review": review,
            "consecutive_losses_today": self._consecutive_losses_today,
            "review_quality": review.review_quality,
        }

    def update_daily_pnl(self, pnl_pct: float) -> None:
        """Update running daily P&L percentage."""
        self._daily_pnl_pct = pnl_pct

    def daily_summary(self) -> Dict:
        """Generate end-of-day summary."""
        df = self.trade_journal.to_dataframe()
        today = date.today()

        if df.empty or "date_entry" not in df.columns:
            return {"trades_today": 0}

        df["date_entry"] = pd.to_datetime(df["date_entry"])
        today_trades = df[df["date_entry"].dt.date == today]

        if today_trades.empty:
            return {"trades_today": 0, "date": today.isoformat()}

        closed = today_trades[today_trades["pnl"].notna()]

        # Latest emotion reading
        latest_emo = self.emotion_tracker.latest_reading()

        return {
            "date": today.isoformat(),
            "trades_today": int(len(today_trades)),
            "closed_today": int(len(closed)),
            "wins_today": int((closed["pnl"] > 0).sum()) if not closed.empty else 0,
            "net_pnl_today": float(closed["pnl"].sum()) if not closed.empty else 0.0,
            "rule_adherence_today": float(closed["followed_rules"].mean()) if not closed.empty else 1.0,
            "consecutive_losses": self._consecutive_losses_today,
            "latest_emotion_composite": latest_emo.composite_score if latest_emo else None,
            "latest_emotion_recommendation": latest_emo.trading_recommendation if latest_emo else None,
        }

    def weekly_review(self) -> str:
        """Automated weekly review."""
        lines = ["=" * 80]
        lines.append("WEEKLY AUTOMATED REVIEW")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        # Trade metrics
        if self.trade_journal.entries:
            df = self.trade_journal.to_dataframe()
            df["date_entry"] = pd.to_datetime(df["date_entry"])
            week_ago = datetime.now() - timedelta(days=7)
            this_week = df[df["date_entry"] >= week_ago]
            closed = this_week[this_week["pnl"].notna()] if not this_week.empty else this_week

            lines.append(f"\n--- This Week ---")
            lines.append(f"Trades: {len(this_week)}")
            if not closed.empty:
                wins = (closed["pnl"] > 0).sum()
                lines.append(f"Wins/Losses: {wins}/{len(closed) - wins}")
                lines.append(f"Win rate: {wins/len(closed)*100:.1f}%")
                lines.append(f"Net P&L: ${closed['pnl'].sum():,.2f}")
                lines.append(f"Rule adherence: {closed['followed_rules'].mean()*100:.1f}%")

        # Emotion trends
        if self.emotion_tracker.readings:
            trends = self.emotion_tracker.trends(days=7)
            if not trends.empty:
                lines.append(f"\n--- Emotion Trends (7 days) ---")
                lines.append(f"Avg stress: {trends['stress'].mean():.1f}/10")
                lines.append(f"Avg energy: {trends['energy'].mean():.1f}/10")
                lines.append(f"Avg composite: {trends['composite'].mean():.2f}/10")

        # Habit identity
        lines.append(f"\n--- Habit Identity Strength ---")
        for habit_name in self.habit_tracker.habits:
            strength = self.habit_tracker.identity_strength(habit_name)
            lines.append(f"  {habit_name:20s}: {strength:5.1f}/100")

        # Bias profile
        if len(self.trade_journal.entries) >= 10:
            df = self.trade_journal.to_dataframe()
            detector = BiasDetector(
                df,
                backtest_avg_win=self.backtest_avg_win,
                backtest_avg_loss=self.backtest_avg_loss,
            )
            findings = detector.detect_all_biases()
            lines.append(f"\n--- Bias Status ---")
            for f in findings:
                lines.append(f"  {f.bias_name:20s}: {f.severity}")

        # Identity tracking
        lines.append(f"\n--- Identity Metrics ---")
        avg_identity = sum(
            self.habit_tracker.identity_strength(name)
            for name in self.habit_tracker.habits
        ) / len(self.habit_tracker.habits)
        lines.append(f"Overall identity strength: {avg_identity:.0f}/100")

        # Recommendations
        lines.append(f"\n--- Action Items For Next Week ---")
        action_count = 0
        if avg_identity < 60:
            lines.append("  1. Strengthen habit consistency (focus on weakest habit)")
            action_count += 1
        if len(self.trade_journal.entries) >= 10:
            df = self.trade_journal.to_dataframe()
            detector = BiasDetector(df, backtest_avg_win=self.backtest_avg_win, backtest_avg_loss=self.backtest_avg_loss)
            severe = [f for f in detector.detect_all_biases() if f.severity == "SEVERE"]
            if severe:
                lines.append(f"  {action_count + 1}. Address severe bias: {severe[0].bias_name}")
                action_count += 1
        if action_count == 0:
            lines.append("  No critical actions. Maintain current trajectory.")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)

    def mentor_report(self) -> str:
        """Generate structured report cho mentor consultation."""
        lines = ["=" * 80]
        lines.append("MENTOR CONSULTATION REPORT")
        lines.append("=" * 80)
        lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        lines.append(f"Trader: [Insert Name]")

        # Performance summary
        metrics = self.trade_journal.metrics()
        lines.append(f"\n--- Performance Summary (last 30 trades) ---")
        if metrics.get("n_trades", 0) > 0:
            lines.append(f"  Trades: {metrics['n_trades']}")
            lines.append(f"  Win rate: {metrics.get('win_rate', 0)*100:.1f}%")
            lines.append(f"  Net P&L: ${metrics.get('total_pnl', 0):,.2f}")
            lines.append(f"  Avg R-multiple: {metrics.get('avg_r_multiple', 0):+.2f}")
            lines.append(f"  Rule adherence: {metrics.get('rule_adherence_pct', 0):.1f}%")
        else:
            lines.append("  No closed trades yet")

        # Issues to discuss
        lines.append(f"\n--- Issues To Discuss ---")
        issues = []
        if metrics.get("rule_adherence_pct", 100) < 90:
            issues.append("Rule adherence below 90% target")
        if len(self.trade_journal.entries) >= 10:
            df = self.trade_journal.to_dataframe()
            detector = BiasDetector(
                df,
                backtest_avg_win=self.backtest_avg_win,
                backtest_avg_loss=self.backtest_avg_loss,
            )
            severe = [f for f in detector.detect_all_biases() if f.severity == "SEVERE"]
            for f in severe:
                issues.append(f"Severe bias: {f.bias_name}")

        # Emotion concerns
        if self.emotion_tracker.readings:
            patterns = self.emotion_tracker.detect_patterns()
            for p in patterns:
                if "⚠" in p:
                    issues.append(p)

        if issues:
            for i, issue in enumerate(issues, 1):
                lines.append(f"  {i}. {issue}")
        else:
            lines.append("  No significant issues")

        # Wins to celebrate
        lines.append(f"\n--- Wins This Period ---")
        wins = []
        if metrics.get("rule_adherence_pct", 0) >= 90:
            wins.append("Rule adherence above 90%")
        if metrics.get("total_pnl", 0) > 0:
            wins.append("Profitable period")

        for w in wins:
            lines.append(f"  ✓ {w}")
        if not wins:
            lines.append("  Focus this period on building base, wins will come")

        # Questions for mentor
        lines.append(f"\n--- Questions ---")
        lines.append("  1. [Insert specific question]")
        lines.append("  2. [Insert specific question]")
        lines.append("  3. [Insert specific question]")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)

    def get_dashboard(self) -> BehavioralDashboard:
        """Get behavioral dashboard view of all components."""
        return BehavioralDashboard(
            trade_journal=self.trade_journal,
            emotion_tracker=self.emotion_tracker,
            habit_tracker=self.habit_tracker,
            post_trade_review=self.post_trade_review,
            decision_quality=self.decision_quality,
            backtest_avg_win=self.backtest_avg_win,
            backtest_avg_loss=self.backtest_avg_loss,
            backtest_win_rate=self.backtest_win_rate,
        )


if __name__ == "__main__":
    print("=" * 80)
    print("Integrated Psychology System Demo")
    print("=" * 80)

    system = IntegratedPsychologySystem(
        backtest_avg_win=1.5,
        backtest_avg_loss=-1.0,
        max_risk_per_trade_pct=0.01,
    )

    # Scenario 1: Optimal pre-trade workflow
    print("\n--- Pre-Trade Workflow Scenario 1: Optimal ---")
    result = system.pre_trade_workflow(
        signal_triggered=True,
        setup_grade="A",
        position_size=0.05,
        risk_amount=200,
        current_equity=20000,
        stress=3,
        confidence=6,
        patience=8,
        focus=8,
        energy=7,
    )
    print(f"Can trade: {result.can_trade}")
    print(f"  Emotion check: {result.emotion_check_passed}")
    print(f"  Bias check: {result.bias_check_passed}")
    print(f"  Checklist: {result.checklist_passed}")
    if result.blocking_reasons:
        print(f"  Blocking: {result.blocking_reasons}")

    # Scenario 2: High stress + revenge feeling
    print("\n--- Pre-Trade Workflow Scenario 2: High stress ---")
    result2 = system.pre_trade_workflow(
        signal_triggered=True,
        setup_grade="A",
        position_size=0.05,
        risk_amount=200,
        current_equity=20000,
        stress=8,  # too high
        confidence=4,
        patience=3,  # too low
        focus=5,
        energy=4,  # too low
        feeling_revenge=True,
    )
    print(f"Can trade: {result2.can_trade}")
    print(f"  Blocking reasons:")
    for r in result2.blocking_reasons:
        print(f"    - {r}")

    # Log a trade
    print("\n--- Logging a Trade ---")
    entry = TradeEntry(
        trade_id="DEMO-001",
        date_entry=datetime.now(),
        asset="XAUUSD",
        strategy="trend",
        entry_trigger="MA cross + ADX",
        confidence=4,
        setup_grade="A",
        pre_trade_emotion=3,
        entry_price=2015,
        stop_loss=1995,
        profit_target=2055,
        position_size=0.05,
        risk_amount=200,
    )
    system.log_trade(entry)

    # Post-trade workflow
    post_result = system.post_trade_workflow(
        trade_id="DEMO-001",
        exit_price=2055,
        followed_rules=True,
        what_went_right="Followed system signal, walked away",
        lesson_learned="Pre-defined exits work",
    )
    print(f"Trade closed: P&L = ${post_result['trade'].pnl:.2f}")
    print(f"Review quality: {post_result['review_quality']}")

    # Daily summary
    print("\n--- Daily Summary ---")
    summary = system.daily_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Mentor report
    print("\n")
    print(system.mentor_report())
