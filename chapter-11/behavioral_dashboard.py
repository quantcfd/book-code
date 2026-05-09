"""
QuantCFD - Chapter 11
behavioral_dashboard.py - Comprehensive behavioral metrics dashboard

Combines TradeJournal + EmotionTracker + BiasDetector + HabitTracker
to single dashboard view with HTML output capability.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List

import pandas as pd

from trade_journal import TradeJournal
from emotion_tracker import EmotionTracker
from bias_detector import BiasDetector
from habit_tracker import HabitTracker
from post_trade_review import PostTradeReview
from decision_quality import DecisionQuality


class BehavioralDashboard:
    """
    Comprehensive behavioral metrics dashboard.

    Aggregates data từ multiple tracking systems for single view.
    """

    def __init__(
        self,
        trade_journal: Optional[TradeJournal] = None,
        emotion_tracker: Optional[EmotionTracker] = None,
        habit_tracker: Optional[HabitTracker] = None,
        post_trade_review: Optional[PostTradeReview] = None,
        decision_quality: Optional[DecisionQuality] = None,
        backtest_avg_win: float = 1.5,
        backtest_avg_loss: float = -1.0,
        backtest_win_rate: float = 0.50,
    ):
        self.trade_journal = trade_journal
        self.emotion_tracker = emotion_tracker
        self.habit_tracker = habit_tracker
        self.post_trade_review = post_trade_review
        self.decision_quality = decision_quality
        self.backtest_avg_win = backtest_avg_win
        self.backtest_avg_loss = backtest_avg_loss
        self.backtest_win_rate = backtest_win_rate

    def overall_health_score(self) -> float:
        """
        Compute overall behavioral health score (0-100).

        Combines: trade metrics, emotional stability, habit consistency, bias profile.
        """
        score_components = []

        # Trade journal: rule adherence + profitability
        if self.trade_journal:
            metrics = self.trade_journal.metrics()
            if metrics.get("n_trades", 0) > 0:
                rule_score = metrics["rule_adherence_pct"]  # 0-100
                profit_score = 50 + min(50, metrics["total_pnl"] / 100)  # cap at $5k profit
                score_components.append(rule_score)
                score_components.append(max(0, profit_score))

        # Emotion tracker: composite score
        if self.emotion_tracker and self.emotion_tracker.readings:
            avg = self.emotion_tracker.daily_average()
            if "composite" in avg:
                score_components.append(avg["composite"] * 10)  # 0-10 → 0-100

        # Habit tracker: identity strength average
        if self.habit_tracker:
            identity_scores = [
                self.habit_tracker.identity_strength(name)
                for name in self.habit_tracker.habits
            ]
            if identity_scores:
                score_components.append(sum(identity_scores) / len(identity_scores))

        # Bias detector: inverse severity (less severity = higher score)
        if self.trade_journal and len(self.trade_journal.entries) >= 10:
            df = self.trade_journal.to_dataframe()
            if not df.empty:
                detector = BiasDetector(
                    df,
                    backtest_avg_win=self.backtest_avg_win,
                    backtest_avg_loss=self.backtest_avg_loss,
                    backtest_win_rate=self.backtest_win_rate,
                )
                findings = detector.detect_all_biases()
                severity_to_penalty = {
                    "NONE": 0, "MILD": 10, "MODERATE": 25, "SEVERE": 50,
                }
                total_penalty = sum(severity_to_penalty.get(f.severity, 0) for f in findings)
                bias_score = max(0, 100 - total_penalty)
                score_components.append(bias_score)

        if not score_components:
            return 0.0

        return float(sum(score_components) / len(score_components))

    def status_summary(self) -> Dict[str, str]:
        """Quick status summary for each tracked area."""
        status = {}

        # Trade quality
        if self.trade_journal:
            metrics = self.trade_journal.metrics()
            n = metrics.get("n_trades", 0)
            if n == 0:
                status["trade_quality"] = "No trades logged"
            elif metrics.get("rule_adherence_pct", 0) > 90:
                status["trade_quality"] = "✓ High rule adherence"
            elif metrics.get("rule_adherence_pct", 0) > 75:
                status["trade_quality"] = "⚠ Rule adherence improving"
            else:
                status["trade_quality"] = "✗ Low rule adherence"

        # Emotional state
        if self.emotion_tracker and self.emotion_tracker.readings:
            latest = self.emotion_tracker.latest_reading()
            if latest:
                status["emotional_state"] = latest.trading_recommendation

        # Habit consistency
        if self.habit_tracker:
            avg_identity = sum(
                self.habit_tracker.identity_strength(name)
                for name in self.habit_tracker.habits
            ) / len(self.habit_tracker.habits)
            if avg_identity > 70:
                status["habits"] = "✓ Strong habit foundation"
            elif avg_identity > 50:
                status["habits"] = "⚠ Habits forming"
            else:
                status["habits"] = "✗ Habits inconsistent"

        # Bias profile
        if self.trade_journal and len(self.trade_journal.entries) >= 10:
            df = self.trade_journal.to_dataframe()
            if not df.empty:
                detector = BiasDetector(
                    df,
                    backtest_avg_win=self.backtest_avg_win,
                    backtest_avg_loss=self.backtest_avg_loss,
                    backtest_win_rate=self.backtest_win_rate,
                )
                findings = detector.detect_all_biases()
                severe_count = sum(1 for f in findings if f.severity == "SEVERE")
                moderate_count = sum(1 for f in findings if f.severity == "MODERATE")
                if severe_count > 0:
                    status["bias_profile"] = f"✗ {severe_count} severe biases active"
                elif moderate_count > 0:
                    status["bias_profile"] = f"⚠ {moderate_count} moderate biases"
                else:
                    status["bias_profile"] = "✓ Bias profile healthy"

        return status

    def report(self) -> str:
        """Generate comprehensive text report."""
        lines = ["=" * 80, "BEHAVIORAL DASHBOARD - COMPREHENSIVE REPORT", "=" * 80]
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Overall health score
        health = self.overall_health_score()
        lines.append(f"\nOVERALL BEHAVIORAL HEALTH SCORE: {health:.0f}/100")
        if health >= 80:
            lines.append("  Status: ✓ EXCELLENT - sustainable trader profile")
        elif health >= 60:
            lines.append("  Status: ✓ GOOD - solid foundation, minor improvements")
        elif health >= 40:
            lines.append("  Status: ⚠ DEVELOPING - work needed on multiple areas")
        else:
            lines.append("  Status: ✗ CONCERNING - significant gaps to address")

        # Status summary
        status = self.status_summary()
        if status:
            lines.append(f"\n--- Status Summary ---")
            for area, msg in status.items():
                lines.append(f"  {area:25s}: {msg}")

        # Trade journal section
        if self.trade_journal and self.trade_journal.entries:
            lines.append(f"\n{'-' * 80}\nTRADE JOURNAL\n{'-' * 80}")
            lines.append(self.trade_journal.report())

        # Emotion tracker section
        if self.emotion_tracker and self.emotion_tracker.readings:
            lines.append(f"\n{'-' * 80}\nEMOTION TRACKING\n{'-' * 80}")
            lines.append(self.emotion_tracker.report())

        # Habit tracker section
        if self.habit_tracker:
            lines.append(f"\n{'-' * 80}\nHABIT TRACKING\n{'-' * 80}")
            lines.append(self.habit_tracker.report())

        # Post-trade review
        if self.post_trade_review and self.post_trade_review.reviews:
            lines.append(f"\n{'-' * 80}\nPOST-TRADE REVIEW\n{'-' * 80}")
            lines.append(self.post_trade_review.report())

        # Decision quality
        if self.decision_quality and self.decision_quality.decisions:
            lines.append(f"\n{'-' * 80}\nDECISION QUALITY\n{'-' * 80}")
            lines.append(self.decision_quality.report())

        # Bias detection
        if self.trade_journal and len(self.trade_journal.entries) >= 10:
            df = self.trade_journal.to_dataframe()
            if not df.empty:
                lines.append(f"\n{'-' * 80}\nBIAS DETECTION\n{'-' * 80}")
                detector = BiasDetector(
                    df,
                    backtest_avg_win=self.backtest_avg_win,
                    backtest_avg_loss=self.backtest_avg_loss,
                    backtest_win_rate=self.backtest_win_rate,
                )
                lines.append(detector.report())

        lines.append("\n" + "=" * 80)
        lines.append("END OF DASHBOARD REPORT")
        lines.append("=" * 80)

        return "\n".join(lines)

    def to_html(self, output_path: str = "behavioral_dashboard.html") -> str:
        """Generate HTML report."""
        health = self.overall_health_score()
        status = self.status_summary()

        # Determine health color
        if health >= 80:
            health_color = "#5CB85C"
        elif health >= 60:
            health_color = "#0E7C7B"
        elif health >= 40:
            health_color = "#F0AD4E"
        else:
            health_color = "#D9534F"

        html_parts = [
            "<!DOCTYPE html><html><head>",
            '<meta charset="utf-8">',
            "<title>QuantCFD Behavioral Dashboard</title>",
            "<style>",
            "body { font-family: -apple-system, sans-serif; margin: 30px; "
            "background: #f5f5f5; color: #222; }",
            "h1, h2 { color: #0F1F3D; }",
            "h1 { border-bottom: 3px solid #0E7C7B; padding-bottom: 8px; }",
            ".card { background: white; padding: 20px; margin: 15px 0; "
            "border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
            ".health-score { font-size: 64px; font-weight: bold; text-align: center; }",
            ".status-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }",
            ".status-item { padding: 12px; background: #f8f8f8; border-radius: 4px; }",
            ".status-label { font-size: 11px; color: #666; text-transform: uppercase; }",
            ".status-value { font-size: 14px; font-weight: bold; }",
            "pre { background: #f4f4f4; padding: 15px; border-radius: 4px; "
            "overflow-x: auto; font-family: Consolas, monospace; font-size: 12px; }",
            "</style></head><body>",
            "<h1>QuantCFD Behavioral Dashboard</h1>",
            f'<p style="color:#666;">Generated: '
            f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>',
        ]

        # Health score card
        html_parts.append('<div class="card">')
        html_parts.append(
            f'<div class="health-score" style="color: {health_color};">'
            f'{health:.0f}/100</div>'
        )
        html_parts.append(
            '<p style="text-align:center; color:#666; margin-top:0;">'
            'Overall Behavioral Health Score</p>'
        )
        html_parts.append('</div>')

        # Status grid
        if status:
            html_parts.append('<div class="card"><h2>Status Summary</h2>')
            html_parts.append('<div class="status-grid">')
            for area, msg in status.items():
                html_parts.append(
                    f'<div class="status-item">'
                    f'<div class="status-label">{area.replace("_", " ").title()}</div>'
                    f'<div class="status-value">{msg}</div>'
                    f'</div>'
                )
            html_parts.append('</div></div>')

        # Detailed report as preformatted text
        html_parts.append('<div class="card"><h2>Detailed Report</h2>')
        html_parts.append(f'<pre>{self.report()}</pre>')
        html_parts.append('</div>')

        html_parts.append('</body></html>')

        full_html = "\n".join(html_parts)
        with open(output_path, "w") as f:
            f.write(full_html)
        return output_path


if __name__ == "__main__":
    print("=" * 80)
    print("Behavioral Dashboard Demo")
    print("=" * 80)

    # Build all components
    from trade_journal import generate_synthetic_journal
    from emotion_tracker import EmotionTracker
    from habit_tracker import HabitTracker
    from post_trade_review import PostTradeReview
    import numpy as np

    # 1. Trade journal
    journal = generate_synthetic_journal(n_trades=80, seed=42)

    # 2. Emotion tracker
    rng = np.random.default_rng(42)
    tracker = EmotionTracker()
    base = datetime(2024, 1, 1, 9, 0)
    for day in range(30):
        for hour in [9, 12, 17]:
            tracker.log(
                stress=int(np.clip(rng.normal(5, 1.5), 1, 10)),
                confidence=int(np.clip(rng.normal(6, 1.0), 1, 10)),
                patience=int(np.clip(rng.normal(6, 1.5), 1, 10)),
                focus=int(np.clip(rng.normal(7, 1.0), 1, 10)),
                energy=int(np.clip(rng.normal(6, 1.5), 1, 10)),
                timestamp=base + pd.Timedelta(days=day, hours=hour - 9),
            )

    # 3. Habit tracker
    from datetime import date, timedelta
    import random
    random.seed(42)
    habits = HabitTracker()
    today = date.today()
    for i in range(60):
        d = today - timedelta(days=i)
        if random.random() < 0.92:
            habits.log_habit("journal", d)
        if random.random() < 0.88:
            habits.log_habit("checklist", d)
        if d.weekday() == 5 and random.random() < 0.85:
            habits.log_habit("weekly_review", d)
        if d.weekday() == 1 and random.random() < 0.80:
            habits.log_habit("mentor_call", d)
        if random.random() < 0.65:
            habits.log_habit("exercise", d)

    # Build dashboard
    dashboard = BehavioralDashboard(
        trade_journal=journal,
        emotion_tracker=tracker,
        habit_tracker=habits,
        backtest_avg_win=1.5,
        backtest_avg_loss=-1.0,
    )

    # Show summary first
    health = dashboard.overall_health_score()
    print(f"\nOverall Health Score: {health:.1f}/100")

    status = dashboard.status_summary()
    print(f"\nStatus Summary:")
    for area, msg in status.items():
        print(f"  {area:25s}: {msg}")

    # Generate HTML
    html_path = dashboard.to_html("/tmp/behavioral_dashboard.html")
    print(f"\n✓ HTML report saved: {html_path}")
