"""
QuantCFD - Chapter 11 - Solution Exercise 6 (BONUS)
Production-grade behavioral dashboard with HTML report
"""

from __future__ import annotations
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
import random

from trade_journal import generate_synthetic_journal
from emotion_tracker import EmotionTracker
from habit_tracker import HabitTracker
from post_trade_review import PostTradeReview
from decision_quality import DecisionQuality
from behavioral_dashboard import BehavioralDashboard


def build_complete_data():
    """Build complete dataset for dashboard demo."""

    # 1. Trade journal (3 months of trades)
    journal = generate_synthetic_journal(n_trades=120, seed=42)

    # 2. Emotion tracker (3 months)
    rng = np.random.default_rng(42)
    tracker = EmotionTracker()
    base = datetime(2024, 1, 1, 9, 0)
    for day in range(90):
        is_weekend = (base + timedelta(days=day)).weekday() >= 5
        for hour_idx, hour in enumerate([9, 12, 17]):
            base_stress = 4.5 if is_weekend else 5.5
            tracker.log(
                stress=int(np.clip(rng.normal(base_stress + hour_idx * 0.3, 1.0), 1, 10)),
                confidence=int(np.clip(rng.normal(6, 1.0), 1, 10)),
                patience=int(np.clip(rng.normal(6 - hour_idx * 0.2, 1.2), 1, 10)),
                focus=int(np.clip(rng.normal(7 - hour_idx * 0.4, 1.0), 1, 10)),
                energy=int(np.clip(rng.normal(6.5 - hour_idx * 0.5, 1.0), 1, 10)),
                timestamp=base + timedelta(days=day, hours=hour - 9),
            )

    # 3. Habit tracker
    habits = HabitTracker()
    today = date.today()
    random.seed(42)
    for i in range(60):
        d = today - timedelta(days=i)
        if random.random() < 0.93:
            habits.log_habit("journal", d)
        if random.random() < 0.89:
            habits.log_habit("checklist", d)
        if d.weekday() == 5 and random.random() < 0.85:
            habits.log_habit("weekly_review", d)
        if d.weekday() == 1 and random.random() < 0.80:
            habits.log_habit("mentor_call", d)
        if random.random() < 0.62:
            habits.log_habit("exercise", d)

    # 4. Post-trade reviews
    reviews = PostTradeReview()
    common_violations = [
        "Moved stop to breakeven prematurely",
        "Increased position size after win",
        "Skipped pre-trade checklist",
        "Held loser past stop loss",
    ]
    random.seed(123)
    for i in range(50):
        followed = random.random() > 0.15
        pnl = random.choice([200, 200, -200]) if followed else random.choice([200, -300])
        reviews.review_trade(
            trade_id=f"REV-{i+1:03d}",
            pnl=pnl,
            r_multiple=pnl / 200,
            followed_rules=followed,
            rule_violations=[] if followed else [random.choice(common_violations)],
            what_went_right="System trade" if followed else "",
            why_violated="Frustration" if not followed else "",
        )

    return journal, tracker, habits, reviews


def run_exercise_6():
    """Run Exercise 6 - BONUS - Production Dashboard."""
    print("=" * 70)
    print("EXERCISE 6 (BONUS): PRODUCTION BEHAVIORAL DASHBOARD")
    print("=" * 70)
    print("""
Goal: build production-grade dashboard combining all psychology systems.
Time: 180 minutes

Features:
  - Real-time emotion + performance metrics
  - Habit adherence tracking
  - Bias detection alerts
  - Mentor sharing format (HTML report)
  - Weekly automated review generation
""")

    # Build complete data
    print("\n--- Building Complete Dataset ---")
    journal, tracker, habits, reviews = build_complete_data()
    print(f"  Trade journal: {len(journal.entries)} trades")
    print(f"  Emotion tracker: {len(tracker.readings)} readings")
    print(f"  Habit tracker: {len(habits.habits)} habits")
    print(f"  Post-trade reviews: {len(reviews.reviews)} reviews")

    # Build dashboard
    dashboard = BehavioralDashboard(
        trade_journal=journal,
        emotion_tracker=tracker,
        habit_tracker=habits,
        post_trade_review=reviews,
        backtest_avg_win=1.5,
        backtest_avg_loss=-1.0,
        backtest_win_rate=0.50,
    )

    # Show health score
    health = dashboard.overall_health_score()
    print(f"\n--- Overall Health Score ---")
    print(f"Score: {health:.1f}/100")
    if health >= 80:
        print("Status: ✓ EXCELLENT - sustainable trader profile")
    elif health >= 60:
        print("Status: ✓ GOOD - solid foundation")
    elif health >= 40:
        print("Status: ⚠ DEVELOPING - work needed")
    else:
        print("Status: ✗ CONCERNING - significant gaps")

    # Status summary
    print(f"\n--- Status Summary ---")
    status = dashboard.status_summary()
    for area, msg in status.items():
        print(f"  {area:25s}: {msg}")

    # Generate HTML report
    print(f"\n--- Generating HTML Report ---")
    html_path = dashboard.to_html("/tmp/exercise_6_dashboard.html")
    print(f"  Saved: {html_path}")

    # Generate text report
    print(f"\n--- Saving Text Report ---")
    text_path = "/tmp/exercise_6_dashboard.txt"
    with open(text_path, "w") as f:
        f.write(dashboard.report())
    print(f"  Saved: {text_path}")

    # Mentor sharing format
    print(f"\n" + "=" * 70)
    print("MENTOR SHARING FORMAT (excerpts)")
    print("=" * 70)

    print(f"\nKey Metrics for Mentor Discussion:")
    print(f"  Overall health: {health:.0f}/100")

    # Trade metrics summary
    metrics = journal.metrics()
    if metrics.get("n_trades", 0) > 0:
        print(f"  Trades analyzed: {metrics['n_trades']}")
        print(f"  Win rate: {metrics['win_rate']*100:.1f}%")
        print(f"  Net P&L: ${metrics['total_pnl']:,.2f}")
        print(f"  Rule adherence: {metrics['rule_adherence_pct']:.1f}%")
        print(f"  Mistakes: {metrics['n_mistakes']}")
        print(f"  Noise (acceptable losses): {metrics['n_noise']}")

    # Habit summary
    print(f"\nHabit Identity Strength:")
    for name in habits.habits:
        strength = habits.identity_strength(name)
        print(f"  {name:20s}: {strength:5.1f}/100")

    # Production tips
    print(f"\n" + "=" * 70)
    print("PRODUCTION DEPLOYMENT TIPS")
    print("=" * 70)
    print("""
1. SCHEDULE:
   - Daily: log 3 emotion readings + 1 habit completion
   - Per trade: log via journal entry
   - End of day: post-trade review for each closed trade
   - Weekly: full dashboard review
   - Monthly: HTML report shared với mentor

2. AUTOMATION:
   - Cron job to generate dashboard HTML weekly
   - Email PDF version to mentor
   - Slack notification if health score < 50
   - Alert if severe bias detected

3. SCALING:
   - Use SQLite/PostgreSQL for production storage
   - Build web UI on top of components
   - Mobile app for emotion logging on-the-go
   - Integration với broker API for trade auto-import

4. PRIVACY:
   - Sensitive data, store locally
   - Encrypt at rest if cloud
   - Export-only sharing format for mentor
   - Backup important trades + reviews
""")


if __name__ == "__main__":
    run_exercise_6()
