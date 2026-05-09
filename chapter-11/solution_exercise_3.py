"""
QuantCFD - Chapter 11 - Solution Exercise 3
EmotionTracker 2 weeks data + correlation with trade quality
"""

from __future__ import annotations
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from emotion_tracker import EmotionTracker, EmotionReading
from trade_journal import generate_synthetic_journal


def simulate_two_weeks_emotions(seed: int = 42) -> EmotionTracker:
    """
    Simulate 14 days of emotion tracking, 3 readings per day.

    In real usage: log readings yourself 3x daily for 2 weeks.
    """
    rng = np.random.default_rng(seed)
    tracker = EmotionTracker()

    base = datetime(2024, 6, 1, 9, 0)

    for day in range(14):
        # Day-specific energy baseline
        day_energy_base = float(rng.normal(6.0, 1.0))
        day_stress_base = float(rng.normal(5.0, 1.0))

        # Weekend recovery effect
        is_weekend = (base + timedelta(days=day)).weekday() >= 5
        if is_weekend:
            day_stress_base -= 1.5
            day_energy_base += 1.0

        for hour_idx, hour in enumerate([9, 12, 17]):
            # Energy declines through day
            energy_modifier = -hour_idx * 0.5
            stress_modifier = hour_idx * 0.3  # stress builds up

            tracker.log(
                stress=int(np.clip(rng.normal(day_stress_base + stress_modifier, 1.0), 1, 10)),
                confidence=int(np.clip(rng.normal(6, 1.0), 1, 10)),
                patience=int(np.clip(rng.normal(6 - hour_idx * 0.3, 1.2), 1, 10)),
                focus=int(np.clip(rng.normal(7 - hour_idx * 0.5, 1.0), 1, 10)),
                energy=int(np.clip(rng.normal(day_energy_base + energy_modifier, 1.0), 1, 10)),
                timestamp=base + timedelta(days=day, hours=hour - 9),
            )

    return tracker


def correlation_with_trades(tracker: EmotionTracker, trades_df: pd.DataFrame):
    """Show correlation between emotions and trade quality."""
    print("\n" + "=" * 70)
    print("EMOTION ↔ TRADE QUALITY CORRELATION")
    print("=" * 70)

    correlations = tracker.correlate_with_trades(trades_df)
    if not correlations:
        print("Insufficient data for correlation (need 5+ overlapping days).")
        return

    print("\nCorrelation between daily average emotion and daily P&L:")
    print(f"  Stress ↔ P&L:     {correlations['stress_pnl']:+.3f}")
    print(f"  Confidence ↔ P&L: {correlations['confidence_pnl']:+.3f}")
    print(f"  Patience ↔ P&L:   {correlations['patience_pnl']:+.3f}")
    print(f"  Energy ↔ P&L:     {correlations['energy_pnl']:+.3f}")
    print(f"  Composite ↔ P&L:  {correlations['composite_pnl']:+.3f}")

    print("\nInterpretation:")
    if correlations.get("stress_pnl", 0) < -0.2:
        print("  ⚠ Strong negative stress-PnL correlation")
        print("    → High stress days = worse trades")
        print("    → Implement stress threshold for halt")

    if correlations.get("energy_pnl", 0) > 0.2:
        print("  ✓ Positive energy-PnL correlation")
        print("    → High energy days = better trades")
        print("    → Skip trading on low-energy days")

    if correlations.get("patience_pnl", 0) > 0.2:
        print("  ✓ Positive patience-PnL correlation")
        print("    → Patient state = better outcomes")
        print("    → Mandatory cooling-off after losses")

    if correlations.get("composite_pnl", 0) > 0.3:
        print("  ✓ Strong composite-PnL relationship")
        print("    → Overall emotional state matters significantly")
        print("    → Pre-trade emotion check is HIGH leverage")


def daily_pattern_analysis(tracker: EmotionTracker):
    """Analyze patterns within day (morning/midday/evening)."""
    df = tracker.to_dataframe()
    if df.empty:
        return

    print("\n" + "=" * 70)
    print("INTRADAY EMOTION PATTERNS")
    print("=" * 70)

    df["hour"] = df["timestamp"].dt.hour
    by_hour = df.groupby("hour").agg(
        avg_stress=("stress", "mean"),
        avg_confidence=("confidence", "mean"),
        avg_patience=("patience", "mean"),
        avg_focus=("focus", "mean"),
        avg_energy=("energy", "mean"),
        avg_composite=("composite", "mean"),
    )
    print("\nAverage emotional state by hour:")
    print(by_hour.round(2).to_string())

    # Identify best vs worst hours
    if len(by_hour) > 1:
        best_hour = by_hour["avg_composite"].idxmax()
        worst_hour = by_hour["avg_composite"].idxmin()
        print(f"\nBest hour for trading: {best_hour}:00 (composite {by_hour.loc[best_hour, 'avg_composite']:.2f})")
        print(f"Worst hour for trading: {worst_hour}:00 (composite {by_hour.loc[worst_hour, 'avg_composite']:.2f})")


def day_of_week_analysis(tracker: EmotionTracker):
    """Day of week patterns."""
    df = tracker.to_dataframe()
    if df.empty:
        return

    print("\n" + "=" * 70)
    print("DAY-OF-WEEK EMOTION PATTERNS")
    print("=" * 70)

    df["dow"] = df["timestamp"].dt.day_name()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    by_dow = df.groupby("dow")[["stress", "energy", "composite"]].mean()
    by_dow = by_dow.reindex([d for d in dow_order if d in by_dow.index])
    print("\nEmotional state by day of week:")
    print(by_dow.round(2).to_string())


def run_exercise_3():
    """Run Exercise 3."""
    print("=" * 70)
    print("EXERCISE 3: EMOTION TRACKING + TRADE CORRELATION")
    print("=" * 70)
    print("""
Goal: track emotions for 2 weeks, correlate with trade outcomes.
Time: 90 minutes (analysis), 2 weeks (data collection)

In production:
  - Log 3 emotion readings per day (morning, midday, evening)
  - 14 days of consistent tracking minimum
  - Correlate with trade journal P&L

Here: simulate realistic 2-week dataset for demonstration.
""")

    # Build tracker
    print("\n--- Simulating 2 Weeks of Emotion Data ---")
    tracker = simulate_two_weeks_emotions(seed=42)
    print(f"Logged {len(tracker.readings)} readings (14 days × 3 readings)")

    # Show report
    print("\n")
    print(tracker.report())

    # Show trends
    trends = tracker.trends(days=14)
    print("\n--- 14-Day Trends ---")
    print(trends.round(2).to_string())

    # Generate synthetic trades for correlation
    print("\n--- Generating Trades for Correlation Analysis ---")
    journal = generate_synthetic_journal(n_trades=30, seed=42)
    trades_df = journal.to_dataframe()
    correlation_with_trades(tracker, trades_df)

    # Pattern analysis
    daily_pattern_analysis(tracker)
    day_of_week_analysis(tracker)

    # Reflection
    print("\n" + "=" * 70)
    print("REFLECTION PROMPTS")
    print("=" * 70)
    print("""
1. WHEN ARE YOU AT YOUR BEST?
   - Best hour of day?
   - Best day of week?
   - Trade only during peak windows

2. WHEN ARE YOU AT YOUR WORST?
   - Worst hour (post-lunch dip?)
   - Worst day (Monday recovery?)
   - Avoid trading these times

3. WHICH EMOTIONS PREDICT TRADE QUALITY?
   - Stress, energy, patience matter most
   - Build pre-trade thresholds based on these

4. ARE YOU DETECTING PATTERNS?
   - Does stress build up week by week?
   - Do you recover on weekends?
   - Burnout warning signs?

5. ACTION ITEMS:
   - Halt trading when stress > X
   - Skip trading when energy < Y
   - Trade only during peak hours
   - Build mental rehearsal for weakness windows
""")


if __name__ == "__main__":
    run_exercise_3()
