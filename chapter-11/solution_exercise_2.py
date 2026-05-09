"""
QuantCFD - Chapter 11 - Solution Exercise 2
Trade journal builder with 50 historical trades + statistics analysis
"""

from __future__ import annotations
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from trade_journal import TradeJournal, TradeEntry, generate_synthetic_journal


def build_personal_journal(n_trades: int = 50, seed: int = 42) -> TradeJournal:
    """
    Build TradeJournal with 50 historical trades.

    In real usage: import từ broker statement.
    Here: generate realistic synthetic data.
    """
    rng = np.random.default_rng(seed)
    journal = TradeJournal()

    # Realistic strategy mix
    strategies = ["trend", "mr", "vol_bo"]
    strategy_weights = [0.4, 0.4, 0.2]

    # Realistic asset mix
    assets_by_strategy = {
        "trend": ["XAUUSD", "EURUSD", "USDJPY"],
        "mr": ["EURUSD", "GBPUSD", "AUDUSD"],
        "vol_bo": ["BTCUSD", "ETHUSD", "XAUUSD"],
    }

    grades = ["A", "B", "C"]
    grade_weights = [0.4, 0.4, 0.2]
    grade_win_rates = {"A": 0.62, "B": 0.50, "C": 0.38}

    base_date = datetime(2024, 1, 1)

    for i in range(n_trades):
        strategy = rng.choice(strategies, p=strategy_weights)
        asset = rng.choice(assets_by_strategy[strategy])
        grade = rng.choice(grades, p=grade_weights)
        confidence = int(rng.uniform(2, 5))

        # Asset-specific prices
        price_base = {
            "XAUUSD": 2000, "EURUSD": 1.08, "USDJPY": 150,
            "GBPUSD": 1.27, "AUDUSD": 0.66,
            "BTCUSD": 50000, "ETHUSD": 3000,
        }.get(asset, 2000)

        is_long = rng.random() > 0.5
        price_perturbation = rng.uniform(-0.02, 0.02)
        entry_price = price_base * (1 + price_perturbation)

        atr_pct = {"XAUUSD": 0.012, "EURUSD": 0.005, "BTCUSD": 0.025}.get(asset, 0.01)
        atr = entry_price * atr_pct

        if is_long:
            stop_loss = entry_price - 2 * atr
            profit_target = entry_price + 4 * atr
        else:
            stop_loss = entry_price + 2 * atr
            profit_target = entry_price - 4 * atr

        risk_amount = 200.0
        position_size = abs(risk_amount / (entry_price - stop_loss))

        # Pre-trade emotion (lower for A grade, higher for C)
        emotion_base = {"A": 3, "B": 4, "C": 6}[grade]
        pre_emotion = int(np.clip(rng.normal(emotion_base, 1.5), 1, 10))

        entry = TradeEntry(
            trade_id=f"TR-{i+1:04d}",
            date_entry=base_date + timedelta(days=i // 2, hours=int(rng.integers(9, 17))),
            asset=asset,
            strategy=strategy,
            entry_trigger=f"{strategy} signal trigger",
            confidence=confidence,
            setup_grade=grade,
            pre_trade_emotion=pre_emotion,
            entry_price=entry_price,
            stop_loss=stop_loss,
            profit_target=profit_target,
            position_size=position_size,
            risk_amount=risk_amount,
        )

        wins = rng.random() < grade_win_rates[grade]
        exit_price = profit_target if wins else stop_loss

        followed = rng.random() > 0.10
        entry.followed_rules = followed
        entry.is_mistake = not followed
        entry.is_noise = followed and not wins
        entry.checked_pnl_during = rng.random() > 0.5

        entry.close(exit_price=exit_price, exit_time=entry.date_entry + timedelta(hours=4))
        journal.add_trade(entry)

    return journal


def deep_statistical_analysis(journal: TradeJournal):
    """Deep statistical analysis of journal."""
    df = journal.to_dataframe()
    if df.empty:
        print("No data to analyze.")
        return

    print("=" * 70)
    print("DEEP STATISTICAL ANALYSIS")
    print("=" * 70)

    # Basic metrics
    metrics = journal.metrics()
    print(f"\nBasic Performance:")
    print(f"  Total trades: {metrics['n_trades']}")
    print(f"  Win rate: {metrics['win_rate']*100:.1f}%")
    print(f"  Net P&L: ${metrics['total_pnl']:,.2f}")
    print(f"  Avg R-multiple: {metrics['avg_r_multiple']:+.2f}")
    print(f"  Rule adherence: {metrics['rule_adherence_pct']:.1f}%")

    # By grade
    print(f"\nPerformance by Setup Grade:")
    grade_df = journal.metrics_by_grade()
    print(grade_df.to_string())

    # By strategy
    print(f"\nPerformance by Strategy:")
    strat_df = journal.metrics_by_strategy()
    print(strat_df.to_string())

    # By asset
    print(f"\nPerformance by Asset:")
    closed = df[df["pnl"].notna()]
    if not closed.empty:
        asset_df = closed.groupby("asset").agg(
            n_trades=("pnl", "count"),
            win_rate=("pnl", lambda x: (x > 0).mean()),
            total_pnl=("pnl", "sum"),
            avg_r=("r_multiple", "mean"),
        )
        print(asset_df.to_string())

    # Time-of-day analysis
    print(f"\nPerformance by Hour of Day:")
    if not closed.empty:
        closed_copy = closed.copy()
        closed_copy["hour"] = pd.to_datetime(closed_copy["date_entry"]).dt.hour
        hour_df = closed_copy.groupby("hour").agg(
            n_trades=("pnl", "count"),
            win_rate=("pnl", lambda x: (x > 0).mean()),
            avg_r=("r_multiple", "mean"),
        )
        print(hour_df.to_string())

    # Direction analysis
    print(f"\nLong vs Short Performance:")
    if not closed.empty:
        closed_copy = closed.copy()
        closed_copy["direction"] = closed_copy.apply(
            lambda r: "long" if r["profit_target"] > r["entry_price"] else "short",
            axis=1,
        )
        dir_df = closed_copy.groupby("direction").agg(
            n_trades=("pnl", "count"),
            win_rate=("pnl", lambda x: (x > 0).mean()),
            avg_r=("r_multiple", "mean"),
        )
        print(dir_df.to_string())

    # Emotion correlation
    print(f"\n--- Emotion Correlations ---")
    emo = journal.emotional_correlation()
    for k, v in emo.items():
        print(f"  {k}: {v:+.3f}")
    if emo.get("stress_pnl_correlation", 0) < -0.15:
        print("  ⚠ Higher stress correlates with worse outcomes")

    # Streak analysis
    print(f"\n--- Streak Analysis ---")
    if not closed.empty:
        results = (closed["pnl"] > 0).astype(int)
        max_win_streak = 0
        max_loss_streak = 0
        current = 0
        last = None
        for r in results:
            if last is None or r == last:
                current += 1
            else:
                current = 1
            last = r
            if r == 1:
                max_win_streak = max(max_win_streak, current)
            else:
                max_loss_streak = max(max_loss_streak, current)
        print(f"  Longest win streak: {max_win_streak}")
        print(f"  Longest loss streak: {max_loss_streak}")


def run_exercise_2():
    """Run Exercise 2 - Trade journal builder."""
    print("=" * 70)
    print("EXERCISE 2: TRADE JOURNAL BUILDER")
    print("=" * 70)
    print("""
Goal: build a TradeJournal with 50 historical trades and analyze patterns.
Time: 90 minutes

In production: import from broker statement (CSV export).
Here: generate realistic synthetic data for demonstration.
""")

    print("\n--- Building Journal with 50 Trades ---")
    journal = build_personal_journal(n_trades=50, seed=42)
    print(f"Built journal with {len(journal.entries)} trades")

    # Save to JSON
    journal.save_json("/tmp/exercise_2_journal.json")
    print("Saved to: /tmp/exercise_2_journal.json")

    # Run journal report
    print("\n")
    print(journal.report())

    # Deep analysis
    deep_statistical_analysis(journal)

    # Reflection
    print("\n" + "=" * 70)
    print("REFLECTION PROMPTS")
    print("=" * 70)
    print("""
Now that you have your journal data, reflect on:

1. WHICH SETUP GRADE IS YOUR EDGE?
   - A grade should have highest win rate
   - If C grade highest → re-evaluate grading criteria
   - If grades similar → grading not differentiating enough

2. WHICH STRATEGY IS WORKING?
   - Which has highest expectancy?
   - Should you scale up the best?
   - Should you retire the worst?

3. WHICH TIMES ARE BEST?
   - Time-of-day patterns
   - Day-of-week patterns
   - Avoid weakest periods

4. EMOTIONAL CORRELATION?
   - Does stress correlate with losses?
   - What does this tell you about pre-trade discipline?

5. RULE ADHERENCE?
   - What pct of trades did you follow rules?
   - Where do violations cluster?

Use these insights to build action items for next 30 days.
""")


if __name__ == "__main__":
    run_exercise_2()
