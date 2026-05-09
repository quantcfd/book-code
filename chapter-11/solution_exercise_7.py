"""
QuantCFD - Chapter 11 - Solution Exercise 7 (BONUS)
Full IntegratedPsychologySystem demonstration

Simulates 30-day trading workflow using complete integrated system:
- Pre-trade: emotion → bias → checklist
- Trade execution
- Post-trade: log → review → pattern analysis
- Weekly review automation
- Mentor reports
"""

from __future__ import annotations
from datetime import datetime, timedelta, date
import numpy as np

from trade_journal import TradeEntry
from integrated_psychology_system import IntegratedPsychologySystem


def simulate_trader_day(
    system: IntegratedPsychologySystem,
    day_number: int,
    rng: np.random.Generator,
    base_date: datetime,
):
    """Simulate one full trader day."""
    print(f"\n{'─' * 70}")
    print(f"DAY {day_number} - {(base_date + timedelta(days=day_number)).date()}")
    print(f"{'─' * 70}")

    # Reset daily counters
    system.reset_daily_state()

    # Trader emotional state for the day (varies)
    base_stress = float(np.clip(rng.normal(5, 1.5), 1, 10))
    base_energy = float(np.clip(rng.normal(6, 1.0), 2, 10))
    base_patience = float(np.clip(rng.normal(6, 1.5), 2, 10))

    # 2-4 trade signals per day
    n_signals = int(rng.integers(2, 5))

    for trade_num in range(n_signals):
        hour = 9 + trade_num * 2
        timestamp = base_date + timedelta(days=day_number, hours=hour)

        # Pre-trade emotion (slightly varies through day)
        stress = int(np.clip(base_stress + rng.normal(0, 0.5), 1, 10))
        energy = int(np.clip(base_energy - trade_num * 0.3 + rng.normal(0, 0.5), 1, 10))
        patience = int(np.clip(base_patience - trade_num * 0.4 + rng.normal(0, 0.5), 1, 10))
        confidence = int(np.clip(rng.normal(6, 1), 1, 10))
        focus = int(np.clip(rng.normal(7, 1), 1, 10))

        # Setup grade
        grade = rng.choice(["A", "B", "C"], p=[0.5, 0.3, 0.2])

        print(f"\n  Trade signal #{trade_num + 1} at {hour}:00")
        print(f"    Setup grade: {grade}")
        print(f"    Stress: {stress}, Energy: {energy}, Patience: {patience}")

        # Run pre-trade workflow
        result = system.pre_trade_workflow(
            signal_triggered=True,
            setup_grade=grade,
            position_size=0.05,
            risk_amount=200,
            current_equity=20000,
            stress=stress,
            confidence=confidence,
            patience=patience,
            focus=focus,
            energy=energy,
            feeling_fomo=False,
            feeling_revenge=system._consecutive_losses_today >= 3,
        )

        if not result.can_trade:
            print(f"    ✗ TRADE BLOCKED")
            for r in result.blocking_reasons[:3]:
                print(f"      - {r}")
            continue

        # Execute trade
        is_long = rng.random() > 0.5
        entry_price = 2000 + rng.uniform(-50, 50)
        if is_long:
            stop_loss = entry_price - 20
            profit_target = entry_price + 40
        else:
            stop_loss = entry_price + 20
            profit_target = entry_price - 40

        trade_id = f"D{day_number:02d}-T{trade_num + 1}"
        entry = TradeEntry(
            trade_id=trade_id,
            date_entry=timestamp,
            asset="XAUUSD",
            strategy="trend",
            entry_trigger="MA cross",
            confidence=confidence,
            setup_grade=grade,
            pre_trade_emotion=stress,
            entry_price=entry_price,
            stop_loss=stop_loss,
            profit_target=profit_target,
            position_size=0.05,
            risk_amount=200,
        )
        system.log_trade(entry)
        print(f"    ✓ Trade executed @ {entry_price:.2f}")

        # Outcome based on grade + emotion
        win_prob = {"A": 0.62, "B": 0.50, "C": 0.38}[grade]
        # Emotional penalty
        if stress > 7 or energy < 4:
            win_prob -= 0.10
        won = rng.random() < win_prob
        exit_price = profit_target if won else stop_loss

        # Followed rules (90% normally, lower if revenge)
        followed = rng.random() > (0.30 if system._consecutive_losses_today >= 3 else 0.10)

        # Post-trade workflow
        post = system.post_trade_workflow(
            trade_id=trade_id,
            exit_price=exit_price,
            followed_rules=followed,
            rule_violations=[] if followed else ["Override during stress"],
            what_went_right="System trade" if followed else "",
            lesson_learned="" if followed else "Need cooling-off rule enforcement",
        )

        result_str = "WIN" if won else "LOSS"
        rule_str = "✓ followed" if followed else "✗ violated"
        print(f"    Result: {result_str} ${post['trade'].pnl:+.0f} ({rule_str})")
        print(f"    Quadrant: {post['review_quality']}")

    # End of day summary
    summary = system.daily_summary()
    print(f"\n  End of day {day_number}:")
    print(f"    Trades: {summary.get('trades_today', 0)}")
    print(f"    Wins: {summary.get('wins_today', 0)}")
    print(f"    Net P&L: ${summary.get('net_pnl_today', 0):+.0f}")
    print(f"    Rule adherence: {summary.get('rule_adherence_today', 0)*100:.0f}%")


def run_exercise_7():
    """Run Exercise 7 - Full integrated system demo."""
    print("=" * 80)
    print("EXERCISE 7 (BONUS): FULL INTEGRATED PSYCHOLOGY SYSTEM")
    print("=" * 80)
    print("""
Goal: demonstrate IntegratedPsychologySystem in complete trader workflow.
Time: 240 minutes (build) + ongoing daily use

This exercise simulates 30 days of trading with full integration:
  - Pre-trade workflow (emotion → bias → checklist)
  - Trade logging
  - Post-trade workflow (close → review → pattern)
  - Weekly automated review
  - Mentor consultation report

In production: this is your daily trading workflow.
""")

    # Initialize system
    print("\n--- Initializing System ---")
    system = IntegratedPsychologySystem(
        backtest_avg_win=1.5,
        backtest_avg_loss=-1.0,
        backtest_win_rate=0.50,
        max_risk_per_trade_pct=0.01,
        daily_loss_limit_pct=0.03,
    )
    print("✓ All components initialized")
    print("  - TradeJournal")
    print("  - EmotionTracker")
    print("  - HabitTracker")
    print("  - PostTradeReview")
    print("  - DecisionQuality")
    print("  - MentalSimulation")
    print("  - PreTradeChecklist")

    # Simulate 14 trading days (compressed demo)
    print("\n--- Simulating 14 Trading Days ---")
    rng = np.random.default_rng(42)
    base_date = datetime(2024, 1, 1)
    n_days = 14

    for day in range(n_days):
        simulate_trader_day(system, day, rng, base_date)

    # Show aggregate results
    print("\n\n" + "=" * 80)
    print("AGGREGATE RESULTS AFTER 14 DAYS")
    print("=" * 80)

    metrics = system.trade_journal.metrics()
    print(f"\nTotal trades: {metrics.get('n_trades', 0)}")
    if metrics.get('n_trades', 0) > 0:
        print(f"Win rate: {metrics['win_rate']*100:.1f}%")
        print(f"Net P&L: ${metrics['total_pnl']:+.2f}")
        print(f"Rule adherence: {metrics['rule_adherence_pct']:.1f}%")
        print(f"Avg R-multiple: {metrics['avg_r_multiple']:+.2f}")

    # Generate weekly review
    print("\n\n")
    print(system.weekly_review())

    # Generate mentor report
    print("\n\n")
    print(system.mentor_report())

    # Get dashboard
    dashboard = system.get_dashboard()
    print(f"\n--- Final Behavioral Health Score ---")
    health = dashboard.overall_health_score()
    print(f"Score: {health:.1f}/100")

    print(f"\n--- Status Summary ---")
    status = dashboard.status_summary()
    for area, msg in status.items():
        print(f"  {area:25s}: {msg}")

    # Save HTML report
    html_path = dashboard.to_html("/tmp/exercise_7_dashboard.html")
    print(f"\n✓ Final HTML dashboard: {html_path}")

    print(f"\n" + "=" * 80)
    print("EXERCISE COMPLETE")
    print("=" * 80)
    print("""
You have now seen the full IntegratedPsychologySystem in action.

In production daily use:
  1. Morning: log emotion + check today's mental rehearsal scenario
  2. Pre-trade: run pre_trade_workflow before every entry
  3. Post-trade: run post_trade_workflow immediately after exit
  4. End of day: run daily_summary
  5. Weekly: run weekly_review (Saturday morning)
  6. Monthly: generate mentor_report for consultation
  7. Quarterly: full system review, identity progress check

This system replaces ad-hoc psychology work with structured workflow.
Outcome: discipline becomes automated, freeing willpower for actual trading.
""")


if __name__ == "__main__":
    run_exercise_7()
