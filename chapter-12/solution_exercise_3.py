"""
Bài tập 3 (Advanced) - Live monitoring dashboard
=================================================

Goal: Build live monitoring system với real-time equity curve,
open positions, risk metrics, Discord alerts.

Simulates 1 week of trading with various scenarios.

QuantCFD Chapter 12 Capstone exercise.
"""
from datetime import datetime, timedelta
import numpy as np
from live_monitoring import LiveMonitor, AlertPriority, AlertCategory


def main():
    """Simulate 1 week of trading with realistic scenarios."""
    print("=" * 70)
    print("EXERCISE 3: LIVE MONITORING DASHBOARD")
    print("=" * 70)

    np.random.seed(42)

    # Initialize monitor
    monitor = LiveMonitor()

    # Track equity curve
    equity = 30000.0
    max_equity_30d = 30000.0
    week_start_equity = 30000.0

    # Simulate 5 trading days với different scenarios
    print("\n--- DAY 1: Normal trading day ---")
    print("Pre-market: routine check")
    snap = monitor.update_dashboard(
        account_equity=equity,
        cash_balance=18000,
        open_positions=0,
        today_pnl=0,
        week_pnl=0,
        month_pnl=2400,
        max_equity_30d=max_equity_30d,
        var_95=400,
        rule_adherence_today=1.0,
        open_risk=0,
    )

    # Mid-day: trades opening
    print("\nMid-day: 3 positions opened")
    monitor.create_alert(
        priority=AlertPriority.MEDIUM,
        category=AlertCategory.TRADE_EVENT,
        title="Trend long XAUUSD opened",
        message="Entry $2050, stop $2030, target $2090, risk $80",
        metadata={'symbol': 'XAUUSD', 'direction': 'long', 'risk': 80},
    )
    monitor.create_alert(
        priority=AlertPriority.MEDIUM,
        category=AlertCategory.TRADE_EVENT,
        title="MR short EURUSD opened",
        message="Entry 1.0850, stop 1.0880, target 1.0790, risk $90",
    )

    snap = monitor.update_dashboard(
        account_equity=equity + 50,
        cash_balance=12000,
        open_positions=3,
        today_pnl=50,
        week_pnl=50,
        month_pnl=2450,
        max_equity_30d=max_equity_30d,
        var_95=600,
        rule_adherence_today=1.0,
        open_risk=240,
    )

    # End of day
    equity += 320
    max_equity_30d = max(max_equity_30d, equity)
    print("\nEnd of Day 1:")
    snap = monitor.update_dashboard(
        account_equity=equity,
        cash_balance=20000,
        open_positions=1,
        today_pnl=320,
        week_pnl=320,
        month_pnl=2720,
        max_equity_30d=max_equity_30d,
        var_95=200,
        rule_adherence_today=1.0,
        open_risk=80,
    )
    print(monitor.generate_dashboard_text())

    # Day 2-3: Drawdown beginning
    print("\n--- DAY 2: Bad day, losses accumulating ---")
    equity -= 480
    today_pnl = -480
    snap = monitor.update_dashboard(
        account_equity=equity,
        cash_balance=20000,
        open_positions=2,
        today_pnl=today_pnl,
        week_pnl=320 + today_pnl,
        month_pnl=2240,
        max_equity_30d=max_equity_30d,
        var_95=400,
        rule_adherence_today=0.95,
        open_risk=160,
    )
    print(monitor.generate_dashboard_text())

    # Day 3: Continued losses
    print("\n--- DAY 3: Losses continuing, daily limit approaching ---")
    today_pnl = -550
    equity += today_pnl
    snap = monitor.update_dashboard(
        account_equity=equity,
        cash_balance=20000,
        open_positions=2,
        today_pnl=today_pnl,
        week_pnl=320 - 480 + today_pnl,
        month_pnl=2240 + today_pnl,
        max_equity_30d=max_equity_30d,
        var_95=400,
        rule_adherence_today=0.92,
        open_risk=160,
    )
    print(monitor.generate_dashboard_text())

    # Day 4: Daily limit breached
    print("\n--- DAY 4: Daily loss limit breached, kill switch ---")
    today_pnl = -1100
    equity += today_pnl
    snap = monitor.update_dashboard(
        account_equity=equity,
        cash_balance=20000,
        open_positions=0,  # Kill switch closed
        today_pnl=today_pnl,
        week_pnl=320 - 480 - 550 + today_pnl,
        month_pnl=2240 - 550 + today_pnl,
        max_equity_30d=max_equity_30d,
        var_95=0,
        rule_adherence_today=1.0,
        open_risk=0,
    )

    # Manual incident report
    monitor.create_alert(
        priority=AlertPriority.CRITICAL,
        category=AlertCategory.RISK_BREACH,
        title="DAY HALTED",
        message="Trading halted by kill switch. Resume tomorrow.",
    )

    print(monitor.generate_dashboard_text())

    # Day 5: Recovery
    print("\n--- DAY 5: Recovery day, lower position sizing ---")
    today_pnl = 280
    equity += today_pnl
    snap = monitor.update_dashboard(
        account_equity=equity,
        cash_balance=20500,
        open_positions=1,
        today_pnl=today_pnl,
        week_pnl=320 - 480 - 550 - 1100 + today_pnl,
        month_pnl=2240 - 550 - 1100 + today_pnl,
        max_equity_30d=max_equity_30d,
        var_95=200,
        rule_adherence_today=1.0,
        open_risk=80,
    )
    print(monitor.generate_dashboard_text())

    # Weekly summary
    print("\n\n=== WEEKLY SUMMARY ===")
    weekly_pnl = equity - week_start_equity
    weekly_pct = weekly_pnl / week_start_equity * 100
    print(f"Starting equity:    ${week_start_equity:,.2f}")
    print(f"Ending equity:      ${equity:,.2f}")
    print(f"Weekly P&L:         ${weekly_pnl:+,.2f} ({weekly_pct:+.2f}%)")
    print(f"Best day:           +$320 (Day 1)")
    print(f"Worst day:          -$1,100 (Day 4)")
    print(f"Days halted:        1 (Day 4)")

    # Alert summary
    print("\n=== ALERT SUMMARY ===")
    all_alerts = monitor.alerts
    print(f"Total alerts:       {len(all_alerts)}")
    print(f"Critical:           {sum(1 for a in all_alerts if a.priority == AlertPriority.CRITICAL)}")
    print(f"High priority:      {sum(1 for a in all_alerts if a.priority == AlertPriority.HIGH)}")
    print(f"Medium priority:    {sum(1 for a in all_alerts if a.priority == AlertPriority.MEDIUM)}")
    print(f"Unacknowledged:     {len(monitor.get_unacknowledged_alerts())}")

    # Categorize alerts
    by_category = {}
    for a in all_alerts:
        cat = a.category.value
        by_category[cat] = by_category.get(cat, 0) + 1

    print("\nBy category:")
    for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
        print(f"   {cat:25s} {count}")

    # Lessons
    print("\n=== LESSONS LEARNED ===")
    print("1. Kill switch worked correctly Day 4 - prevented bigger loss")
    print("2. Recovery Day 5 was disciplined (lower sizing)")
    print("3. Multiple alerts triggered showed system functioning")
    print("4. Need to investigate 3 consecutive losing days pattern")
    print("5. Weekly P&L still recoverable next week")

    print("\n" + "=" * 70)
    print("Live monitoring exercise complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
