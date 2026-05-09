"""
QuantCFD — Chương 10
Solution Exercise 3 — Loss Limit Manager Scenarios

Test LossLimitManager với 3 scenarios:
1. Single big loss (daily limit trigger)
2. Gradual losses (weekly/monthly cascade)
3. Recovery sau halt
"""

from __future__ import annotations
import pandas as pd
from datetime import timedelta
from loss_limit_manager import LossLimitManager


def run_scenario(
    name: str, equity_path: list, dates: list,
    initial_equity: float = 10000,
):
    """Run loss limit manager through equity path."""
    print(f"\n{'═' * 70}")
    print(f"SCENARIO: {name}")
    print(f"{'═' * 70}")

    mgr = LossLimitManager(initial_equity=initial_equity)
    halts_log = []

    for date, equity in zip(dates, equity_path):
        result = mgr.check_limits(equity, date)
        if not result["allow_trade"]:
            for h in result["active_halts"]:
                if h["reason"] not in [x["reason"] for x in halts_log]:
                    halts_log.append({
                        "date": date,
                        "equity": equity,
                        "reason": h["reason"],
                        "level": h["level"],
                        "until": h["until"],
                    })

    # Summary
    print(f"\nFinal equity: ${equity_path[-1]:,.2f}")
    print(f"Total halts triggered: {len(halts_log)}")
    if halts_log:
        print(f"\nHalt history:")
        for h in halts_log:
            print(f"  {h['date'].date()}: ${h['equity']:,.0f} → {h['level']}")
            print(f"    Reason: {h['reason']}")
            print(f"    Until:  {h['until'].date()}")

    print(f"\n{mgr.status_report()}")
    return mgr, halts_log


def scenario_single_big_loss():
    """Single -5% day triggers daily limit."""
    dates = [pd.Timestamp("2024-01-01") + timedelta(days=i) for i in range(10)]
    equity_path = [
        10000,  # day 0
        10100,  # day 1: +1%
        10250,  # day 2: +1.5%
        10300,  # day 3: +0.5%
        9700,   # day 4: -5.8% (triggers daily)
        9700,   # day 5: halted
        9700,   # day 6: halted
        9750,   # day 7: small recovery
        9800,   # day 8
        9850,   # day 9
    ]
    return run_scenario(
        "Single big loss (-5.8% day triggers daily limit)",
        equity_path, dates,
    )


def scenario_gradual_losses():
    """Gradual losses cascade through weekly → monthly limits."""
    dates = [pd.Timestamp("2024-02-01") + timedelta(days=i) for i in range(35)]
    # Slow grind down: -0.5% per day for 35 days = ~-16% total
    equity_path = []
    eq = 10000
    for i in range(35):
        eq *= 0.995  # -0.5% per day
        equity_path.append(eq)
    return run_scenario(
        "Gradual losses (-0.5% daily) — cascade through weekly + monthly",
        equity_path, dates,
    )


def scenario_recovery():
    """DD followed by recovery — halts release after time."""
    dates = [pd.Timestamp("2024-03-01") + timedelta(days=i) for i in range(60)]
    equity_path = []
    eq = 10000

    # Phase 1: gradual loss to -18% over 20 days
    for i in range(20):
        eq *= 0.99
        equity_path.append(eq)

    # Phase 2: stabilize 10 days
    for i in range(10):
        equity_path.append(eq)

    # Phase 3: gradual recovery 30 days
    for i in range(30):
        eq *= 1.008  # +0.8% per day
        equity_path.append(eq)

    return run_scenario(
        "Loss + recovery (halt release sau time)",
        equity_path, dates,
    )


def scenario_terminal():
    """Catastrophic DD triggers terminal limit."""
    dates = [pd.Timestamp("2024-04-01") + timedelta(days=i) for i in range(50)]
    equity_path = []
    eq = 10000
    # Steep drop over 30 days
    for i in range(30):
        eq *= 0.983  # -1.7% per day = -41% over 30 days
        equity_path.append(eq)
    # Stabilize
    for i in range(20):
        equity_path.append(eq)
    return run_scenario(
        "Terminal limit triggered (-41% DD, stop trading)",
        equity_path, dates,
    )


if __name__ == "__main__":
    print("=" * 80)
    print("Bài 3 — Loss Limit Manager: 4 Scenarios")
    print("=" * 80)

    # Run all scenarios
    scenario_single_big_loss()
    scenario_gradual_losses()
    scenario_recovery()
    scenario_terminal()

    print(f"\n{'═' * 80}")
    print("CONCLUSIONS")
    print(f"{'═' * 80}")
    print("""
Limit framework summary:
  Daily   (-3%):  catches single-day catastrophes (news events, fat finger)
  Weekly  (-7%):  catches cluster of bad trades over 5-7 days
  Monthly (-15%): catches sustained DD periods, forces 2-week cooling off
  Total DD (-20%): catches structural problems, requires full review
  Terminal (-40%): catastrophic failure, stop trading 6+ months

Key insights:
  - Pre-committed limits prevent emotional override
  - Halts have automatic expiry — graduated response, not permanent stop
  - Multiple levels protect against different failure modes
  - Terminal limit is final safety net before total ruin

Behavioral benefit:
  - Removes "should I stop today?" decision in moment of stress
  - Enforces capital preservation discipline
  - Forces review/recovery time after losses
""")
