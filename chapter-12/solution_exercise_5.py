"""
Bài tập 5 (Advanced) - Phased deployment manager
==================================================

Goal: Implement và simulate phase progression from Phase 0 paper
to Phase 4 institutional, validating exit criteria each step.

Demonstrates 30-month deployment journey of trader Trí.

QuantCFD Chapter 12 Capstone exercise.
"""
from datetime import datetime, timedelta
from deployment_phases import (
    PhasedDeploymentManager,
    DeploymentPhase,
    PhaseRequirements,
)


def simulate_phase_journey():
    """Simulate 30-month deployment journey through phases."""
    print("=" * 70)
    print("EXERCISE 5: PHASED DEPLOYMENT SIMULATION")
    print("=" * 70)
    print("\nSimulating Trí's deployment journey from Phase 0 to Phase 4")
    print("over 30 months (2.5 years)")
    print("=" * 70)

    manager = PhasedDeploymentManager()

    # ============ PHASE 0: PAPER (Months 0-2) ============
    print("\n\n┌─ PHASE 0: PAPER TRADING (Months 0-2) ─┐")
    start_phase0 = datetime.now() - timedelta(days=900)  # 30 months ago

    manager.initialize_at_phase(
        phase=DeploymentPhase.PHASE_0_PAPER,
        starting_capital=0,
        start_date=start_phase0,
    )

    # Simulate 60 days of paper trading
    manager.update_stats(
        current_date=start_phase0 + timedelta(days=60),
        current_capital=24500,  # Paper started $20k, +22.5%
        closed_trades=58,
        realized_pnl=4500,
        rolling_sharpe=0.95,
        max_dd_experienced=-0.07,
        rule_adherence_pct=0.93,
        mentor_approval=True,
        family_supportive=True,
    )

    print(manager.generate_status_report())

    # Advance to Phase 1
    print("\n→ Attempting advance to Phase 1 ($1k):")
    result = manager.advance_to_next_phase(new_capital=1000)
    print(f"   Result: {'SUCCESS' if result['success'] else 'FAILED'}")
    if result['success']:
        print(f"   Promoted from {result['old_phase'].name} to {result['new_phase'].name}")

    # ============ PHASE 1: TINY LIVE (Months 2-4) ============
    print("\n\n┌─ PHASE 1: TINY LIVE $1k (Months 2-4) ─┐")
    # Update to 60 days in Phase 1
    manager.update_stats(
        current_date=manager.current_stats.start_date + timedelta(days=60),
        current_capital=1180,  # +18% on $1k
        closed_trades=35,
        realized_pnl=180,
        rolling_sharpe=0.65,
        max_dd_experienced=-0.05,
        rule_adherence_pct=0.94,
        mentor_approval=True,
        family_supportive=True,
    )
    print(manager.generate_status_report())

    print("\n→ Attempting advance to Phase 2 ($5k):")
    result = manager.advance_to_next_phase(new_capital=5000)
    print(f"   Result: {'SUCCESS' if result['success'] else 'FAILED'}")

    # ============ PHASE 2: SMALL LIVE (Months 4-9) ============
    print("\n\n┌─ PHASE 2: SMALL LIVE $5k (Months 4-9) ─┐")
    # 5 months in Phase 2
    manager.update_stats(
        current_date=manager.current_stats.start_date + timedelta(days=150),
        current_capital=5650,  # +13% on $5k
        closed_trades=120,
        realized_pnl=650,
        rolling_sharpe=0.85,
        max_dd_experienced=-0.11,
        rule_adherence_pct=0.92,
        mentor_approval=True,
        family_supportive=True,
    )
    print(manager.generate_status_report())

    print("\n→ Attempting advance to Phase 3 ($20k):")
    result = manager.advance_to_next_phase(new_capital=20000)
    print(f"   Result: {'SUCCESS' if result['success'] else 'FAILED'}")

    # ============ PHASE 3: MEDIUM LIVE (Months 9-18) ============
    print("\n\n┌─ PHASE 3: MEDIUM LIVE $20k (Months 9-18) ─┐")
    # 9 months in Phase 3
    manager.update_stats(
        current_date=manager.current_stats.start_date + timedelta(days=270),
        current_capital=25600,  # +28% on $20k
        closed_trades=235,
        realized_pnl=5600,
        rolling_sharpe=1.18,
        max_dd_experienced=-0.10,
        rule_adherence_pct=0.95,
        mentor_approval=True,
        family_supportive=True,
    )
    print(manager.generate_status_report())

    print("\n→ Attempting advance to Phase 4 ($100k):")
    result = manager.advance_to_next_phase(new_capital=100000)
    print(f"   Result: {'SUCCESS' if result['success'] else 'FAILED'}")

    # ============ PHASE 4: SIGNIFICANT (Months 18-30) ============
    print("\n\n┌─ PHASE 4: SIGNIFICANT $100k (Months 18-30) ─┐")
    # 12 months in Phase 4
    manager.update_stats(
        current_date=manager.current_stats.start_date + timedelta(days=365),
        current_capital=122000,  # +22% on $100k
        closed_trades=520,
        realized_pnl=22000,
        rolling_sharpe=1.18,
        max_dd_experienced=-0.08,
        rule_adherence_pct=0.96,
        mentor_approval=True,
        family_supportive=True,
    )
    print(manager.generate_status_report())

    # ============ JOURNEY SUMMARY ============
    print("\n\n" + "=" * 70)
    print("DEPLOYMENT JOURNEY SUMMARY")
    print("=" * 70)

    print(f"\nTotal phases completed: {len(manager.phase_history)}")
    print(f"Currently in:           {manager.current_phase.name}")
    print(f"Starting capital:       $0 (paper)")
    print(f"Current capital:        ${manager.current_stats.current_capital:,.2f}")

    print("\nPhase-by-phase performance:")
    for h in manager.phase_history:
        print(f"  {h['phase'].name:30s} {h['duration_days']:>3}d  "
              f"${h['final_capital']:>10,.0f}  "
              f"Sharpe={h['sharpe']:.2f}  "
              f"Trades={h['trades']}")

    # Current phase stats
    print(f"\nCurrent phase ({manager.current_phase.name}):")
    s = manager.current_stats
    print(f"  Days in phase:    {s.days_in_phase}")
    print(f"  Capital:          ${s.current_capital:,.2f}")
    print(f"  Total return:     {s.total_return_pct:+.2f}%")
    print(f"  Trades:           {s.closed_trades}")
    print(f"  Sharpe:           {s.rolling_sharpe:.2f}")

    # Lessons
    print("\n\nLESSONS FROM SIMULATION:")
    print("  1. Phase progression takes time — Phase 0 to 4 = 30 months")
    print("  2. Each phase has clear exit criteria")
    print("  3. Skipping criteria not allowed — system enforces")
    print("  4. Sharpe and rule adherence improve over phases")
    print("  5. Capital scaled gradually không 0-to-100")
    print("  6. Day job continued throughout (per Ch12 recommendation)")

    print("\n" + "=" * 70)
    print("Phased deployment exercise complete.")
    print("=" * 70)


if __name__ == "__main__":
    simulate_phase_journey()
