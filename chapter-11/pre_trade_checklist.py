"""
QuantCFD - Chapter 11
pre_trade_checklist.py - Hardcoded pre-trade checklist enforcer

30-second pre-trade checklist with 3 sections (setup/risk/mental).
Blocks trade execution if any check fails. Override prevention.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class ChecklistResult:
    """Result of pre-trade checklist evaluation."""
    timestamp: datetime
    passed: bool
    failed_checks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    section_results: Dict[str, bool] = field(default_factory=dict)

    @property
    def can_execute(self) -> bool:
        return self.passed

    def report(self) -> str:
        lines = []
        status = "✓ PASS - execute trade" if self.passed else "✗ FAIL - skip trade"
        lines.append(f"Pre-trade checklist: {status}")
        lines.append(f"Time: {self.timestamp}")
        for section, ok in self.section_results.items():
            mark = "✓" if ok else "✗"
            lines.append(f"  {mark} {section}")
        if self.failed_checks:
            lines.append("\nFailed checks:")
            for f in self.failed_checks:
                lines.append(f"  ✗ {f}")
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        return "\n".join(lines)


class PreTradeChecklist:
    """
    Hardcoded pre-trade checklist enforcer.

    Three sections:
    1. Setup verification (signal validity)
    2. Risk check (position sizing, limits)
    3. Mental check (emotional state, recent activity)

    Cannot be bypassed except via explicit override (logged as rule violation).
    """

    def __init__(
        self,
        max_risk_per_trade_pct: float = 0.01,
        daily_loss_limit_pct: float = 0.03,
        max_stress_for_trading: int = 6,
        min_energy_for_trading: int = 5,
        cooling_off_after_losses: int = 3,
    ):
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_stress_for_trading = max_stress_for_trading
        self.min_energy_for_trading = min_energy_for_trading
        self.cooling_off_after_losses = cooling_off_after_losses

    def check(
        self,
        # Setup parameters
        signal_triggered: bool,
        all_entry_criteria_met: bool,
        setup_grade: str,  # A / B / C
        # Risk parameters
        position_size: float,
        risk_amount: float,
        current_equity: float,
        daily_loss_so_far_pct: float,
        # Mental parameters
        stress_level: int,  # 1-10
        energy_level: int,  # 1-10
        consecutive_losses_today: int,
        feeling_fomo: bool,
        feeling_revenge: bool,
        recent_big_win_or_loss: bool,
    ) -> ChecklistResult:
        """
        Run full pre-trade checklist.

        Returns:
            ChecklistResult with pass/fail and details
        """
        result = ChecklistResult(timestamp=datetime.now(), passed=True)

        # =================================================================
        # SECTION 1: SETUP VERIFICATION
        # =================================================================
        section_1_ok = True

        if not signal_triggered:
            result.failed_checks.append("No valid strategy signal")
            section_1_ok = False

        if not all_entry_criteria_met:
            result.failed_checks.append("Not all entry criteria met")
            section_1_ok = False

        if setup_grade not in ("A", "B"):
            result.failed_checks.append(
                f"Setup grade {setup_grade} - only A or B allowed (skip C grade)"
            )
            section_1_ok = False

        result.section_results["Setup verification"] = section_1_ok

        # =================================================================
        # SECTION 2: RISK CHECK
        # =================================================================
        section_2_ok = True

        risk_pct = risk_amount / current_equity if current_equity > 0 else 0
        if risk_pct > self.max_risk_per_trade_pct:
            result.failed_checks.append(
                f"Risk {risk_pct:.2%} exceeds max {self.max_risk_per_trade_pct:.2%}"
            )
            section_2_ok = False

        if daily_loss_so_far_pct < -self.daily_loss_limit_pct:
            result.failed_checks.append(
                f"Daily loss limit reached: {daily_loss_so_far_pct:.2%}"
            )
            section_2_ok = False

        # Warning if approaching daily limit
        if daily_loss_so_far_pct < -self.daily_loss_limit_pct * 0.7:
            result.warnings.append(
                f"Approaching daily loss limit: {daily_loss_so_far_pct:.2%} of {-self.daily_loss_limit_pct:.2%}"
            )

        if position_size <= 0:
            result.failed_checks.append("Position size invalid (must be positive)")
            section_2_ok = False

        result.section_results["Risk check"] = section_2_ok

        # =================================================================
        # SECTION 3: MENTAL CHECK
        # =================================================================
        section_3_ok = True

        if stress_level > self.max_stress_for_trading:
            result.failed_checks.append(
                f"Stress level {stress_level}/10 exceeds threshold {self.max_stress_for_trading}"
            )
            section_3_ok = False

        if energy_level < self.min_energy_for_trading:
            result.failed_checks.append(
                f"Energy level {energy_level}/10 below threshold {self.min_energy_for_trading}"
            )
            section_3_ok = False

        if feeling_fomo:
            result.failed_checks.append("FOMO detected - signal to NOT trade")
            section_3_ok = False

        if feeling_revenge:
            result.failed_checks.append("Revenge feeling detected - mandatory halt")
            section_3_ok = False

        if consecutive_losses_today >= self.cooling_off_after_losses:
            result.failed_checks.append(
                f"Cooling-off triggered: {consecutive_losses_today} losses today"
            )
            section_3_ok = False

        if recent_big_win_or_loss:
            result.warnings.append(
                "Recent big win/loss - emotion may be elevated, proceed carefully"
            )

        result.section_results["Mental check"] = section_3_ok

        # Final verdict: ALL sections must pass
        result.passed = section_1_ok and section_2_ok and section_3_ok
        return result

    def quick_check(
        self,
        signal_triggered: bool = True,
        setup_grade: str = "A",
        risk_amount: float = 200,
        current_equity: float = 20000,
        daily_loss_so_far_pct: float = 0.0,
        stress_level: int = 4,
        energy_level: int = 7,
        consecutive_losses_today: int = 0,
        feeling_fomo: bool = False,
        feeling_revenge: bool = False,
    ) -> ChecklistResult:
        """Convenience method với defaults for simpler calls."""
        return self.check(
            signal_triggered=signal_triggered,
            all_entry_criteria_met=signal_triggered,
            setup_grade=setup_grade,
            position_size=0.05,
            risk_amount=risk_amount,
            current_equity=current_equity,
            daily_loss_so_far_pct=daily_loss_so_far_pct,
            stress_level=stress_level,
            energy_level=energy_level,
            consecutive_losses_today=consecutive_losses_today,
            feeling_fomo=feeling_fomo,
            feeling_revenge=feeling_revenge,
            recent_big_win_or_loss=False,
        )


if __name__ == "__main__":
    print("=" * 60)
    print("Pre-Trade Checklist Demo")
    print("=" * 60)

    checklist = PreTradeChecklist()

    # Scenario 1: Clean trade
    print("\n--- Scenario 1: Optimal conditions ---")
    r1 = checklist.quick_check(
        signal_triggered=True,
        setup_grade="A",
        risk_amount=200,
        current_equity=20000,
        stress_level=3,
        energy_level=8,
    )
    print(r1.report())

    # Scenario 2: High stress
    print("\n--- Scenario 2: High stress ---")
    r2 = checklist.quick_check(
        signal_triggered=True,
        setup_grade="A",
        stress_level=8,
        energy_level=6,
    )
    print(r2.report())

    # Scenario 3: FOMO
    print("\n--- Scenario 3: FOMO ---")
    r3 = checklist.quick_check(
        signal_triggered=True,
        setup_grade="B",
        feeling_fomo=True,
    )
    print(r3.report())

    # Scenario 4: Cooling-off
    print("\n--- Scenario 4: After 3 losses ---")
    r4 = checklist.quick_check(
        signal_triggered=True,
        setup_grade="A",
        consecutive_losses_today=3,
    )
    print(r4.report())

    # Scenario 5: Risk exceeded
    print("\n--- Scenario 5: Risk too high ---")
    r5 = checklist.quick_check(
        signal_triggered=True,
        setup_grade="A",
        risk_amount=500,  # 2.5% of $20k - exceeds 1% limit
        current_equity=20000,
    )
    print(r5.report())

    # Scenario 6: Setup grade C
    print("\n--- Scenario 6: Grade C setup ---")
    r6 = checklist.quick_check(
        signal_triggered=True,
        setup_grade="C",
    )
    print(r6.report())
