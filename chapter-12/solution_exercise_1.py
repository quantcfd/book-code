"""
Bài tập 1 (Beginner) - Pre-launch checklist self-audit
=========================================================

Goal: Run through 78-item checklist, mark current state,
identify gaps, build action plan.

QuantCFD Chapter 12 Capstone exercise.
"""
from datetime import datetime
from pre_launch_checklist import PreLaunchChecklist, ChecklistSection


def perform_self_audit():
    """Walk through full 78-item self-audit."""
    print("=" * 70)
    print("EXERCISE 1: PRE-LAUNCH SELF-AUDIT")
    print("=" * 70)

    checklist = PreLaunchChecklist()

    # Simulated trader self-audit responses
    # In real use: trader fills out manually
    audit_responses = {
        # Section A: Data infrastructure
        'A1': True,   # Real-time price feed working
        'A2': True,   # Historical data 5+ years stored
        'A3': True,   # Data cleaning pipeline tested
        'A4': True,   # Timezone handling correct
        'A5': True,   # Gap detection automated
        'A6': False,  # Multi-broker comparison
        'A7': True,   # News integration
        'A8': True,   # Data backup automated
        'A9': False,  # Fallback data source
        'A10': True,  # Data quality monitoring
        'A11': True,  # Storage capacity
        'A12': False, # Recovery procedure documented

        # Section B: Strategy validation (mostly complete)
        **{f'B{i}': True for i in range(1, 14)},
        'B14': True,  # Strategy code unit tested
        'B15': False, # Code peer reviewed

        # Section C: Risk infrastructure
        **{f'C{i}': True for i in range(1, 11)},
        'C11': False,  # Risk dashboard real-time
        'C12': False,  # Manual override documented

        # Section D: Psychology
        **{f'D{i}': True for i in range(1, 7)},  # Most done
        'D7': False,   # Mental rehearsal scheduled
        'D8': False,   # Mentor relationship established
        'D9': True,    # Family awareness
        'D10': True,   # Lifestyle foundations
        'D11': False,  # Identity work
        'D12': False,  # DD psychology rehearsed

        # Section E: Execution
        **{f'E{i}': True for i in range(1, 11)},
        'E11': True,   # Multi-asset
        'E12': False,  # After-hours
        'E13': False,  # Weekend gap
        'E14': False,  # Failover broker
        'E15': False,  # Recovery procedure

        # Section F: Operations
        **{f'F{i}': True for i in range(1, 8)},
        'F8': True,    # Backup procedures
        'F9': True,    # Git committed
        'F10': False,  # Documentation up to date
        'F11': False,  # Disaster recovery tested
        'F12': True,   # Personal financial buffer
    }

    # Apply responses
    for item_id, completed in audit_responses.items():
        if completed:
            checklist.mark_completed(item_id, notes="Self-audit completed")

    # Generate report
    print(checklist.generate_report())

    # Detail incomplete items by section
    print("\n\nDETAIL OF INCOMPLETE ITEMS:")
    print("-" * 70)
    incomplete = checklist.get_incomplete_items()

    by_section = {}
    for item in incomplete:
        section = item.section.value
        if section not in by_section:
            by_section[section] = []
        by_section[section].append(item)

    section_priority = {
        ChecklistSection.RISK.value: 1,           # Most critical
        ChecklistSection.PSYCHOLOGY.value: 2,
        ChecklistSection.EXECUTION.value: 3,
        ChecklistSection.OPERATIONS.value: 4,
        ChecklistSection.DATA.value: 5,
        ChecklistSection.STRATEGY.value: 6,
    }

    sorted_sections = sorted(by_section.items(), key=lambda x: section_priority.get(x[0], 99))

    for section_key, items in sorted_sections:
        print(f"\n{section_key}:")
        for item in items:
            print(f"  ✗ {item.item_id}: {item.description}")

    # Action plan
    print("\n\nACTION PLAN — recommended order:")
    print("-" * 70)
    print("Priority 1 (BLOCKERS - cannot deploy without):")
    print("  - Mentor relationship (D8)")
    print("  - Risk dashboard (C11)")
    print("  - Failover broker (E14)")
    print("  - Recovery procedures (E15, A12)")
    print()
    print("Priority 2 (Important - 2 weeks):")
    print("  - DD psychology rehearsal (D12)")
    print("  - Identity work (D11)")
    print("  - Mental rehearsal scenarios (D7)")
    print("  - Manual override docs (C12)")
    print("  - Disaster recovery test (F11)")
    print()
    print("Priority 3 (Nice to have - 1 month):")
    print("  - Multi-broker comparison (A6)")
    print("  - Documentation update (F10)")
    print("  - After-hours/weekend handling (E12, E13)")
    print("  - Code peer review (B15)")
    print("  - Fallback data source (A9)")

    # Time estimate
    total_incomplete = len(incomplete)
    print(f"\n\nTime estimate to complete ALL {total_incomplete} items:")
    print(f"  Aggressive (full-time):   2-3 weeks")
    print(f"  Realistic (part-time):    4-6 weeks")
    print(f"  Conservative (10 hrs/wk): 6-10 weeks")

    print("\n" + "=" * 70)
    print("Self-audit complete. Address gaps before live deployment.")
    print("=" * 70)


if __name__ == "__main__":
    perform_self_audit()
