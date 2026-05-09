"""
QuantCFD - Chapter 11 - Solution Exercise 1
Personal bias assessment from trade history

Run BiasDetector on past trades, output personalized bias profile
with reflection prompts for self-analysis.
"""

from __future__ import annotations
from datetime import datetime
import pandas as pd
import numpy as np

from trade_journal import generate_synthetic_journal
from bias_detector import BiasDetector


def reflection_prompts(bias_findings) -> str:
    """Generate reflection prompts based on detected biases."""
    lines = ["", "=" * 70, "REFLECTION PROMPTS", "=" * 70]
    lines.append("\nFor each bias detected, journal your responses to these prompts.")
    lines.append("Be honest. Be specific. Use real trade examples.\n")

    prompts_by_bias = {
        "Loss Aversion": [
            "Remember the last time you moved a stop to breakeven prematurely. What were you feeling?",
            "What is the worst-case scenario you imagine when accepting a loss?",
            "Identity check: do you see losses as failure of self, or normal cost of doing business?",
            "Specific commitment: what mechanical rule will prevent this next time?",
        ],
        "Overconfidence": [
            "After your last 5-win streak, how did your position sizing change?",
            "What was the result when you sized up after wins?",
            "Identity check: do you attribute wins to skill or to favorable conditions?",
            "Specific commitment: what hardcoded rule prevents size increase from emotion?",
        ],
        "Revenge Trading": [
            "Describe in detail the last time you took a revenge trade. What triggered it?",
            "Physical signals: where did you feel the urgency in your body?",
            "Identity check: who am I when in revenge mode? Is this who I want to be?",
            "Specific commitment: what halt rule will fire automatically next time?",
        ],
        "Recency Bias": [
            "After your last losing streak, did you skip valid setups?",
            "How many trades samples do you need to evaluate strategy performance?",
            "Identity check: do I evaluate strategies on 5 trades or 100+?",
            "Specific commitment: minimum sample size before strategy decisions?",
        ],
        "P&L Obsession": [
            "How often do you check P&L during open trades?",
            "What feelings drive the urge to check?",
            "Identity check: am I a process executor or a P&L watcher?",
            "Specific commitment: what environment changes prevent constant checking?",
        ],
    }

    for finding in bias_findings:
        if finding.severity in ("MODERATE", "SEVERE"):
            lines.append(f"\n--- {finding.bias_name} ({finding.severity}) ---")
            for i, p in enumerate(prompts_by_bias.get(finding.bias_name, ["Reflect on this bias"]), 1):
                lines.append(f"  {i}. {p}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def countermeasures_action_plan(bias_findings) -> str:
    """Generate concrete action plan based on biases."""
    lines = ["", "=" * 70, "30-DAY ACTION PLAN", "=" * 70]
    lines.append("\nWeek-by-week focus to address detected biases:\n")

    # Sort by severity
    severe = [f for f in bias_findings if f.severity == "SEVERE"]
    moderate = [f for f in bias_findings if f.severity == "MODERATE"]

    week = 1

    # Week 1-2: address severe first
    for finding in severe[:1]:
        lines.append(f"\nWeek {week}-{week+1}: Address SEVERE - {finding.bias_name}")
        lines.append(f"  Recommendation: {finding.recommendation}")
        lines.append("  Action items:")
        if "Loss Aversion" in finding.bias_name:
            lines.append("    - Hardcode: no breakeven moves until 1.5R profit")
            lines.append("    - Set-and-forget orders mandatory")
            lines.append("    - Walk away after entry, no checking")
        elif "Revenge" in finding.bias_name:
            lines.append("    - Halt 2 hours after 3 consecutive losses")
            lines.append("    - Halt 24 hours after 5 losses")
            lines.append("    - Mandatory 10-min walk after each loss")
        elif "Overconfidence" in finding.bias_name:
            lines.append("    - Lock position size formula (no manual adjustments)")
            lines.append("    - 24-hour cooling off before any size change")
            lines.append("    - Mentor approval required for size increases")
        else:
            lines.append("    - Implement specific countermeasures from bias detector")
        week += 2

    # Week 3-4: address moderate
    for finding in moderate[:2]:
        lines.append(f"\nWeek {week}: Address MODERATE - {finding.bias_name}")
        lines.append(f"  Recommendation: {finding.recommendation}")
        week += 1

    if not severe and not moderate:
        lines.append("\n✓ No major biases detected. Maintain current discipline.")
        lines.append("  Action items:")
        lines.append("    - Continue daily journaling")
        lines.append("    - Weekly review pattern analysis")
        lines.append("    - Mental rehearsal scenarios")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def run_exercise_1(trade_history_df=None):
    """Run Exercise 1 — personal bias assessment."""
    print("=" * 70)
    print("EXERCISE 1: PERSONAL BIAS ASSESSMENT")
    print("=" * 70)
    print("""
Goal: identify your dominant cognitive biases from past trade history.
Time: 60 minutes

Steps:
  1. Generate (or load) trade history
  2. Run statistical bias detection
  3. Reflect on findings with prompts
  4. Build 30-day action plan
""")

    # If no history provided, generate synthetic
    if trade_history_df is None:
        print("\n--- Generating synthetic trade history ---")
        journal = generate_synthetic_journal(n_trades=80, seed=42)
        trade_history_df = journal.to_dataframe()

    print(f"Analyzing {len(trade_history_df)} trades...\n")

    # Run bias detection
    detector = BiasDetector(
        trade_history_df,
        backtest_avg_win=1.5,
        backtest_avg_loss=-1.0,
        backtest_win_rate=0.50,
    )

    print(detector.report())

    findings = detector.detect_all_biases()

    # Reflection prompts
    print(reflection_prompts(findings))

    # Action plan
    print(countermeasures_action_plan(findings))

    # Summary metrics
    severe_count = sum(1 for f in findings if f.severity == "SEVERE")
    moderate_count = sum(1 for f in findings if f.severity == "MODERATE")

    print(f"\n=== EXERCISE 1 SUMMARY ===")
    print(f"Biases detected: {severe_count} severe, {moderate_count} moderate")
    print(f"Estimated 90-day improvement potential:")
    if severe_count > 0:
        print(f"  Address severe biases first - 5-10% expected return improvement")
    if moderate_count > 0:
        print(f"  Address moderate biases - 2-5% expected return improvement")
    if severe_count == 0 and moderate_count == 0:
        print(f"  Foundation solid - focus on optimization")


if __name__ == "__main__":
    run_exercise_1()
