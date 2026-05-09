"""
QuantCFD - Chapter 11
habit_tracker.py - Daily habit tracking with streaks and identity-based metrics

Track 5 critical trader habits per James Clear (Atomic Habits):
journal, checklist, weekly review, mentor consultation, exercise.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, timedelta, datetime
from typing import Dict, List, Optional, Set

import pandas as pd


@dataclass
class HabitDefinition:
    """Definition of a habit to track."""
    name: str
    description: str
    target_frequency: str  # "daily" / "weekly"
    expected_per_week: int  # 7 for daily, 1 for weekly etc


# Standard 5 critical trader habits
DEFAULT_TRADER_HABITS = [
    HabitDefinition(
        "journal",
        "Daily trade journal entry",
        "daily", 7,
    ),
    HabitDefinition(
        "checklist",
        "Pre-trade checklist completion (every trade)",
        "daily", 7,
    ),
    HabitDefinition(
        "weekly_review",
        "Weekly trade pattern review",
        "weekly", 1,
    ),
    HabitDefinition(
        "mentor_call",
        "Mentor consultation",
        "weekly", 1,
    ),
    HabitDefinition(
        "exercise",
        "Physical exercise (30+ min)",
        "daily", 4,
    ),
]


class HabitTracker:
    """
    Track daily habits with streak management and statistics.

    Identity-based: focus on consistency, not perfection.
    """

    def __init__(self, habits: Optional[List[HabitDefinition]] = None):
        self.habits = {h.name: h for h in (habits or DEFAULT_TRADER_HABITS)}
        # logs[habit_name] = set of dates completed
        self.logs: Dict[str, Set[date]] = {name: set() for name in self.habits}

    def log_habit(self, habit_name: str, completion_date: Optional[date] = None) -> None:
        """Log habit completion."""
        if habit_name not in self.habits:
            raise ValueError(f"Unknown habit: {habit_name}")
        d = completion_date or date.today()
        if isinstance(d, datetime):
            d = d.date()
        self.logs[habit_name].add(d)

    def did_habit_today(self, habit_name: str) -> bool:
        """Check if habit done today."""
        return date.today() in self.logs.get(habit_name, set())

    def current_streak(self, habit_name: str) -> int:
        """Current consecutive days streak (only meaningful for daily habits)."""
        if habit_name not in self.logs or not self.logs[habit_name]:
            return 0

        habit = self.habits[habit_name]
        if habit.target_frequency != "daily":
            return 0

        completed = self.logs[habit_name]
        # Start from today, count back
        streak = 0
        check_date = date.today()
        while check_date in completed:
            streak += 1
            check_date -= timedelta(days=1)
        return streak

    def longest_streak(self, habit_name: str) -> int:
        """Longest consecutive days streak ever."""
        if habit_name not in self.logs or not self.logs[habit_name]:
            return 0

        completed = sorted(self.logs[habit_name])
        max_streak = 1
        current = 1
        for i in range(1, len(completed)):
            if (completed[i] - completed[i - 1]).days == 1:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 1
        return max_streak

    def completion_rate(self, habit_name: str, days_back: int = 30) -> float:
        """
        Compute completion rate over recent period.

        For daily habits: pct of days in last N days that habit was done.
        For weekly habits: pct of weeks in last N days.
        """
        if habit_name not in self.habits:
            return 0.0
        habit = self.habits[habit_name]
        completed = self.logs.get(habit_name, set())

        end_date = date.today()
        start_date = end_date - timedelta(days=days_back - 1)

        if habit.target_frequency == "daily":
            relevant_completions = sum(
                1 for d in completed if start_date <= d <= end_date
            )
            return relevant_completions / days_back
        else:  # weekly
            n_weeks = days_back // 7
            if n_weeks == 0:
                return 0.0
            weeks_completed = set()
            for d in completed:
                if start_date <= d <= end_date:
                    week_start = d - timedelta(days=d.weekday())
                    weeks_completed.add(week_start)
            return len(weeks_completed) / n_weeks

    def identity_strength(self, habit_name: str) -> float:
        """
        Identity-based metric: 0-100 score for habit identity strength.

        Combines: streak, completion rate, consistency over time.
        """
        if habit_name not in self.habits:
            return 0.0

        rate = self.completion_rate(habit_name, days_back=60)
        longest = min(self.longest_streak(habit_name), 30) / 30  # cap at 30 days
        current = min(self.current_streak(habit_name), 30) / 30

        # Weighted combination
        score = (rate * 0.5 + longest * 0.25 + current * 0.25) * 100
        return float(score)

    def report(self) -> str:
        """Generate habit tracking report."""
        lines = ["=" * 70, "HABIT TRACKER REPORT", "=" * 70]

        for name, habit in self.habits.items():
            status_today = "✓" if self.did_habit_today(name) else "✗"
            streak = self.current_streak(name)
            longest = self.longest_streak(name)
            rate_30 = self.completion_rate(name, 30) * 100
            identity = self.identity_strength(name)

            lines.append(f"\n{status_today} {name.upper()}")
            lines.append(f"  Description: {habit.description}")
            lines.append(f"  Frequency: {habit.target_frequency} (target: {habit.expected_per_week}/week)")
            lines.append(f"  Today: {'Done' if self.did_habit_today(name) else 'Not done'}")
            if habit.target_frequency == "daily":
                lines.append(f"  Current streak: {streak} days")
                lines.append(f"  Longest streak: {longest} days")
            lines.append(f"  30-day completion rate: {rate_30:.0f}%")
            lines.append(f"  Identity strength: {identity:.0f}/100")

            if identity >= 80:
                lines.append("  Status: ✓ Strong identity habit")
            elif identity >= 60:
                lines.append("  Status: ✓ Established habit")
            elif identity >= 40:
                lines.append("  Status: ⚠ Building habit")
            else:
                lines.append("  Status: ✗ Habit not yet established")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 70)
    print("Habit Tracker Demo")
    print("=" * 70)

    tracker = HabitTracker()

    # Simulate 60 days of habit logging
    today = date.today()
    import random
    random.seed(42)

    # Strong journal habit (95% completion)
    for i in range(60):
        d = today - timedelta(days=i)
        if random.random() < 0.95:
            tracker.log_habit("journal", d)

    # Strong checklist habit (90%)
    for i in range(60):
        d = today - timedelta(days=i)
        if random.random() < 0.90:
            tracker.log_habit("checklist", d)

    # Weekly review habit (every Saturday)
    for i in range(60):
        d = today - timedelta(days=i)
        if d.weekday() == 5:  # Saturday
            if random.random() < 0.85:
                tracker.log_habit("weekly_review", d)

    # Mentor call habit (weekly Tuesday)
    for i in range(60):
        d = today - timedelta(days=i)
        if d.weekday() == 1:
            if random.random() < 0.80:
                tracker.log_habit("mentor_call", d)

    # Exercise (60% completion - target 4/week)
    for i in range(60):
        d = today - timedelta(days=i)
        if random.random() < 0.60:
            tracker.log_habit("exercise", d)

    print(tracker.report())
