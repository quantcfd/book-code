"""
QuantCFD - Chapter 11
emotion_tracker.py - 5-dimensional emotional state tracking with correlation analysis

Track stress, confidence, patience, focus, energy 3x daily.
Identify patterns: emotional state ↔ trade quality.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class EmotionReading:
    """Single emotion reading at point in time."""
    timestamp: datetime
    stress: int  # 1-10 (10 = panic)
    confidence: int  # 1-10
    patience: int  # 1-10
    focus: int  # 1-10
    energy: int  # 1-10
    notes: str = ""

    @property
    def composite_score(self) -> float:
        """
        Composite emotional score (0-10).

        Stress is inverted (low stress = good).
        Others: higher = better.
        Composite higher = better state.
        """
        return (
            (10 - self.stress)  # Inverted
            + self.confidence
            + self.patience
            + self.focus
            + self.energy
        ) / 5

    @property
    def is_high_stress(self) -> bool:
        return self.stress >= 7

    @property
    def is_low_energy(self) -> bool:
        return self.energy <= 4

    @property
    def is_low_patience(self) -> bool:
        return self.patience <= 4

    @property
    def trading_recommendation(self) -> str:
        """Recommend trading action based on state."""
        if self.is_high_stress:
            return "HALT — stress too high, do not trade"
        if self.is_low_energy:
            return "HALT — energy too low, decisions degraded"
        if self.is_low_patience:
            return "CAUTION — patience low, revenge trading risk"
        if self.composite_score < 5:
            return "CAUTION — overall emotional state suboptimal"
        if self.composite_score >= 7:
            return "OPTIMAL — good state for trading"
        return "OK — acceptable state, proceed normally"


class EmotionTracker:
    """
    5-dimensional emotion tracking system.

    Records readings, identifies patterns, correlates with trade outcomes.
    """

    def __init__(self):
        self.readings: List[EmotionReading] = []

    def log_reading(self, reading: EmotionReading) -> None:
        """Add new emotion reading."""
        self.readings.append(reading)

    def log(
        self,
        stress: int,
        confidence: int,
        patience: int,
        focus: int,
        energy: int,
        notes: str = "",
        timestamp: Optional[datetime] = None,
    ) -> EmotionReading:
        """Convenient logging method."""
        reading = EmotionReading(
            timestamp=timestamp or datetime.now(),
            stress=stress,
            confidence=confidence,
            patience=patience,
            focus=focus,
            energy=energy,
            notes=notes,
        )
        self.log_reading(reading)
        return reading

    def to_dataframe(self) -> pd.DataFrame:
        """Convert readings to DataFrame."""
        if not self.readings:
            return pd.DataFrame()
        rows = [asdict(r) for r in self.readings]
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["composite"] = df.apply(
            lambda r: ((10 - r["stress"]) + r["confidence"] + r["patience"] + r["focus"] + r["energy"]) / 5,
            axis=1,
        )
        return df

    def latest_reading(self) -> Optional[EmotionReading]:
        """Get most recent reading."""
        return self.readings[-1] if self.readings else None

    def daily_average(self, target_date: Optional[datetime] = None) -> Dict[str, float]:
        """Average readings for a specific day (or all days)."""
        df = self.to_dataframe()
        if df.empty:
            return {}

        if target_date:
            day_str = target_date.date().isoformat() if isinstance(target_date, datetime) else str(target_date)
            df = df[df["timestamp"].dt.date.astype(str) == day_str]
            if df.empty:
                return {}

        return {
            "stress": float(df["stress"].mean()),
            "confidence": float(df["confidence"].mean()),
            "patience": float(df["patience"].mean()),
            "focus": float(df["focus"].mean()),
            "energy": float(df["energy"].mean()),
            "composite": float(df["composite"].mean()),
        }

    def trends(self, days: int = 14) -> pd.DataFrame:
        """Compute daily averages for trend analysis."""
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()

        df["date"] = df["timestamp"].dt.date
        daily = df.groupby("date").agg(
            stress=("stress", "mean"),
            confidence=("confidence", "mean"),
            patience=("patience", "mean"),
            focus=("focus", "mean"),
            energy=("energy", "mean"),
            composite=("composite", "mean"),
        )
        return daily.tail(days)

    def correlate_with_trades(self, trade_journal_df: pd.DataFrame) -> Dict[str, float]:
        """
        Correlate emotional state với trade outcomes.

        Args:
            trade_journal_df: DataFrame with date_entry and pnl columns

        Returns:
            Dict of correlations
        """
        emo_df = self.to_dataframe()
        if emo_df.empty or trade_journal_df.empty:
            return {}

        # Aggregate emotions to daily averages
        emo_df["date"] = emo_df["timestamp"].dt.date
        daily_emo = emo_df.groupby("date").agg(
            stress=("stress", "mean"),
            confidence=("confidence", "mean"),
            patience=("patience", "mean"),
            energy=("energy", "mean"),
            composite=("composite", "mean"),
        )

        # Aggregate trades to daily P&L
        trades = trade_journal_df.copy()
        if "pnl" not in trades.columns or "date_entry" not in trades.columns:
            return {}
        trades = trades[trades["pnl"].notna()]
        if trades.empty:
            return {}
        trades["date"] = pd.to_datetime(trades["date_entry"]).dt.date
        daily_pnl = trades.groupby("date")["pnl"].sum()

        # Merge và correlate
        merged = daily_emo.join(daily_pnl, how="inner")
        if len(merged) < 5:
            return {}

        return {
            "stress_pnl": float(merged["stress"].corr(merged["pnl"])),
            "confidence_pnl": float(merged["confidence"].corr(merged["pnl"])),
            "patience_pnl": float(merged["patience"].corr(merged["pnl"])),
            "energy_pnl": float(merged["energy"].corr(merged["pnl"])),
            "composite_pnl": float(merged["composite"].corr(merged["pnl"])),
        }

    def detect_patterns(self) -> List[str]:
        """Identify emotional patterns from readings."""
        df = self.to_dataframe()
        if len(df) < 10:
            return ["Need at least 10 readings for pattern detection"]

        patterns = []

        # Pattern 1: chronically high stress
        avg_stress = df["stress"].mean()
        if avg_stress > 6:
            patterns.append(
                f"⚠ Chronic high stress: average {avg_stress:.1f}/10. "
                "Consider lifestyle changes (sleep, exercise, smaller positions)."
            )

        # Pattern 2: low energy days
        low_energy_pct = (df["energy"] <= 4).mean() * 100
        if low_energy_pct > 30:
            patterns.append(
                f"⚠ Low energy frequent: {low_energy_pct:.0f}% of readings. "
                "Investigate sleep, diet, or burnout."
            )

        # Pattern 3: stress trending up
        if len(df) >= 20:
            recent_stress = df.tail(10)["stress"].mean()
            earlier_stress = df.head(10)["stress"].mean()
            if recent_stress > earlier_stress + 1.5:
                patterns.append(
                    f"⚠ Stress trending up: {earlier_stress:.1f} → {recent_stress:.1f}. "
                    "Consider what changed."
                )

        # Pattern 4: patience low (revenge risk)
        avg_patience = df["patience"].mean()
        if avg_patience < 5:
            patterns.append(
                f"⚠ Low patience: average {avg_patience:.1f}/10. "
                "Revenge trading risk elevated. Review halt rules."
            )

        # Pattern 5: confidence overcalibrated
        avg_confidence = df["confidence"].mean()
        if avg_confidence > 8:
            patterns.append(
                f"⚠ High confidence: average {avg_confidence:.1f}/10. "
                "Watch for overconfidence — review position sizing discipline."
            )

        if not patterns:
            patterns.append("✓ No concerning patterns detected. State stable.")

        return patterns

    def report(self) -> str:
        """Generate formatted report."""
        lines = ["=" * 60, "EMOTION TRACKING REPORT", "=" * 60]

        if not self.readings:
            return "\n".join(lines + ["No readings recorded."])

        lines.append(f"\nTotal readings: {len(self.readings)}")
        lines.append(f"Date range: {self.readings[0].timestamp.date()} to {self.readings[-1].timestamp.date()}")

        latest = self.latest_reading()
        if latest:
            lines.append(f"\n--- Latest Reading ---")
            lines.append(f"  Time: {latest.timestamp}")
            lines.append(f"  Stress: {latest.stress}/10")
            lines.append(f"  Confidence: {latest.confidence}/10")
            lines.append(f"  Patience: {latest.patience}/10")
            lines.append(f"  Focus: {latest.focus}/10")
            lines.append(f"  Energy: {latest.energy}/10")
            lines.append(f"  Composite: {latest.composite_score:.1f}/10")
            lines.append(f"  Recommendation: {latest.trading_recommendation}")

        avg = self.daily_average()
        if avg:
            lines.append(f"\n--- Overall Average ---")
            for k, v in avg.items():
                lines.append(f"  {k:15s}: {v:.2f}")

        patterns = self.detect_patterns()
        if patterns:
            lines.append(f"\n--- Detected Patterns ---")
            for p in patterns:
                lines.append(f"  {p}")

        lines.append("=" * 60)
        return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("Emotion Tracker Demo")
    print("=" * 60)

    tracker = EmotionTracker()
    rng = np.random.default_rng(42)

    # Simulate 14 days, 3 readings/day
    base = datetime(2024, 6, 1, 9, 0)
    for day in range(14):
        # Morning, midday, evening
        for hour in [9, 12, 17]:
            tracker.log(
                stress=int(np.clip(rng.normal(5, 1.5), 1, 10)),
                confidence=int(np.clip(rng.normal(6, 1.0), 1, 10)),
                patience=int(np.clip(rng.normal(6, 1.5), 1, 10)),
                focus=int(np.clip(rng.normal(7, 1.0), 1, 10)),
                energy=int(np.clip(rng.normal(6, 1.5), 1, 10)),
                timestamp=base + timedelta(days=day, hours=hour - 9),
            )

    print(tracker.report())

    # Show trends
    print("\n--- 14-day trends ---")
    print(tracker.trends().round(2))
