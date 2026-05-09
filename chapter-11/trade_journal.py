"""
QuantCFD - Chapter 11
trade_journal.py - Structured trade journal with pattern analysis

Per Mark Douglas, Brett Steenbarger: trade journal is #1 tool for improvement.
This module provides structured TradeJournal class with quantitative analysis.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Optional, List, Dict
import json

import numpy as np
import pandas as pd


@dataclass
class TradeEntry:
    """Single trade journal entry with full context."""

    # Identification
    trade_id: str
    date_entry: datetime
    asset: str
    strategy: str  # trend / mr / vol_bo

    # Setup
    entry_trigger: str
    confidence: int  # 1-5
    setup_grade: str  # A / B / C
    pre_trade_emotion: int  # 1-10 stress

    # Execution
    entry_price: float
    stop_loss: float
    profit_target: float
    position_size: float
    risk_amount: float

    # Outcome
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    r_multiple: Optional[float] = None
    followed_rules: bool = True

    # Process
    checked_pnl_during: bool = False
    wanted_to_override: bool = False
    override_taken: bool = False

    # Reflection
    what_went_right: str = ""
    what_could_improve: str = ""
    emotional_state_during: str = ""
    lesson_learned: str = ""
    is_mistake: bool = False  # True = process violation
    is_noise: bool = False  # True = unlucky outcome but process correct

    def close(self, exit_price: float, exit_time: Optional[datetime] = None):
        """Close trade and compute metrics."""
        self.exit_price = exit_price
        self.exit_time = exit_time or datetime.now()
        # Direction-aware P&L (assumes buy if entry < target, sell if entry > target)
        is_long = self.profit_target > self.entry_price
        if is_long:
            self.pnl = (exit_price - self.entry_price) * self.position_size
        else:
            self.pnl = (self.entry_price - exit_price) * self.position_size
        self.r_multiple = self.pnl / self.risk_amount if self.risk_amount else 0

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert datetime to ISO
        for k in ["date_entry", "exit_time"]:
            if d.get(k) and isinstance(d[k], datetime):
                d[k] = d[k].isoformat()
        return d


class TradeJournal:
    """
    Structured trade journal with pattern analysis.

    Tracks trades, computes metrics, identifies patterns over time.
    """

    def __init__(self):
        self.entries: List[TradeEntry] = []

    def add_trade(self, entry: TradeEntry) -> None:
        """Add new trade entry."""
        self.entries.append(entry)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert journal to DataFrame for analysis."""
        if not self.entries:
            return pd.DataFrame()
        rows = [e.to_dict() for e in self.entries]
        df = pd.DataFrame(rows)
        if "date_entry" in df.columns:
            df["date_entry"] = pd.to_datetime(df["date_entry"])
        return df

    def metrics(self) -> Dict[str, float]:
        """Compute aggregate metrics."""
        df = self.to_dataframe()
        if df.empty or "pnl" not in df.columns:
            return {}

        closed = df[df["pnl"].notna()]
        if closed.empty:
            return {"n_trades": 0, "n_open": len(df)}

        wins = closed[closed["pnl"] > 0]
        losses = closed[closed["pnl"] <= 0]

        return {
            "n_trades": int(len(closed)),
            "n_wins": int(len(wins)),
            "n_losses": int(len(losses)),
            "win_rate": float(len(wins) / len(closed)) if len(closed) > 0 else 0.0,
            "total_pnl": float(closed["pnl"].sum()),
            "avg_winner": float(wins["pnl"].mean()) if len(wins) > 0 else 0.0,
            "avg_loser": float(losses["pnl"].mean()) if len(losses) > 0 else 0.0,
            "avg_r_multiple": float(closed["r_multiple"].mean()),
            "best_trade": float(closed["pnl"].max()),
            "worst_trade": float(closed["pnl"].min()),
            "rule_adherence_pct": float(closed["followed_rules"].mean() * 100),
            "n_mistakes": int(closed["is_mistake"].sum()),
            "n_noise": int(closed["is_noise"].sum()),
        }

    def metrics_by_grade(self) -> pd.DataFrame:
        """Performance metrics grouped by setup grade (A/B/C)."""
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()
        closed = df[df["pnl"].notna()]
        if closed.empty:
            return pd.DataFrame()

        grouped = closed.groupby("setup_grade").agg(
            n_trades=("pnl", "count"),
            win_rate=("pnl", lambda x: (x > 0).mean()),
            avg_pnl=("pnl", "mean"),
            avg_r=("r_multiple", "mean"),
            total_pnl=("pnl", "sum"),
        )
        return grouped

    def metrics_by_strategy(self) -> pd.DataFrame:
        """Performance grouped by strategy."""
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()
        closed = df[df["pnl"].notna()]
        if closed.empty:
            return pd.DataFrame()

        return closed.groupby("strategy").agg(
            n_trades=("pnl", "count"),
            win_rate=("pnl", lambda x: (x > 0).mean()),
            total_pnl=("pnl", "sum"),
            avg_r=("r_multiple", "mean"),
        )

    def emotional_correlation(self) -> Dict[str, float]:
        """Correlate pre-trade emotion với outcome."""
        df = self.to_dataframe()
        if df.empty or len(df) < 5:
            return {}
        closed = df[df["pnl"].notna()]
        if len(closed) < 5:
            return {}

        # Higher pre_trade_emotion = more stress
        # Hypothesis: high stress correlates with worse outcomes
        corr_stress_pnl = closed["pre_trade_emotion"].corr(closed["pnl"])
        corr_confidence_pnl = closed["confidence"].corr(closed["pnl"])

        return {
            "stress_pnl_correlation": float(corr_stress_pnl) if not np.isnan(corr_stress_pnl) else 0.0,
            "confidence_pnl_correlation": float(corr_confidence_pnl) if not np.isnan(corr_confidence_pnl) else 0.0,
        }

    def report(self) -> str:
        """Generate formatted report."""
        lines = ["=" * 60]
        lines.append("TRADE JOURNAL REPORT")
        lines.append("=" * 60)

        m = self.metrics()
        if not m or m.get("n_trades", 0) == 0:
            return "\n".join(lines + ["No closed trades yet."])

        lines.append(f"\nTotal trades: {m['n_trades']}")
        lines.append(f"Wins/Losses: {m['n_wins']}/{m['n_losses']} (WR: {m['win_rate']*100:.1f}%)")
        lines.append(f"Total P&L: ${m['total_pnl']:,.2f}")
        lines.append(f"Avg winner: ${m['avg_winner']:,.2f}")
        lines.append(f"Avg loser: ${m['avg_loser']:,.2f}")
        lines.append(f"Avg R-multiple: {m['avg_r_multiple']:+.2f}")
        lines.append(f"Best/Worst: ${m['best_trade']:,.2f} / ${m['worst_trade']:,.2f}")
        lines.append(f"Rule adherence: {m['rule_adherence_pct']:.1f}%")
        lines.append(f"Mistakes vs noise: {m['n_mistakes']} mistakes, {m['n_noise']} noise")

        # By grade
        grade_df = self.metrics_by_grade()
        if not grade_df.empty:
            lines.append(f"\n--- By Setup Grade ---")
            lines.append(grade_df.to_string())

        # By strategy
        strat_df = self.metrics_by_strategy()
        if not strat_df.empty:
            lines.append(f"\n--- By Strategy ---")
            lines.append(strat_df.to_string())

        # Emotional correlation
        ec = self.emotional_correlation()
        if ec:
            lines.append(f"\n--- Emotional Correlations ---")
            for k, v in ec.items():
                lines.append(f"  {k}: {v:+.3f}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def save_json(self, filepath: str):
        """Save journal to JSON."""
        data = [e.to_dict() for e in self.entries]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)


def generate_synthetic_journal(n_trades: int = 50, seed: int = 42) -> TradeJournal:
    """Generate synthetic journal for testing."""
    rng = np.random.default_rng(seed)
    journal = TradeJournal()

    strategies = ["trend", "mr", "vol_bo"]
    grades = ["A", "B", "C"]
    grade_weights = [0.4, 0.4, 0.2]
    grade_win_rates = {"A": 0.62, "B": 0.50, "C": 0.38}

    base_date = datetime(2024, 1, 1)
    for i in range(n_trades):
        strategy = rng.choice(strategies)
        grade = rng.choice(grades, p=grade_weights)

        # Higher emotion on early trades (newer trader pattern)
        emotion = int(rng.uniform(2, 8))
        confidence = int(rng.uniform(2, 5))

        risk = 200.0
        position_size = 0.05
        entry_price = 2000.0 + rng.uniform(-50, 50)
        stop_loss = entry_price - 20  # Long bias
        profit_target = entry_price + 40

        entry = TradeEntry(
            trade_id=f"TR-{i+1:04d}",
            date_entry=base_date + pd.Timedelta(days=i),
            asset="XAUUSD",
            strategy=strategy,
            entry_trigger=f"{strategy} signal",
            confidence=confidence,
            setup_grade=grade,
            pre_trade_emotion=emotion,
            entry_price=entry_price,
            stop_loss=stop_loss,
            profit_target=profit_target,
            position_size=position_size,
            risk_amount=risk,
        )

        # Outcome based on grade
        wins = rng.random() < grade_win_rates[grade]
        if wins:
            exit_price = profit_target
        else:
            exit_price = stop_loss

        # Random rule violation 8% of trades
        followed = rng.random() > 0.08
        entry.followed_rules = followed
        entry.is_mistake = not followed
        entry.is_noise = followed and not wins

        entry.close(exit_price, exit_time=base_date + pd.Timedelta(days=i, hours=4))
        journal.add_trade(entry)

    return journal


if __name__ == "__main__":
    print("=" * 60)
    print("Trade Journal Demo")
    print("=" * 60)

    journal = generate_synthetic_journal(n_trades=100)
    print(journal.report())

    # Save sample
    journal.save_json("/tmp/sample_journal.json")
    print(f"\nSample journal saved: /tmp/sample_journal.json")
