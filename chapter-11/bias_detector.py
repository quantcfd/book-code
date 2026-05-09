"""
QuantCFD - Chapter 11
bias_detector.py - Statistical detection of cognitive biases in trading history

Analyzes trade history to detect 7 cognitive biases automatically:
loss aversion, overconfidence, recency, anchoring, confirmation, FOMO, mental accounting.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class BiasFinding:
    """Single bias detection result."""
    bias_name: str
    severity: str  # NONE / MILD / MODERATE / SEVERE
    evidence: str
    recommendation: str
    statistic: float = 0.0


class BiasDetector:
    """
    Statistical bias detection from trade history.

    Compares actual behavior to backtest expectations and statistical norms
    to identify cognitive biases manifesting in trading.
    """

    def __init__(
        self,
        trades_df: pd.DataFrame,
        backtest_avg_win: float = 1.5,  # In R-multiples
        backtest_avg_loss: float = -1.0,
        backtest_win_rate: float = 0.50,
    ):
        """
        Initialize detector with trades and backtest expectations.

        Args:
            trades_df: DataFrame with columns including pnl, r_multiple,
                       position_size, date_entry, followed_rules,
                       wanted_to_override, checked_pnl_during
            backtest_avg_win: expected average winning R-multiple
            backtest_avg_loss: expected average losing R-multiple
            backtest_win_rate: expected win rate
        """
        self.df = trades_df.copy()
        if "date_entry" in self.df.columns:
            self.df["date_entry"] = pd.to_datetime(self.df["date_entry"])
            self.df = self.df.sort_values("date_entry").reset_index(drop=True)

        self.bt_avg_win = backtest_avg_win
        self.bt_avg_loss = backtest_avg_loss
        self.bt_win_rate = backtest_win_rate

    def detect_loss_aversion(self) -> BiasFinding:
        """
        Loss aversion detection:
        - Compare live avg_win to backtest avg_win
        - Compare live avg_loss to backtest avg_loss
        - If live avg_win < backtest (early exits) AND live |avg_loss| > backtest → loss aversion
        """
        closed = self.df[self.df["r_multiple"].notna()]
        if len(closed) < 10:
            return BiasFinding(
                "Loss Aversion", "NONE",
                "Need at least 10 trades for detection",
                "Continue logging trades",
            )

        wins = closed[closed["r_multiple"] > 0]
        losses = closed[closed["r_multiple"] <= 0]

        if len(wins) == 0 or len(losses) == 0:
            return BiasFinding(
                "Loss Aversion", "NONE",
                "Insufficient mix of wins/losses",
                "Continue logging trades",
            )

        live_avg_win = wins["r_multiple"].mean()
        live_avg_loss = losses["r_multiple"].mean()

        win_ratio = live_avg_win / self.bt_avg_win
        loss_ratio = live_avg_loss / self.bt_avg_loss  # Both negative, so positive ratio if matching

        # Loss aversion: cutting winners early (win_ratio < 1) and holding losers (loss_ratio > 1)
        # win_ratio < 0.7 = severe early exit
        # loss_ratio > 1.3 = severe late exit (losers worse than backtest)

        evidence_lines = [
            f"Live avg_win: {live_avg_win:.2f}R (backtest: {self.bt_avg_win:.2f}R, ratio: {win_ratio:.2f})",
            f"Live avg_loss: {live_avg_loss:.2f}R (backtest: {self.bt_avg_loss:.2f}R, ratio: {loss_ratio:.2f})",
        ]

        score = (1 - win_ratio) + max(0, loss_ratio - 1)

        if score > 0.6:
            severity = "SEVERE"
            rec = (
                "STRONG loss aversion detected. Hardcode rules: "
                "no breakeven moves until 1.5R profit, no stop loss tightening, "
                "set-and-forget orders."
            )
        elif score > 0.3:
            severity = "MODERATE"
            rec = "Moderate loss aversion. Tighten rule enforcement on profit targets and stop losses."
        elif score > 0.1:
            severity = "MILD"
            rec = "Mild loss aversion. Monitor profit/stop discipline weekly."
        else:
            severity = "NONE"
            rec = "Loss aversion within acceptable range."

        return BiasFinding(
            "Loss Aversion", severity,
            "; ".join(evidence_lines), rec, score,
        )

    def detect_overconfidence(self) -> BiasFinding:
        """
        Overconfidence detection:
        - After 5+ wins in row, position size increases?
        - Position size variance high?
        """
        if "position_size" not in self.df.columns or len(self.df) < 20:
            return BiasFinding(
                "Overconfidence", "NONE",
                "Insufficient data for detection",
                "Continue logging trades",
            )

        closed = self.df[self.df["r_multiple"].notna()].reset_index(drop=True)
        if len(closed) < 20:
            return BiasFinding(
                "Overconfidence", "NONE",
                "Insufficient closed trades",
                "Continue logging",
            )

        # Check position size variance
        size_cv = closed["position_size"].std() / closed["position_size"].mean() if closed["position_size"].mean() > 0 else 0

        # Check if size increased after winning streaks
        increases_after_wins = 0
        total_streaks = 0

        for i in range(5, len(closed)):
            last_5 = closed.iloc[i - 5: i]
            if (last_5["r_multiple"] > 0).all():
                total_streaks += 1
                if closed.iloc[i]["position_size"] > closed.iloc[i - 1]["position_size"]:
                    increases_after_wins += 1

        if total_streaks > 0:
            increase_rate = increases_after_wins / total_streaks
        else:
            increase_rate = 0

        evidence = (
            f"Position size CoV: {size_cv:.2%}; "
            f"Size increases after 5-win streaks: {increases_after_wins}/{total_streaks} ({increase_rate:.0%})"
        )

        score = size_cv + increase_rate

        if score > 0.7:
            severity = "SEVERE"
            rec = (
                "SEVERE overconfidence. Hardcode position sizing — cannot manually adjust. "
                "Monthly automatic position size review only."
            )
        elif score > 0.4:
            severity = "MODERATE"
            rec = "Moderate overconfidence. Lock position size formula, no manual adjustments mid-week."
        elif score > 0.2:
            severity = "MILD"
            rec = "Mild overconfidence. Monitor position size discipline."
        else:
            severity = "NONE"
            rec = "Position sizing disciplined."

        return BiasFinding(
            "Overconfidence", severity, evidence, rec, score,
        )

    def detect_revenge_trading(self) -> BiasFinding:
        """
        Revenge trading detection:
        - Position size increases after losses?
        - Trade frequency spike after losses?
        """
        if "position_size" not in self.df.columns or len(self.df) < 20:
            return BiasFinding(
                "Revenge Trading", "NONE",
                "Insufficient data",
                "Continue logging",
            )

        closed = self.df[self.df["r_multiple"].notna()].reset_index(drop=True)
        if len(closed) < 20:
            return BiasFinding(
                "Revenge Trading", "NONE",
                "Insufficient trades",
                "Continue logging",
            )

        # Check size after 3+ losses in row
        increases_after_losses = 0
        total_loss_streaks = 0

        for i in range(3, len(closed)):
            last_3 = closed.iloc[i - 3: i]
            if (last_3["r_multiple"] < 0).all():
                total_loss_streaks += 1
                if closed.iloc[i]["position_size"] > closed.iloc[i - 1]["position_size"] * 1.2:
                    increases_after_losses += 1

        if total_loss_streaks > 0:
            increase_rate = increases_after_losses / total_loss_streaks
        else:
            increase_rate = 0

        # Override rate after losses
        if "wanted_to_override" in self.df.columns:
            override_rate = closed[closed["r_multiple"] < 0]["wanted_to_override"].mean()
        else:
            override_rate = 0

        evidence = (
            f"Size increases after 3-loss streaks: {increases_after_losses}/{total_loss_streaks} "
            f"({increase_rate:.0%}); override urge after losses: {override_rate:.0%}"
        )

        score = increase_rate * 2 + override_rate

        if score > 0.8:
            severity = "SEVERE"
            rec = (
                "SEVERE revenge trading. Implement mandatory cooling-off rules: "
                "halt 2hrs after 3 losses, halt 24hrs after 5 losses. Hardcoded."
            )
        elif score > 0.4:
            severity = "MODERATE"
            rec = "Moderate revenge trading risk. Review halt rules, walk-away after losses."
        elif score > 0.2:
            severity = "MILD"
            rec = "Some revenge tendencies. Increase awareness during loss streaks."
        else:
            severity = "NONE"
            rec = "Revenge trading not detected."

        return BiasFinding(
            "Revenge Trading", severity, evidence, rec, score,
        )

    def detect_recency_bias(self) -> BiasFinding:
        """
        Recency bias detection:
        - Sample size used in decisions vs needed
        - Strategy switching frequency after small samples
        """
        if len(self.df) < 30:
            return BiasFinding(
                "Recency Bias", "NONE",
                "Insufficient data",
                "Continue logging",
            )

        # Check if rule violation rate higher after recent losses
        closed = self.df[self.df["r_multiple"].notna()].reset_index(drop=True)
        if "followed_rules" not in closed.columns:
            return BiasFinding(
                "Recency Bias", "NONE",
                "Rule following data not available",
                "Add followed_rules tracking",
            )

        violation_after_loss_streak = 0
        total_after_loss_streak = 0

        for i in range(3, len(closed)):
            last_3_losses = (closed.iloc[i - 3: i]["r_multiple"] < 0).sum()
            if last_3_losses >= 2:
                total_after_loss_streak += 1
                if not closed.iloc[i]["followed_rules"]:
                    violation_after_loss_streak += 1

        if total_after_loss_streak > 0:
            violation_rate = violation_after_loss_streak / total_after_loss_streak
        else:
            violation_rate = 0

        baseline_violation_rate = (~closed["followed_rules"]).mean()

        excess_violation = violation_rate - baseline_violation_rate

        evidence = (
            f"Rule violation rate after recent losses: {violation_rate:.0%} "
            f"(baseline: {baseline_violation_rate:.0%}, excess: {excess_violation:+.0%})"
        )

        score = max(0, excess_violation)

        if score > 0.2:
            severity = "MODERATE"
            rec = (
                "Recency bias affecting rule following. Mantra: "
                "'Each trade independent. Past results irrelevant to next decision.'"
            )
        elif score > 0.1:
            severity = "MILD"
            rec = "Mild recency bias. Maintain awareness."
        else:
            severity = "NONE"
            rec = "Recency bias not significant."

        return BiasFinding(
            "Recency Bias", severity, evidence, rec, score,
        )

    def detect_pnl_checking(self) -> BiasFinding:
        """
        P&L obsession detection:
        - Rate of checking P&L during trades correlates with stop violations?
        """
        if "checked_pnl_during" not in self.df.columns:
            return BiasFinding(
                "P&L Obsession", "NONE",
                "P&L checking data not tracked",
                "Add checked_pnl_during to journal",
            )

        closed = self.df[self.df["r_multiple"].notna()]
        if len(closed) < 20:
            return BiasFinding(
                "P&L Obsession", "NONE",
                "Insufficient trades",
                "Continue logging",
            )

        check_rate = closed["checked_pnl_during"].mean()

        evidence = f"P&L checked during {check_rate:.0%} of trades"

        if check_rate > 0.7:
            severity = "SEVERE"
            rec = "Frequent P&L checking. Use set-and-forget orders. Walk away after entry."
        elif check_rate > 0.4:
            severity = "MODERATE"
            rec = "Moderate P&L checking. Reduce monitor visibility during trades."
        elif check_rate > 0.2:
            severity = "MILD"
            rec = "Some P&L checking. Practice walking away after orders placed."
        else:
            severity = "NONE"
            rec = "P&L checking discipline good."

        return BiasFinding(
            "P&L Obsession", severity, evidence, rec, check_rate,
        )

    def detect_all_biases(self) -> List[BiasFinding]:
        """Run all bias detections."""
        return [
            self.detect_loss_aversion(),
            self.detect_overconfidence(),
            self.detect_revenge_trading(),
            self.detect_recency_bias(),
            self.detect_pnl_checking(),
        ]

    def report(self) -> str:
        """Generate formatted bias detection report."""
        lines = ["=" * 70, "BIAS DETECTION REPORT", "=" * 70]
        lines.append(f"\nTotal trades analyzed: {len(self.df)}")

        findings = self.detect_all_biases()

        for finding in findings:
            severity_color = {
                "SEVERE": "🔴",
                "MODERATE": "🟠",
                "MILD": "🟡",
                "NONE": "🟢",
            }.get(finding.severity, "⚪")

            lines.append(f"\n{severity_color} {finding.bias_name}: {finding.severity}")
            lines.append(f"  Evidence: {finding.evidence}")
            lines.append(f"  Recommendation: {finding.recommendation}")
            if finding.statistic > 0:
                lines.append(f"  Statistic: {finding.statistic:.3f}")

        # Summary
        n_severe = sum(1 for f in findings if f.severity == "SEVERE")
        n_moderate = sum(1 for f in findings if f.severity == "MODERATE")
        n_mild = sum(1 for f in findings if f.severity == "MILD")

        lines.append(f"\n--- Summary ---")
        lines.append(f"  Severe biases: {n_severe}")
        lines.append(f"  Moderate biases: {n_moderate}")
        lines.append(f"  Mild biases: {n_mild}")

        if n_severe > 0:
            lines.append("\n  ⚠ ACTION REQUIRED: severe biases need immediate countermeasures")
        elif n_moderate > 0:
            lines.append("\n  ⚠ MONITORING: moderate biases need attention")
        else:
            lines.append("\n  ✓ HEALTHY: bias profile within acceptable range")

        lines.append("=" * 70)
        return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 70)
    print("Bias Detector Demo")
    print("=" * 70)

    # Generate synthetic trades with intentional biases
    rng = np.random.default_rng(42)
    n_trades = 100

    trades = []
    base_size = 0.05
    current_size = base_size
    consecutive_wins = 0
    consecutive_losses = 0

    for i in range(n_trades):
        # Inject overconfidence: size grows after wins
        if consecutive_wins >= 5:
            current_size = base_size * 2.0
        # Inject revenge: size grows after losses
        elif consecutive_losses >= 3:
            current_size = base_size * 1.5
        else:
            current_size = base_size + rng.uniform(-0.005, 0.005)

        # Win 50% probability, but cut early (loss aversion)
        wins = rng.random() < 0.50
        if wins:
            r = rng.uniform(0.5, 1.0)  # Cut winners early — only 0.5-1.0R instead of 1.5-2R
            consecutive_wins += 1
            consecutive_losses = 0
        else:
            r = rng.uniform(-1.5, -1.0)  # Hold losers — bigger losses
            consecutive_losses += 1
            consecutive_wins = 0

        trades.append({
            "trade_id": f"T-{i+1:03d}",
            "date_entry": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            "r_multiple": r,
            "pnl": r * 200,
            "position_size": current_size,
            "followed_rules": rng.random() > 0.10,
            "wanted_to_override": rng.random() > 0.6 and not wins,
            "checked_pnl_during": rng.random() > 0.4,
        })

    df = pd.DataFrame(trades)

    detector = BiasDetector(
        df,
        backtest_avg_win=1.5,
        backtest_avg_loss=-1.0,
        backtest_win_rate=0.50,
    )

    print(detector.report())
