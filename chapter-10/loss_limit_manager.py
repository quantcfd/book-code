"""
QuantCFD — Chương 10.9
Loss Limit Manager

Pre-committed circuit breakers: daily, weekly, monthly, total DD.
Halt trading when limits exceeded.
"""

from __future__ import annotations
import pandas as pd
from datetime import timedelta


class LossLimitManager:
    """
    Pre-committed loss limits + halt mechanism.

    Levels:
    - Daily: -3% (halt 24h)
    - Weekly: -7% (halt 7 days)
    - Monthly: -15% (halt 14 days)
    - Total DD: -20% (halt 30 days)
    - Terminal: -40% (stop trading, return to paper)
    """

    def __init__(
        self,
        initial_equity: float,
        daily_limit: float = -0.03,
        weekly_limit: float = -0.07,
        monthly_limit: float = -0.15,
        total_dd_limit: float = -0.20,
        terminal_limit: float = -0.40,
    ):
        self.initial = initial_equity
        self.peak = initial_equity
        self.daily_limit = daily_limit
        self.weekly_limit = weekly_limit
        self.monthly_limit = monthly_limit
        self.total_dd_limit = total_dd_limit
        self.terminal_limit = terminal_limit
        self.halts = []  # active halts: list of {reason, until}
        self.terminal = False
        self.equity_history = []

    def update_equity(self, equity: float, current_date: pd.Timestamp):
        """Update equity history and peak."""
        self.equity_history.append({
            "date": current_date,
            "equity": equity,
        })
        if equity > self.peak:
            self.peak = equity

    def _compute_pnl(
        self, current_equity: float, period_days: int, current_date: pd.Timestamp,
    ) -> float:
        """Compute P&L over last N days."""
        if not self.equity_history:
            return 0
        cutoff = current_date - timedelta(days=period_days)
        prior = [
            h for h in self.equity_history
            if h["date"] <= cutoff
        ]
        if not prior:
            start_equity = self.initial
        else:
            start_equity = prior[-1]["equity"]
        return (current_equity - start_equity) / start_equity if start_equity > 0 else 0

    def check_limits(
        self, current_equity: float, current_date: pd.Timestamp,
    ) -> dict:
        """
        Check all limits, return status dict.

        Returns:
            {
                "allow_trade": bool,
                "active_halts": list,
                "metrics": dict of current pnl vs limits,
            }
        """
        self.update_equity(current_equity, current_date)

        # Clear expired halts
        self.halts = [h for h in self.halts if h["until"] > current_date]

        # Check terminal first
        total_dd = (current_equity - self.peak) / self.peak if self.peak > 0 else 0
        if total_dd <= self.terminal_limit and not self.terminal:
            self.terminal = True
            self.halts.append({
                "reason": f"TERMINAL: DD {total_dd*100:.1f}% <= {self.terminal_limit*100}%",
                "until": current_date + timedelta(days=180),
                "level": "terminal",
            })

        # Compute period P&Ls
        daily_pnl = self._compute_pnl(current_equity, 1, current_date)
        weekly_pnl = self._compute_pnl(current_equity, 7, current_date)
        monthly_pnl = self._compute_pnl(current_equity, 30, current_date)

        # Check each limit (only add halt if not already present)
        existing_reasons = [h["reason"] for h in self.halts]

        if daily_pnl < self.daily_limit:
            reason = f"Daily loss {daily_pnl*100:.2f}% <= {self.daily_limit*100}%"
            if reason not in existing_reasons:
                self.halts.append({
                    "reason": reason,
                    "until": current_date + timedelta(days=1),
                    "level": "daily",
                })

        if weekly_pnl < self.weekly_limit:
            reason = f"Weekly loss {weekly_pnl*100:.2f}% <= {self.weekly_limit*100}%"
            if reason not in existing_reasons:
                self.halts.append({
                    "reason": reason,
                    "until": current_date + timedelta(days=7),
                    "level": "weekly",
                })

        if monthly_pnl < self.monthly_limit:
            reason = f"Monthly loss {monthly_pnl*100:.2f}% <= {self.monthly_limit*100}%"
            if reason not in existing_reasons:
                self.halts.append({
                    "reason": reason,
                    "until": current_date + timedelta(days=14),
                    "level": "monthly",
                })

        if total_dd < self.total_dd_limit and not self.terminal:
            reason = f"Total DD {total_dd*100:.2f}% <= {self.total_dd_limit*100}%"
            if reason not in existing_reasons:
                self.halts.append({
                    "reason": reason,
                    "until": current_date + timedelta(days=30),
                    "level": "total_dd",
                })

        return {
            "allow_trade": len(self.halts) == 0,
            "terminal": self.terminal,
            "active_halts": self.halts,
            "metrics": {
                "daily_pnl_pct": daily_pnl * 100,
                "weekly_pnl_pct": weekly_pnl * 100,
                "monthly_pnl_pct": monthly_pnl * 100,
                "total_dd_pct": total_dd * 100,
                "current_equity": current_equity,
                "peak_equity": self.peak,
            },
        }

    def status_report(self) -> str:
        """Pretty print current status."""
        lines = [
            "=" * 60,
            "LOSS LIMIT MANAGER STATUS",
            "=" * 60,
            f"Initial: ${self.initial:,.2f}",
            f"Peak:    ${self.peak:,.2f}",
        ]
        if self.equity_history:
            current = self.equity_history[-1]["equity"]
            lines.append(f"Current: ${current:,.2f}")
            lines.append(f"DD:      {(current - self.peak) / self.peak * 100:.2f}%")

        lines.append("\nLimits:")
        lines.append(f"  Daily:    {self.daily_limit * 100}%")
        lines.append(f"  Weekly:   {self.weekly_limit * 100}%")
        lines.append(f"  Monthly:  {self.monthly_limit * 100}%")
        lines.append(f"  Total DD: {self.total_dd_limit * 100}%")
        lines.append(f"  Terminal: {self.terminal_limit * 100}%")

        if self.halts:
            lines.append("\nActive halts:")
            for h in self.halts:
                lines.append(f"  - {h['reason']} (until {h['until']})")
        else:
            lines.append("\nNo active halts. Trading allowed.")

        if self.terminal:
            lines.append("\n*** TERMINAL LIMIT TRIGGERED — STOP TRADING ***")

        return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("Loss Limit Manager — Demo")
    print("=" * 60)

    mgr = LossLimitManager(initial_equity=10000)

    # Simulate scenarios
    scenarios = [
        ("Day 1", pd.Timestamp("2024-01-01"), 10100, "Normal day +1%"),
        ("Day 2", pd.Timestamp("2024-01-02"), 10250, "Continued +1.5%"),
        ("Day 3", pd.Timestamp("2024-01-03"), 9700, "Bad day -5.4% (triggers daily limit)"),
        ("Day 5", pd.Timestamp("2024-01-05"), 9650, "Halt active, no trading"),
        ("Day 10", pd.Timestamp("2024-01-10"), 9400, "More losses, weekly limit"),
        ("Day 30", pd.Timestamp("2024-01-30"), 9000, "Monthly check"),
        ("Day 60", pd.Timestamp("2024-03-01"), 8500, "Total DD -15%, approaching limit"),
        ("Day 90", pd.Timestamp("2024-04-01"), 7900, "Total DD -21%, halt triggered"),
    ]

    for label, date, equity, description in scenarios:
        print(f"\n{'─' * 60}")
        print(f"{label}: {description}")
        print(f"  Equity: ${equity:,}")
        result = mgr.check_limits(equity, date)
        print(f"  Allow trade: {result['allow_trade']}")
        if not result["allow_trade"]:
            print(f"  Reasons:")
            for h in result["active_halts"]:
                print(f"    - {h['reason']}")

    # Final status
    print(f"\n{mgr.status_report()}")
