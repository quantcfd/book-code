"""
QuantCFD — Chương 10
Combined Risk System

Integrates all risk components into single class:
- Position sizing (ATR-based)
- Loss limit manager
- DD control
- Vol targeting
- Correlation tracking
- Risk dashboard
- Kill switch
- Crisis playbook
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime

from position_sizing import atr_sized_position
from loss_limit_manager import LossLimitManager
from dd_control import dd_size_multiplier, streak_loss_multiplier
from vol_targeting import compute_realized_vol, vol_targeted_leverage
from correlation_tracker import correlation_matrix, detect_correlation_spike
from kill_switch import standard_trading_kill_switch
from crisis_playbook import CrisisPlaybook


# Strategy-specific risk config
STRATEGY_RISK_CONFIG = {
    "trend":  {"risk_pct": 0.010, "atr_stop_mult": 2.5, "max_dd_halt": 0.20},
    "mr":     {"risk_pct": 0.005, "atr_stop_mult": 1.5, "max_dd_halt": 0.15},
    "vol_bo": {"risk_pct": 0.007, "atr_stop_mult": 1.5, "max_dd_halt": 0.15},
}


class CombinedRiskSystem:
    """
    Full integrated risk management system for multi-strategy portfolio.
    """

    def __init__(
        self,
        initial_equity: float,
        strategies: list,
        target_vol_annual: float = 0.10,
    ):
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.peak_equity = initial_equity
        self.strategies = strategies
        self.target_vol = target_vol_annual

        # Component systems
        self.loss_mgr = LossLimitManager(initial_equity)
        self.kill_switch = standard_trading_kill_switch(initial_equity)
        self.crisis_pb = CrisisPlaybook()

        # State per strategy
        self.strategy_state = {
            name: {
                "equity": initial_equity / len(strategies),
                "peak": initial_equity / len(strategies),
                "consecutive_losses": 0,
                "is_halted": False,
                "halt_reason": None,
            }
            for name in strategies
        }

        # Returns history
        self.returns_history = {name: [] for name in strategies}

    def update_equity(
        self, current_equity: float, current_date: pd.Timestamp,
    ):
        self.current_equity = current_equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        self.loss_mgr.update_equity(current_equity, current_date)

    def update_strategy(
        self, strategy_name: str, returns_value: float, equity: float,
    ):
        """Update strategy-level state with new return."""
        if strategy_name not in self.strategy_state:
            return

        self.returns_history[strategy_name].append(returns_value)

        st = self.strategy_state[strategy_name]
        st["equity"] = equity
        if equity > st["peak"]:
            st["peak"] = equity

        if returns_value < 0:
            st["consecutive_losses"] += 1
        else:
            st["consecutive_losses"] = 0

        # Check strategy-level DD halt
        config = STRATEGY_RISK_CONFIG.get(strategy_name, {})
        max_halt = config.get("max_dd_halt", 0.20)
        current_dd = (equity - st["peak"]) / st["peak"] if st["peak"] > 0 else 0
        if current_dd <= -max_halt:
            st["is_halted"] = True
            st["halt_reason"] = f"Strategy DD {current_dd*100:.1f}% <= -{max_halt*100}%"

    def compute_position_size(
        self,
        strategy_name: str,
        atr: float,
        contract_value_per_point: float,
    ) -> dict:
        """
        Compute risk-adjusted position size for a trade.

        Returns:
            Dict with size, multipliers, reasoning.
        """
        if strategy_name not in self.strategy_state:
            return {"size": 0, "reason": f"unknown strategy {strategy_name}"}

        st = self.strategy_state[strategy_name]
        config = STRATEGY_RISK_CONFIG.get(
            strategy_name, {"risk_pct": 0.005, "atr_stop_mult": 2.0},
        )

        # Check halts
        if st["is_halted"]:
            return {"size": 0, "reason": st["halt_reason"]}

        # Base size from ATR
        base_size = atr_sized_position(
            equity=self.current_equity,
            risk_pct=config["risk_pct"],
            atr=atr,
            atr_stop_multiplier=config["atr_stop_mult"],
            contract_value_per_point=contract_value_per_point,
        )

        # DD-based scaling (per strategy)
        current_dd_strat = (
            (st["equity"] - st["peak"]) / st["peak"] if st["peak"] > 0 else 0
        )
        dd_mult = dd_size_multiplier(current_dd_strat)

        # Streak loss scaling
        streak_mult = streak_loss_multiplier(st["consecutive_losses"])

        # Vol targeting (portfolio level)
        if len(self.returns_history[strategy_name]) >= 30:
            recent_returns = pd.Series(self.returns_history[strategy_name])
            realized_vol = compute_realized_vol(recent_returns)
            vol_lev = vol_targeted_leverage(realized_vol, self.target_vol)
        else:
            vol_lev = 1.0

        # Final size
        final_size = base_size * dd_mult * streak_mult * vol_lev
        final_size = round(final_size, 2)

        return {
            "size": final_size,
            "base_size": base_size,
            "dd_multiplier": dd_mult,
            "streak_multiplier": streak_mult,
            "vol_leverage": vol_lev,
            "current_dd": current_dd_strat,
            "consecutive_losses": st["consecutive_losses"],
        }

    def can_trade(
        self,
        current_date: pd.Timestamp,
        current_state: dict = None,
    ) -> dict:
        """
        Master check: combined evaluation of all halts.
        """
        # Loss limit check
        loss_status = self.loss_mgr.check_limits(self.current_equity, current_date)

        # Kill switch check
        if current_state is None:
            current_state = {}
        current_state.setdefault("daily_loss_pct", 0)
        current_state.setdefault("current_equity", self.current_equity)
        kill_active = self.kill_switch.check(current_state)

        # Crisis phase check
        if current_state.get("market_state"):
            crisis_result = self.crisis_pb.transition(current_state["market_state"])
            crisis_phase = crisis_result["new_phase"]
        else:
            crisis_phase = "normal"

        allow = (
            loss_status["allow_trade"]
            and not kill_active
            and crisis_phase in ["normal", "reentry"]
        )

        reasons = []
        if not loss_status["allow_trade"]:
            reasons.extend([h["reason"] for h in loss_status["active_halts"]])
        if kill_active:
            reasons.append("Kill switch active")
        if crisis_phase not in ["normal", "reentry"]:
            reasons.append(f"Crisis phase: {crisis_phase}")

        return {
            "allow_trade": allow,
            "reasons": reasons,
            "loss_status": loss_status,
            "kill_switch_active": kill_active,
            "crisis_phase": crisis_phase,
        }

    def status_report(self):
        print("=" * 70)
        print(f"COMBINED RISK SYSTEM — {datetime.now()}")
        print("=" * 70)
        print(f"\nCapital:")
        print(f"  Initial:  ${self.initial_equity:,.2f}")
        print(f"  Current:  ${self.current_equity:,.2f}")
        print(f"  Peak:     ${self.peak_equity:,.2f}")
        port_dd = (self.current_equity - self.peak_equity) / self.peak_equity
        print(f"  DD:       {port_dd*100:.2f}%")

        print(f"\nStrategy states:")
        for name, st in self.strategy_state.items():
            dd = (st["equity"] - st["peak"]) / st["peak"] if st["peak"] > 0 else 0
            halt = "HALTED" if st["is_halted"] else "active"
            print(
                f"  {name:<10}: ${st['equity']:>9,.0f}  "
                f"DD {dd*100:>6.2f}%  "
                f"streak {st['consecutive_losses']:>2}  "
                f"[{halt}]"
            )

        ks_status = self.kill_switch.status()
        print(f"\nKill switch: {'ACTIVE' if ks_status['is_active'] else 'inactive'}")
        print(f"Crisis phase: {self.crisis_pb.phase.value}")


if __name__ == "__main__":
    print("=" * 70)
    print("Combined Risk System — Demo")
    print("=" * 70)

    crs = CombinedRiskSystem(
        initial_equity=25000,
        strategies=["trend", "mr", "vol_bo"],
        target_vol_annual=0.10,
    )

    # Simulate position size computation
    print(f"\n--- Computing position size for trend XAU trade ---")
    sizing = crs.compute_position_size(
        strategy_name="trend",
        atr=20,  # XAUUSD ATR ~$20
        contract_value_per_point=100,  # $100/point/lot
    )
    for k, v in sizing.items():
        print(f"  {k:<22}: {v}")

    # Check master can_trade
    print(f"\n--- Can trade check ---")
    result = crs.can_trade(
        current_date=pd.Timestamp("2024-09-15"),
        current_state={
            "daily_loss_pct": -0.01,
            "current_equity": 25000,
            "consecutive_losses": 0,
            "unexpected_position_count": 0,
            "normal_position_size": 0.05,
            "max_position_size": 0.05,
        },
    )
    print(f"  Allow trade: {result['allow_trade']}")
    if result["reasons"]:
        print(f"  Reasons:")
        for r in result["reasons"]:
            print(f"    - {r}")

    # Simulate strategy update
    print(f"\n--- Simulating 5 trades on trend ---")
    base_eq = 8333  # 25000 / 3 strategies
    for i in range(5):
        # 60% wins
        ret = 0.018 if i % 3 != 0 else -0.012
        base_eq *= (1 + ret)
        crs.update_strategy("trend", ret, base_eq)
    crs.update_equity(25000 + (base_eq - 8333), pd.Timestamp("2024-09-20"))

    # Final status
    print()
    crs.status_report()
