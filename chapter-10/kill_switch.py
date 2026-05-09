"""
QuantCFD — Chương 10.12
Kill Switch — emergency halt mechanism

Multiple triggers, single action: halt all trading immediately.
"""

from __future__ import annotations
from datetime import datetime
from typing import Callable


class KillSwitch:
    """
    Emergency halt mechanism for trading systems.
    Multiple triggers, single action: halt + alert + lock.
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self.triggers = []  # list of (condition_fn, name)
        self.is_active = False
        self.activation_history = []
        self.lock_until_manual_reset = False

    def add_trigger(self, condition_fn: Callable[[dict], bool], name: str):
        """Add a trigger condition. condition_fn takes state dict, returns bool."""
        self.triggers.append((condition_fn, name))

    def check(self, state: dict) -> bool:
        """
        Run all triggers. Activate if any True.

        Args:
            state: Dict of current system state.

        Returns:
            True if kill switch activated.
        """
        if self.lock_until_manual_reset:
            return True

        for condition_fn, trigger_name in self.triggers:
            try:
                if condition_fn(state):
                    self.activate(trigger_name, state)
                    return True
            except Exception as e:
                print(f"Trigger '{trigger_name}' raised exception: {e}")

        return False

    def activate(self, reason: str, state: dict = None):
        """Activate kill switch."""
        self.is_active = True
        self.lock_until_manual_reset = True
        timestamp = datetime.now()

        self.activation_history.append({
            "timestamp": timestamp,
            "reason": reason,
            "state_snapshot": state or {},
        })

        print(f"╔{'═' * 60}╗")
        print(f"║ KILL SWITCH ACTIVATED: {self.name}")
        print(f"║ Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"║ Reason: {reason}")
        print(f"╚{'═' * 60}╝")

        # In production, these would actually execute:
        # 1. Cancel all pending orders
        # 2. Close all open positions (if appropriate for trigger)
        # 3. Send alert email/SMS/Telegram
        # 4. Lock trading account
        # 5. Log to audit trail

    def manual_reset(self, password: str = None) -> bool:
        """Manual reset — only way to re-enable trading."""
        # In production: require strong authentication
        # For demo: just clear flags
        self.is_active = False
        self.lock_until_manual_reset = False
        print(f"Kill switch '{self.name}' manually reset")
        return True

    def status(self) -> dict:
        return {
            "name": self.name,
            "is_active": self.is_active,
            "locked": self.lock_until_manual_reset,
            "n_triggers": len(self.triggers),
            "n_activations": len(self.activation_history),
            "last_activation": (
                self.activation_history[-1] if self.activation_history else None
            ),
        }


def standard_trading_kill_switch(initial_equity: float) -> KillSwitch:
    """
    Build kill switch with standard trading triggers.

    Triggers:
    - Daily loss > 5%
    - Account balance < 50% of initial (catastrophic)
    - 10 consecutive losses
    - Unexpected position count > 0
    - Single position size > 2x normal (fat finger)
    """
    ks = KillSwitch(name="trading_safety")

    ks.add_trigger(
        lambda s: s.get("daily_loss_pct", 0) < -0.05,
        "Daily loss > 5%",
    )
    ks.add_trigger(
        lambda s: s.get("current_equity", initial_equity) < initial_equity * 0.5,
        "Account < 50% initial (catastrophic)",
    )
    ks.add_trigger(
        lambda s: s.get("consecutive_losses", 0) >= 10,
        "10 consecutive losses",
    )
    ks.add_trigger(
        lambda s: s.get("unexpected_position_count", 0) > 0,
        "Unexpected position(s) detected",
    )
    ks.add_trigger(
        lambda s: s.get("max_position_size", 0) > s.get("normal_position_size", 1) * 2,
        "Position size 2x normal (fat finger?)",
    )
    ks.add_trigger(
        lambda s: s.get("api_error_count_1h", 0) > 50,
        "Excessive API errors (>50/hour)",
    )

    return ks


if __name__ == "__main__":
    print("=" * 70)
    print("Kill Switch — Demo")
    print("=" * 70)

    initial_equity = 25000
    ks = standard_trading_kill_switch(initial_equity)

    print(f"\nKill switch initialized với {len(ks.triggers)} triggers:")
    for _, name in ks.triggers:
        print(f"  - {name}")

    # Test scenarios
    test_states = [
        ("Normal operation",
            {"daily_loss_pct": -0.01, "current_equity": 24800,
             "consecutive_losses": 2, "unexpected_position_count": 0,
             "normal_position_size": 0.05, "max_position_size": 0.05,
             "api_error_count_1h": 3}),
        ("Daily loss exceeded",
            {"daily_loss_pct": -0.06, "current_equity": 23500,
             "consecutive_losses": 0, "unexpected_position_count": 0}),
    ]

    for label, state in test_states:
        print(f"\n--- Scenario: {label} ---")
        triggered = ks.check(state)
        if triggered:
            print(f"  Kill switch ACTIVATED")
            print(f"  Status: {ks.status()['is_active']}")
            ks.manual_reset()  # Reset for next test
        else:
            print(f"  Normal operation, no triggers")

    # Test fat finger scenario
    print(f"\n--- Scenario: Fat finger error ---")
    fat_finger_state = {
        "daily_loss_pct": -0.01, "current_equity": 24800,
        "consecutive_losses": 0, "unexpected_position_count": 0,
        "normal_position_size": 0.05, "max_position_size": 0.5,
        "api_error_count_1h": 3,
    }
    ks.check(fat_finger_state)
    print(f"  Status after: {ks.status()}")
