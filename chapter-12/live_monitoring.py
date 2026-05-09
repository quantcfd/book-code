"""
Live monitoring infrastructure.

Real-time dashboard data + Discord/Telegram webhook alert system
for production trading.

QuantCFD Chapter 12 - Capstone deployment
"""
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional


class AlertPriority(Enum):
    CRITICAL = "critical"   # Immediate action
    HIGH = "high"           # Response within 1 hour
    MEDIUM = "medium"       # Response within 24 hours
    LOW = "low"             # Informational only


class AlertCategory(Enum):
    RISK_BREACH = "risk_breach"
    SYSTEM_ERROR = "system_error"
    TRADE_EVENT = "trade_event"
    PERFORMANCE = "performance"
    BIAS_DETECTED = "bias_detected"
    DATA_ISSUE = "data_issue"
    BROKER_ISSUE = "broker_issue"
    INFORMATIONAL = "informational"


@dataclass
class Alert:
    """Single alert entry."""
    timestamp: datetime
    priority: AlertPriority
    category: AlertCategory
    title: str
    message: str
    metadata: Dict = field(default_factory=dict)
    is_acknowledged: bool = False

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority.value,
            'category': self.category.value,
            'title': self.title,
            'message': self.message,
            'metadata': self.metadata,
            'is_acknowledged': self.is_acknowledged,
        }


@dataclass
class DashboardSnapshot:
    """Real-time dashboard data snapshot."""
    timestamp: datetime
    account_equity: float
    cash_balance: float
    open_positions: int
    today_pnl: float
    today_pnl_pct: float
    week_pnl: float
    month_pnl: float
    current_drawdown_pct: float
    var_95: float
    rule_adherence_today: float
    open_risk: float
    system_health: str  # 'healthy', 'warning', 'critical'
    data_feed_status: str
    broker_connection_status: str
    active_alerts_count: int


class LiveMonitor:
    """
    Real-time monitoring system.

    Maintains dashboard snapshot + alerts queue.
    Sends Discord/Telegram notifications based on priority.
    """

    def __init__(
        self,
        discord_webhook_url: Optional[str] = None,
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
    ):
        self.discord_webhook_url = discord_webhook_url
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id

        self.alerts: List[Alert] = []
        self.dashboard_history: List[DashboardSnapshot] = []
        self.last_snapshot: Optional[DashboardSnapshot] = None

        # Alert thresholds
        self.daily_loss_warning_pct = -0.02   # -2% warning
        self.daily_loss_critical_pct = -0.03  # -3% critical (kill switch)
        self.weekly_loss_warning_pct = -0.04
        self.weekly_loss_critical_pct = -0.06
        self.monthly_loss_critical_pct = -0.10
        self.dd_warning_pct = -0.10           # -10% DD warning
        self.dd_critical_pct = -0.15          # -15% DD critical

    def update_dashboard(
        self,
        account_equity: float,
        cash_balance: float,
        open_positions: int,
        today_pnl: float,
        week_pnl: float,
        month_pnl: float,
        max_equity_30d: float = None,
        var_95: float = 0,
        rule_adherence_today: float = 1.0,
        open_risk: float = 0,
        data_feed_ok: bool = True,
        broker_ok: bool = True,
    ) -> DashboardSnapshot:
        """Update dashboard snapshot và check for alerts."""

        # Compute drawdown
        if max_equity_30d is None:
            max_equity_30d = account_equity
        current_dd_pct = ((account_equity - max_equity_30d) / max_equity_30d
                          if max_equity_30d > 0 else 0)

        today_pnl_pct = today_pnl / account_equity if account_equity > 0 else 0

        # Determine system health
        if not data_feed_ok or not broker_ok:
            system_health = 'critical'
        elif current_dd_pct < self.dd_critical_pct:
            system_health = 'critical'
        elif current_dd_pct < self.dd_warning_pct:
            system_health = 'warning'
        elif today_pnl_pct < self.daily_loss_critical_pct:
            system_health = 'critical'
        elif today_pnl_pct < self.daily_loss_warning_pct:
            system_health = 'warning'
        else:
            system_health = 'healthy'

        snapshot = DashboardSnapshot(
            timestamp=datetime.now(),
            account_equity=account_equity,
            cash_balance=cash_balance,
            open_positions=open_positions,
            today_pnl=today_pnl,
            today_pnl_pct=today_pnl_pct * 100,
            week_pnl=week_pnl,
            month_pnl=month_pnl,
            current_drawdown_pct=current_dd_pct * 100,
            var_95=var_95,
            rule_adherence_today=rule_adherence_today,
            open_risk=open_risk,
            system_health=system_health,
            data_feed_status='OK' if data_feed_ok else 'DOWN',
            broker_connection_status='OK' if broker_ok else 'DOWN',
            active_alerts_count=sum(1 for a in self.alerts if not a.is_acknowledged),
        )

        # Auto-generate alerts based on thresholds
        self._check_threshold_alerts(snapshot)

        # Persist
        self.dashboard_history.append(snapshot)
        self.last_snapshot = snapshot

        # Keep only last 1000 snapshots
        if len(self.dashboard_history) > 1000:
            self.dashboard_history = self.dashboard_history[-1000:]

        return snapshot

    def _check_threshold_alerts(self, snapshot: DashboardSnapshot):
        """Auto-generate alerts based on threshold breaches."""

        # Daily loss
        daily_pct = snapshot.today_pnl_pct / 100
        if daily_pct < self.daily_loss_critical_pct:
            self.create_alert(
                priority=AlertPriority.CRITICAL,
                category=AlertCategory.RISK_BREACH,
                title="DAILY LOSS LIMIT HIT",
                message=f"Today P&L {daily_pct*100:.2f}% breached limit "
                        f"{self.daily_loss_critical_pct*100:.0f}%. Kill switch activated.",
            )
        elif daily_pct < self.daily_loss_warning_pct:
            self.create_alert(
                priority=AlertPriority.HIGH,
                category=AlertCategory.RISK_BREACH,
                title="Daily loss approaching limit",
                message=f"Today P&L {daily_pct*100:.2f}%, limit "
                        f"{self.daily_loss_critical_pct*100:.0f}%",
            )

        # Drawdown
        dd_pct = snapshot.current_drawdown_pct / 100
        if dd_pct < self.dd_critical_pct:
            self.create_alert(
                priority=AlertPriority.CRITICAL,
                category=AlertCategory.RISK_BREACH,
                title="CRITICAL DRAWDOWN",
                message=f"Current DD {dd_pct*100:.1f}% exceeds critical threshold",
            )

        # System health
        if not snapshot.data_feed_status == 'OK':
            self.create_alert(
                priority=AlertPriority.CRITICAL,
                category=AlertCategory.DATA_ISSUE,
                title="DATA FEED DOWN",
                message="Real-time price feed disconnected",
            )

        if not snapshot.broker_connection_status == 'OK':
            self.create_alert(
                priority=AlertPriority.CRITICAL,
                category=AlertCategory.BROKER_ISSUE,
                title="BROKER CONNECTION LOST",
                message="Cannot reach broker API",
            )

    def create_alert(
        self,
        priority: AlertPriority,
        category: AlertCategory,
        title: str,
        message: str,
        metadata: Dict = None,
    ) -> Alert:
        """Create new alert."""
        alert = Alert(
            timestamp=datetime.now(),
            priority=priority,
            category=category,
            title=title,
            message=message,
            metadata=metadata or {},
        )
        self.alerts.append(alert)

        # Send notification (in production, actually send to Discord/Telegram)
        self._send_notification(alert)

        return alert

    def _send_notification(self, alert: Alert):
        """Send alert via Discord/Telegram (mock implementation)."""
        # In production: actually send via webhook/API
        # Here just print to demonstrate
        emoji_map = {
            AlertPriority.CRITICAL: '🔴',
            AlertPriority.HIGH: '🟠',
            AlertPriority.MEDIUM: '🟡',
            AlertPriority.LOW: '🟢',
        }
        emoji = emoji_map.get(alert.priority, '⚪')
        # Real implementation would POST to webhook URL
        # print(f"  [NOTIFICATION] {emoji} {alert.priority.value.upper()}: {alert.title}")

    def acknowledge_alert(self, alert_index: int) -> bool:
        """Mark alert as acknowledged."""
        if 0 <= alert_index < len(self.alerts):
            self.alerts[alert_index].is_acknowledged = True
            return True
        return False

    def get_unacknowledged_alerts(self) -> List[Alert]:
        """Get all alerts not yet acknowledged."""
        return [a for a in self.alerts if not a.is_acknowledged]

    def get_critical_alerts(self) -> List[Alert]:
        """Get all critical priority alerts."""
        return [a for a in self.alerts if a.priority == AlertPriority.CRITICAL]

    def generate_dashboard_text(self) -> str:
        """Generate text-based dashboard."""
        if self.last_snapshot is None:
            return "No data."

        s = self.last_snapshot

        lines = []
        lines.append("=" * 60)
        lines.append(f"LIVE DASHBOARD — {s.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)

        # Health indicator
        health_emoji = {'healthy': '✅', 'warning': '⚠️', 'critical': '🔴'}
        lines.append(f"System Health:        {health_emoji.get(s.system_health, '?')} "
                     f"{s.system_health.upper()}")
        lines.append(f"Data Feed:            {s.data_feed_status}")
        lines.append(f"Broker Connection:    {s.broker_connection_status}")
        lines.append("")

        # Account
        lines.append("ACCOUNT:")
        lines.append(f"  Equity:             ${s.account_equity:>12,.2f}")
        lines.append(f"  Cash balance:       ${s.cash_balance:>12,.2f}")
        lines.append(f"  Open positions:     {s.open_positions}")
        lines.append(f"  Open risk:          ${s.open_risk:>12,.2f}")
        lines.append("")

        # P&L
        lines.append("P&L:")
        lines.append(f"  Today:              ${s.today_pnl:>+10,.2f} ({s.today_pnl_pct:+.2f}%)")
        lines.append(f"  Week:               ${s.week_pnl:>+10,.2f}")
        lines.append(f"  Month:              ${s.month_pnl:>+10,.2f}")
        lines.append(f"  Current DD:         {s.current_drawdown_pct:.2f}%")
        lines.append("")

        # Risk
        lines.append("RISK METRICS:")
        lines.append(f"  VaR 95%:            ${s.var_95:>12,.2f}")
        lines.append(f"  Rule adherence:     {s.rule_adherence_today*100:.0f}%")
        lines.append("")

        # Alerts
        unack = self.get_unacknowledged_alerts()
        if unack:
            lines.append(f"ACTIVE ALERTS ({len(unack)}):")
            for a in unack[-5:]:  # Show last 5
                emoji = {'critical': '🔴', 'high': '🟠',
                         'medium': '🟡', 'low': '🟢'}.get(a.priority.value, '⚪')
                lines.append(f"  {emoji} [{a.priority.value.upper()}] {a.title}")
        else:
            lines.append("ALERTS: All clear ✓")

        lines.append("=" * 60)
        return "\n".join(lines)


def demo():
    """Demo live monitoring system."""
    print("=" * 60)
    print("DEMO: Live monitoring infrastructure")
    print("=" * 60)

    monitor = LiveMonitor()

    print("\n1. Healthy state:")
    snap = monitor.update_dashboard(
        account_equity=30000,
        cash_balance=12000,
        open_positions=5,
        today_pnl=120,
        week_pnl=380,
        month_pnl=1450,
        max_equity_30d=30500,
        var_95=600,
        rule_adherence_today=0.95,
        open_risk=180,
    )
    print(monitor.generate_dashboard_text())

    print("\n\n2. Daily loss approaching limit:")
    snap = monitor.update_dashboard(
        account_equity=29500,
        cash_balance=12000,
        open_positions=3,
        today_pnl=-650,
        week_pnl=-280,
        month_pnl=950,
        max_equity_30d=30500,
        var_95=600,
        rule_adherence_today=0.92,
        open_risk=180,
    )
    print(monitor.generate_dashboard_text())

    print("\n\n3. Critical: daily loss limit breached:")
    snap = monitor.update_dashboard(
        account_equity=29000,
        cash_balance=12000,
        open_positions=0,  # closed by kill switch
        today_pnl=-1000,
        week_pnl=-630,
        month_pnl=600,
        max_equity_30d=30500,
        var_95=600,
        rule_adherence_today=0.88,
        open_risk=0,
    )
    print(monitor.generate_dashboard_text())

    print("\n\n4. Critical: data feed down:")
    snap = monitor.update_dashboard(
        account_equity=29000,
        cash_balance=12000,
        open_positions=0,
        today_pnl=-1000,
        week_pnl=-630,
        month_pnl=600,
        max_equity_30d=30500,
        var_95=600,
        rule_adherence_today=0.88,
        open_risk=0,
        data_feed_ok=False,
    )
    print(monitor.generate_dashboard_text())

    # Manual alert
    print("\n\n5. Manual trade event alert:")
    monitor.create_alert(
        priority=AlertPriority.MEDIUM,
        category=AlertCategory.TRADE_EVENT,
        title="Position closed at target",
        message="XAUUSD long closed at $2050, +$120 profit",
        metadata={'symbol': 'XAUUSD', 'pnl': 120, 'r_multiple': 1.5},
    )

    # Summary
    print("\n6. Alert summary:")
    print(f"   Total alerts:    {len(monitor.alerts)}")
    print(f"   Unacknowledged:  {len(monitor.get_unacknowledged_alerts())}")
    print(f"   Critical:        {len(monitor.get_critical_alerts())}")

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
