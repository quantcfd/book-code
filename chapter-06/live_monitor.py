"""
QuantCFD Chương 6 — live_monitor.py
Live trading monitoring: tracking error, alerts, stop-trading rules.

Usage:
    from live_monitor import live_alerts, StopTradingRules, live_vs_backtest_tracking

    alerts = live_alerts(live_returns, baseline)
    rules = StopTradingRules(max_dd_threshold=-0.20)
    should_stop, reason = rules.should_stop(live_returns)
"""
import numpy as np
import pandas as pd


def live_vs_backtest_tracking(
    backtest_returns: pd.Series,
    live_returns: pd.Series,
) -> dict:
    """Compare live vs backtest. Idealy correlation > 0.7 sau slippage."""
    aligned = pd.concat([backtest_returns, live_returns], axis=1).dropna()
    aligned.columns = ['backtest', 'live']

    if len(aligned) == 0:
        return {}

    diff = aligned['live'] - aligned['backtest']
    return {
        'n_days_live':           len(aligned),
        'backtest_mean':         aligned['backtest'].mean(),
        'live_mean':             aligned['live'].mean(),
        'mean_diff':             diff.mean(),
        'tracking_error':        diff.std(),
        'correlation':           aligned['backtest'].corr(aligned['live']),
        'live_total_return':     (1 + aligned['live']).prod() - 1,
        'backtest_total_return': (1 + aligned['backtest']).prod() - 1,
        'slippage_drag':         aligned['backtest'].mean() - aligned['live'].mean(),
    }


def live_alerts(
    live_returns: pd.Series,
    backtest_baseline: dict,
    rolling_window: int = 60,
) -> list:
    """
    Generate alerts khi live metrics deviate khỏi expected.
    backtest_baseline: {'sharpe': 1.5, 'max_dd': -0.12}
    """
    alerts = []

    if len(live_returns) < rolling_window:
        return [f"Live data only {len(live_returns)} days — need {rolling_window} for full alerts"]

    # Rolling Sharpe
    recent = live_returns.tail(rolling_window)
    if recent.std() > 0:
        rs = recent.mean() / recent.std() * np.sqrt(252)
        baseline_sharpe = backtest_baseline.get('sharpe', 1.0)
        if rs < baseline_sharpe * 0.3:
            alerts.append(
                f"⚠ Rolling {rolling_window}-day Sharpe {rs:.2f} < 30% of expected {baseline_sharpe:.2f}"
            )

    # Current drawdown
    equity = (1 + live_returns).cumprod()
    current_dd = equity.iloc[-1] / equity.cummax().iloc[-1] - 1
    baseline_dd = backtest_baseline.get('max_dd', -0.20)
    if current_dd < baseline_dd * 1.2:
        alerts.append(
            f"🔴 Current DD {current_dd*100:.1f}% > 120% of expected max {baseline_dd*100:.1f}%"
        )

    # Consecutive losing days
    is_loss = (live_returns.tail(15) < 0).astype(int)
    if is_loss.sum() >= 7:
        groups = (is_loss != is_loss.shift()).cumsum()
        streak = is_loss.groupby(groups).cumsum().iloc[-1]
        if streak >= 7:
            alerts.append(f"⚠ {streak} consecutive losing days — review strategy")

    # Vol regime change
    recent_vol = live_returns.tail(20).std() * np.sqrt(252)
    historical_vol = live_returns.std() * np.sqrt(252)
    if historical_vol > 0 and recent_vol > historical_vol * 1.5:
        alerts.append(
            f"📊 Recent 20-day vol {recent_vol:.1%} > 1.5× historical {historical_vol:.1%}"
        )

    return alerts


class StopTradingRules:
    """
    Define rules để auto-stop strategy khi metrics deteriorate.
    Quan trọng: rules phải define trước live, base trên backtest.
    """

    def __init__(
        self,
        max_dd_threshold: float = -0.20,
        rolling_sharpe_threshold: float = -0.5,
        max_consec_losses: int = 15,
        vol_regime_multiplier: float = 2.5,
    ):
        self.max_dd_threshold = max_dd_threshold
        self.rolling_sharpe_threshold = rolling_sharpe_threshold
        self.max_consec_losses = max_consec_losses
        self.vol_regime_multiplier = vol_regime_multiplier

    def should_stop(self, live_returns: pd.Series) -> tuple:
        """Check tất cả rules. Return (stop_flag, reason)."""
        if len(live_returns) == 0:
            return False, ""

        equity = (1 + live_returns).cumprod()
        current_dd = equity.iloc[-1] / equity.cummax().iloc[-1] - 1
        if current_dd < self.max_dd_threshold:
            return True, f"Max DD breached: {current_dd*100:.1f}% < {self.max_dd_threshold*100:.0f}%"

        if len(live_returns) >= 60:
            recent = live_returns.tail(60)
            if recent.std() > 0:
                rs = recent.mean() / recent.std() * np.sqrt(252)
                if rs < self.rolling_sharpe_threshold:
                    return True, f"Rolling 60d Sharpe breached: {rs:.2f} < {self.rolling_sharpe_threshold:.2f}"

        check_window = min(self.max_consec_losses + 5, len(live_returns))
        is_loss = (live_returns.tail(check_window) < 0).astype(int)
        if is_loss.sum() >= self.max_consec_losses:
            groups = (is_loss != is_loss.shift()).cumsum()
            streak = is_loss.groupby(groups).cumsum().max()
            if streak >= self.max_consec_losses:
                return True, f"{int(streak)} consecutive losses"

        if len(live_returns) >= 40:
            recent_vol = live_returns.tail(20).std()
            historical_vol = live_returns.std()
            if historical_vol > 0 and recent_vol > historical_vol * self.vol_regime_multiplier:
                return True, f"Vol spike: recent {recent_vol*np.sqrt(252):.1%} > {self.vol_regime_multiplier}× historical"

        return False, ""


def daily_summary_report(
    live_returns: pd.Series,
    baseline: dict = None,
) -> str:
    """Text summary cho daily monitoring (dùng để send Telegram)."""
    if len(live_returns) == 0:
        return "No live data"

    if baseline is None:
        baseline = {'sharpe': 1.5, 'max_dd': -0.15}

    equity = (1 + live_returns).cumprod()
    current_eq = equity.iloc[-1]
    total_ret = current_eq - 1
    daily_ret = live_returns.iloc[-1]

    current_dd = equity.iloc[-1] / equity.cummax().iloc[-1] - 1

    if len(live_returns) >= 60 and live_returns.tail(60).std() > 0:
        rs60 = live_returns.tail(60).mean() / live_returns.tail(60).std() * np.sqrt(252)
        rs_str = f"{rs60:.2f}"
    else:
        rs_str = "N/A"

    alerts = live_alerts(live_returns, baseline)
    alert_block = "\n".join(alerts) if alerts else "✓ No alerts"

    return f"""
QuantCFD Daily Summary — {live_returns.index[-1].date() if isinstance(live_returns.index, pd.DatetimeIndex) else 'N/A'}
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Equity: {current_eq:.4f}× initial
Today:  {daily_ret*100:+.2f}%
Total:  {total_ret*100:+.1f}%
Current DD: {current_dd*100:.2f}%
Rolling 60d Sharpe: {rs_str}

Alerts:
{alert_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━
""".strip()


if __name__ == '__main__':
    np.random.seed(13)
    dates = pd.date_range('2024-01-01', periods=180, freq='D')

    backtest = pd.Series(np.random.normal(0.0008, 0.012, 180), index=dates)
    live = pd.Series(np.random.normal(0.0003, 0.015, 180), index=dates)

    print("=== Tracking Error ===")
    track = live_vs_backtest_tracking(backtest, live)
    for k, v in track.items():
        print(f"  {k:25s} {v:+.4f}" if isinstance(v, float) else f"  {k:25s} {v}")

    print("\n=== Alerts ===")
    alerts = live_alerts(live, {'sharpe': 1.5, 'max_dd': -0.15})
    for a in alerts:
        print(f"  {a}")

    print("\n=== Stop Trading Check ===")
    rules = StopTradingRules(max_dd_threshold=-0.10, rolling_sharpe_threshold=-0.3)
    should, reason = rules.should_stop(live)
    print(f"  Should stop: {should}, reason: {reason}")

    print("\n=== Daily Summary ===")
    print(daily_summary_report(live))
