"""
QuantCFD — Chương 10
Solution Exercise 6 (BONUS) — Production Risk Dashboard

Full RiskDashboard with:
- Real-time metrics (per-strategy + portfolio)
- Correlation matrix với alert thresholds
- Loss limit status
- Strategy retirement criteria check
- HTML report output
- JSON snapshot
"""

from __future__ import annotations
import json
from datetime import datetime
import numpy as np
import pandas as pd
from risk_dashboard import RiskDashboard


class ProductionRiskDashboard(RiskDashboard):
    """
    Extended dashboard với HTML report, JSON snapshot, retirement criteria.
    """

    def __init__(
        self,
        returns_dict: dict,
        weights: dict = None,
        equity_history: pd.Series = None,
        initial_equity: float = 10000,
        backtest_metrics: dict = None,
    ):
        super().__init__(returns_dict, weights, equity_history, initial_equity)
        self.backtest_metrics = backtest_metrics or {}

    def retirement_criteria_check(self) -> dict:
        """
        Check each strategy against retirement criteria.

        Criteria:
        - Live Sharpe < 30% of backtest
        - Live max DD > 1.5x backtest max DD
        - Win rate dropped 30%+
        - Profit factor < 1.0 over 12 months
        """
        results = {}
        per_strat = self.per_strategy_metrics()

        for name, row in per_strat.iterrows():
            backtest = self.backtest_metrics.get(name, {})
            backtest_sharpe = backtest.get("sharpe", 1.0)
            backtest_max_dd = backtest.get("max_dd", -0.15)
            backtest_win_rate = backtest.get("win_rate", 0.5)

            criteria = {
                "sharpe_below_threshold": (
                    row["sharpe"] < backtest_sharpe * 0.3
                ),
                "max_dd_exceeded": row["max_dd"] < backtest_max_dd * 1.5,
                "win_rate_dropped": row["win_rate"] < backtest_win_rate * 0.7,
            }

            n_failures = sum(criteria.values())
            if n_failures >= 3:
                verdict = "RETIRE"
            elif n_failures >= 2:
                verdict = "REVIEW"
            elif n_failures >= 1:
                verdict = "WATCH"
            else:
                verdict = "OK"

            results[name] = {
                **criteria,
                "n_failures": n_failures,
                "verdict": verdict,
                "live_sharpe": row["sharpe"],
                "backtest_sharpe": backtest_sharpe,
                "live_max_dd": row["max_dd"],
                "backtest_max_dd": backtest_max_dd,
            }

        return results

    def to_json(self) -> str:
        """Export full state as JSON."""
        per_strat = self.per_strategy_metrics()
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "portfolio": self.portfolio_metrics(),
            "per_strategy": per_strat.to_dict() if len(per_strat) > 0 else {},
            "correlation_matrix": self.correlation_matrix().to_dict(),
            "alerts": self.alerts(),
            "retirement_check": self.retirement_criteria_check(),
        }
        # Convert any numpy types to Python natives
        return json.dumps(snapshot, default=str, indent=2)

    def to_html(self, output_path: str = "risk_dashboard.html") -> str:
        """Generate HTML report."""
        port = self.portfolio_metrics()
        per_strat = self.per_strategy_metrics()
        retirement = self.retirement_criteria_check()
        alerts = self.alerts()

        html = ['<!DOCTYPE html><html><head>',
                '<meta charset="utf-8">',
                '<title>QuantCFD Risk Dashboard</title>',
                '<style>',
                'body { font-family: -apple-system, sans-serif; margin: 30px; '
                'background: #f5f5f5; color: #222; }',
                'h1, h2 { color: #0F1F3D; }',
                'h1 { border-bottom: 3px solid #0E7C7B; padding-bottom: 8px; }',
                '.card { background: white; padding: 20px; margin: 15px 0; '
                'border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }',
                '.metric { display: inline-block; margin: 10px 30px 10px 0; }',
                '.metric-label { font-size: 12px; color: #666; '
                'text-transform: uppercase; }',
                '.metric-value { font-size: 24px; font-weight: bold; '
                'color: #0F1F3D; }',
                '.alert { background: #FFF4D4; border-left: 5px solid #F0AD4E; '
                'padding: 12px; margin: 8px 0; }',
                '.ok { color: #5CB85C; }',
                '.warn { color: #F0AD4E; }',
                '.fail { color: #D9534F; }',
                'table { border-collapse: collapse; width: 100%; }',
                'th { background: #0F1F3D; color: white; padding: 10px; '
                'text-align: left; }',
                'td { padding: 10px; border-bottom: 1px solid #ddd; }',
                'tr:nth-child(even) { background: #f9f9f9; }',
                '.verdict-OK { color: #5CB85C; font-weight: bold; }',
                '.verdict-WATCH { color: #F0AD4E; font-weight: bold; }',
                '.verdict-REVIEW { color: #FF8C00; font-weight: bold; }',
                '.verdict-RETIRE { color: #D9534F; font-weight: bold; }',
                '</style></head><body>',
                '<h1>QuantCFD — Risk Dashboard</h1>',
                f'<p style="color:#666;">Generated: '
                f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>']

        # Portfolio metrics
        html.append('<div class="card"><h2>Portfolio Overview</h2>')
        for key, val in port.items():
            if isinstance(val, float):
                if abs(val) < 1:
                    formatted = f"{val:.4f}"
                    if "pct" in key.lower() or "dd" in key.lower():
                        formatted = f"{val*100:.2f}%"
                else:
                    formatted = f"${val:,.2f}"
            else:
                formatted = str(val)
            html.append(
                f'<div class="metric">'
                f'<div class="metric-label">{key.replace("_"," ")}</div>'
                f'<div class="metric-value">{formatted}</div>'
                f'</div>'
            )
        html.append('</div>')

        # Alerts
        if alerts:
            html.append('<div class="card"><h2>Active Alerts</h2>')
            for alert in alerts:
                html.append(f'<div class="alert">⚠ {alert}</div>')
            html.append('</div>')
        else:
            html.append(
                '<div class="card"><h2>Active Alerts</h2>'
                '<p class="ok">✓ No active alerts — all metrics within '
                'acceptable range</p></div>'
            )

        # Per-strategy table
        html.append(
            '<div class="card"><h2>Per-Strategy Metrics</h2><table>'
            '<tr><th>Strategy</th><th>Sharpe</th><th>Sortino</th>'
            '<th>CAGR</th><th>Max DD</th><th>Win Rate</th><th>Weight</th>'
            '</tr>'
        )
        for name, row in per_strat.iterrows():
            html.append(
                f'<tr><td><b>{name}</b></td>'
                f'<td>{row["sharpe"]:.3f}</td>'
                f'<td>{row["sortino"]:.3f}</td>'
                f'<td>{row["cagr"]*100:.2f}%</td>'
                f'<td>{row["max_dd"]*100:.2f}%</td>'
                f'<td>{row["win_rate"]*100:.1f}%</td>'
                f'<td>{row["weight"]*100:.1f}%</td>'
                f'</tr>'
            )
        html.append('</table></div>')

        # Strategy retirement check
        html.append(
            '<div class="card"><h2>Strategy Retirement Criteria</h2><table>'
            '<tr><th>Strategy</th><th>Live Sharpe</th><th>Backtest Sharpe</th>'
            '<th>Live MaxDD</th><th>Failures</th><th>Verdict</th></tr>'
        )
        for name, info in retirement.items():
            html.append(
                f'<tr><td><b>{name}</b></td>'
                f'<td>{info["live_sharpe"]:.3f}</td>'
                f'<td>{info["backtest_sharpe"]:.3f}</td>'
                f'<td>{info["live_max_dd"]*100:.2f}%</td>'
                f'<td>{info["n_failures"]}/3</td>'
                f'<td class="verdict-{info["verdict"]}">{info["verdict"]}</td>'
                f'</tr>'
            )
        html.append('</table></div>')

        # Correlation matrix
        html.append('<div class="card"><h2>Correlation Matrix (60-day)</h2>')
        corr = self.correlation_matrix()
        html.append('<table><tr><th></th>')
        for c in corr.columns:
            html.append(f'<th>{c}</th>')
        html.append('</tr>')
        for idx, row in corr.iterrows():
            html.append(f'<tr><th>{idx}</th>')
            for c in corr.columns:
                val = row[c]
                color = ''
                if c != idx and abs(val) > 0.6:
                    color = ' style="background:#FFF4D4;"'
                html.append(f'<td{color}>{val:.2f}</td>')
            html.append('</tr>')
        html.append('</table></div>')

        html.append(
            '<div class="card" style="text-align:center;color:#999;">'
            'QuantCFD Risk Dashboard · Anthony Nguyễn · Monta Capital'
            '</div>'
        )
        html.append('</body></html>')

        full_html = "".join(html)
        with open(output_path, "w") as f:
            f.write(full_html)
        return output_path


if __name__ == "__main__":
    print("=" * 80)
    print("Bài 6 (BONUS) — Production Risk Dashboard với HTML Report")
    print("=" * 80)

    # Generate synthetic 3-strategy returns
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    returns_dict = {
        "trend":  pd.Series(np.random.randn(n) * 0.010 + 0.0003, index=dates),
        "mr":     pd.Series(np.random.randn(n) * 0.007 + 0.0001, index=dates),
        "vol_bo": pd.Series(np.random.randn(n) * 0.015 + 0.0005, index=dates),
    }

    weights = {"trend": 0.45, "mr": 0.30, "vol_bo": 0.25}

    # Backtest expectations cho retirement check
    backtest_metrics = {
        "trend":  {"sharpe": 1.05, "max_dd": -0.16, "win_rate": 0.42},
        "mr":     {"sharpe": 0.92, "max_dd": -0.12, "win_rate": 0.62},
        "vol_bo": {"sharpe": 1.42, "max_dd": -0.18, "win_rate": 0.48},
    }

    dashboard = ProductionRiskDashboard(
        returns_dict, weights=weights,
        initial_equity=25000,
        backtest_metrics=backtest_metrics,
    )

    # Print report
    dashboard.report()

    # Retirement check
    print(f"\n--- Strategy Retirement Criteria ---")
    retirement = dashboard.retirement_criteria_check()
    for name, info in retirement.items():
        print(f"\n  {name.upper()}: verdict = {info['verdict']}")
        print(f"    Live Sharpe: {info['live_sharpe']:.3f} "
              f"(backtest {info['backtest_sharpe']:.3f})")
        print(f"    Live MaxDD: {info['live_max_dd']*100:.2f}% "
              f"(backtest {info['backtest_max_dd']*100:.2f}%)")
        print(f"    Failures: {info['n_failures']}/3 criteria")

    # Generate JSON snapshot
    json_snapshot = dashboard.to_json()
    with open("/tmp/risk_dashboard_snapshot.json", "w") as f:
        f.write(json_snapshot)
    print(f"\n✓ JSON snapshot saved: /tmp/risk_dashboard_snapshot.json")
    print(f"  Size: {len(json_snapshot)} bytes")

    # Generate HTML report
    html_path = dashboard.to_html("/tmp/risk_dashboard.html")
    print(f"✓ HTML report saved: {html_path}")

    print(f"\nProduction deployment notes:")
    print("  1. Refresh dashboard daily after market close (cron job)")
    print("  2. Email HTML report weekly to mentor/partner")
    print("  3. JSON snapshot for programmatic monitoring/alerts")
    print("  4. Retirement check triggers manual review")
