"""
QuantCFD — Chương 10.13.5
Risk Dashboard

Production-grade risk monitoring class.
Computes per-strategy + portfolio-level metrics.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime


class RiskDashboard:
    """
    Multi-strategy risk dashboard.

    Computes:
    - Per-strategy: Sharpe, Sortino, CAGR, Max DD, VaR, CVaR, Kelly, Ulcer
    - Portfolio: Sharpe, CAGR, Max DD, current DD, VaR, Calmar
    - Correlation matrix
    - Active alerts
    """

    def __init__(
        self,
        returns_dict: dict,
        weights: dict = None,
        equity_history: pd.Series = None,
        initial_equity: float = 10000,
    ):
        self.returns = pd.DataFrame(returns_dict).fillna(0)
        if weights is None:
            n = len(self.returns.columns)
            weights = {name: 1 / n for name in self.returns.columns}
        self.weights = weights
        self.initial_equity = initial_equity

        # Build equity history if not provided
        if equity_history is None:
            portfolio_ret = pd.Series(0.0, index=self.returns.index)
            for name, w in weights.items():
                if name in self.returns.columns:
                    portfolio_ret += self.returns[name] * w
            self.equity_history = (1 + portfolio_ret).cumprod() * initial_equity
        else:
            self.equity_history = equity_history

        self.peak_equity = self.equity_history.max()

    def _sortino(self, returns: pd.Series) -> float:
        """Sortino ratio = mean / downside_std × sqrt(252)."""
        downside = returns[returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return 0
        return (returns.mean() / downside.std()) * np.sqrt(252)

    def _ulcer_index(self, equity: pd.Series) -> float:
        """Ulcer Index = sqrt(mean(DD²))."""
        dd = (equity / equity.cummax() - 1)
        return float(np.sqrt((dd ** 2).mean()))

    def per_strategy_metrics(self) -> pd.DataFrame:
        """Compute per-strategy metrics."""
        results = {}
        for name in self.returns.columns:
            r = self.returns[name].dropna()
            if len(r) < 30 or r.std() == 0:
                continue

            eq = (1 + r).cumprod()
            kelly_full = r.mean() / r.var() if r.var() > 0 else 0

            results[name] = {
                "sharpe": (r.mean() / r.std()) * np.sqrt(252),
                "sortino": self._sortino(r),
                "cagr": (1 + r.mean()) ** 252 - 1,
                "max_dd": (eq / eq.cummax() - 1).min(),
                "var_95": -r.quantile(0.05),
                "cvar_95": -r[r < r.quantile(0.05)].mean()
                            if (r < r.quantile(0.05)).any() else 0,
                "win_rate": (r > 0).mean(),
                "kelly_full": kelly_full,
                "ulcer_idx": self._ulcer_index(eq),
                "weight": self.weights.get(name, 0),
            }
        return pd.DataFrame(results).T

    def portfolio_metrics(self) -> dict:
        """Compute portfolio-level metrics."""
        portfolio_ret = pd.Series(0.0, index=self.returns.index)
        for name, w in self.weights.items():
            if name in self.returns.columns:
                portfolio_ret += self.returns[name] * w

        clean = portfolio_ret.dropna()
        if len(clean) < 30 or clean.std() == 0:
            return {}

        eq = (1 + clean).cumprod()
        max_dd = (eq / eq.cummax() - 1).min()
        cagr = (1 + clean.mean()) ** 252 - 1

        return {
            "sharpe": (clean.mean() / clean.std()) * np.sqrt(252),
            "cagr": cagr,
            "max_dd": max_dd,
            "current_dd": (
                (self.equity_history.iloc[-1] - self.peak_equity)
                / self.peak_equity
            ),
            "var_95": -clean.quantile(0.05),
            "cvar_95": -clean[clean < clean.quantile(0.05)].mean()
                        if (clean < clean.quantile(0.05)).any() else 0,
            "calmar": cagr / abs(max_dd) if max_dd != 0 else 0,
            "current_equity": float(self.equity_history.iloc[-1]),
            "peak_equity": float(self.peak_equity),
        }

    def correlation_matrix(self, lookback_days: int = 63) -> pd.DataFrame:
        """Rolling correlation matrix (last N days)."""
        recent = self.returns.tail(lookback_days)
        return recent.corr()

    def loss_limit_status(
        self,
        daily_pnl: float,
        weekly_pnl: float,
        monthly_pnl: float,
        daily_limit: float = -0.03,
        weekly_limit: float = -0.07,
        monthly_limit: float = -0.15,
    ) -> dict:
        """Check current status vs pre-committed limits."""
        return {
            "daily": {
                "actual": daily_pnl, "limit": daily_limit,
                "ok": daily_pnl > daily_limit,
            },
            "weekly": {
                "actual": weekly_pnl, "limit": weekly_limit,
                "ok": weekly_pnl > weekly_limit,
            },
            "monthly": {
                "actual": monthly_pnl, "limit": monthly_limit,
                "ok": monthly_pnl > monthly_limit,
            },
        }

    def alerts(self) -> list:
        """Generate active alerts."""
        alerts = []

        # Correlation spike
        corr = self.correlation_matrix()
        # Get max off-diagonal correlation
        n = len(corr.columns)
        if n > 1:
            corr_values = corr.values.copy()
            np.fill_diagonal(corr_values, 0)
            max_corr = float(np.abs(corr_values).max())
            if max_corr > 0.6:
                alerts.append(f"CORRELATION SPIKE: max pairwise = {max_corr:.2f}")

        # Portfolio DD
        port = self.portfolio_metrics()
        if port and abs(port["current_dd"]) > 0.15:
            alerts.append(
                f"PORTFOLIO DD WARNING: {port['current_dd']*100:.1f}%"
            )
        if port and abs(port["current_dd"]) > 0.20:
            alerts.append("HALT TRIGGERED: portfolio DD > 20%")

        # Per-strategy DD
        per_strat = self.per_strategy_metrics()
        for name, row in per_strat.iterrows():
            if row["max_dd"] < -0.15:
                alerts.append(
                    f"{name} MAX DD WARNING: {row['max_dd']*100:.1f}%"
                )

        return alerts

    def report(self):
        """Print full dashboard report."""
        print("=" * 70)
        print(
            f"RISK DASHBOARD — "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print("=" * 70)

        port = self.portfolio_metrics()
        print("\nPORTFOLIO:")
        for k, v in port.items():
            if isinstance(v, float):
                if abs(v) < 1:
                    print(f"  {k:<20}: {v:.4f}")
                else:
                    print(f"  {k:<20}: ${v:,.2f}")

        print("\nPER-STRATEGY:")
        per_strat = self.per_strategy_metrics()
        if len(per_strat) > 0:
            print(per_strat.round(3).to_string())

        print("\nCORRELATION MATRIX:")
        print(self.correlation_matrix().round(2))

        print("\nALERTS:")
        alerts = self.alerts()
        if alerts:
            for alert in alerts:
                print(f"  ⚠ {alert}")
        else:
            print("  None — all metrics within acceptable range")


if __name__ == "__main__":
    print("=" * 70)
    print("Risk Dashboard — Demo")
    print("=" * 70)

    # Generate synthetic 3-strategy returns
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    returns_dict = {
        "trend":  pd.Series(np.random.randn(n) * 0.010 + 0.0003, index=dates),
        "mr":     pd.Series(np.random.randn(n) * 0.007 + 0.0002, index=dates),
        "vol_bo": pd.Series(np.random.randn(n) * 0.015 + 0.0005, index=dates),
    }

    weights = {"trend": 0.45, "mr": 0.30, "vol_bo": 0.25}

    dashboard = RiskDashboard(
        returns_dict, weights=weights, initial_equity=25000,
    )
    dashboard.report()

    # Loss limit status example
    print("\n--- Loss limit status example ---")
    status = dashboard.loss_limit_status(
        daily_pnl=-0.025, weekly_pnl=-0.04, monthly_pnl=-0.12,
    )
    for level, info in status.items():
        symbol = "✓" if info["ok"] else "✗"
        print(f"  {symbol} {level:<10}: actual {info['actual']*100:+.2f}% "
              f"vs limit {info['limit']*100}%")
