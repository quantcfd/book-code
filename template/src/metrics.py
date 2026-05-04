"""
src/metrics.py — Performance metrics cho backtest.

Re-export một số function từ chapter-02 để các strategy file dùng được mà
không cần import chéo qua đường dẫn lằng nhằng.

Đầy đủ học ở Chương 6.
"""
import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio."""
    r = returns.dropna()
    if len(r) == 0 or r.std() == 0:
        return 0.0
    return r.mean() / r.std() * np.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Sortino ratio — chỉ tính downside std (penalty volatility tốt < volatility xấu)."""
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    downside = r[r < 0]
    downside_std = downside.std() if len(downside) > 0 else 0.0
    if downside_std == 0:
        return 0.0
    return r.mean() / downside_std * np.sqrt(periods_per_year)


def max_drawdown(returns: pd.Series) -> float:
    """Max drawdown từ một series returns."""
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    equity = (1 + r).cumprod()
    return (equity / equity.cummax() - 1).min()


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calmar = annual return / |max drawdown|."""
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    annual_ret = r.mean() * periods_per_year
    mdd = abs(max_drawdown(r))
    return annual_ret / mdd if mdd > 0 else 0.0


def expectancy(trades_pnl: pd.Series) -> float:
    """
    Expectancy per trade = P(win)*avg_win - P(loss)*avg_loss.

    Reference: chapter-02/expectancy_calculator.py for full report.
    """
    t = trades_pnl.dropna()
    n = len(t)
    if n == 0:
        return 0.0
    wins = t[t > 0]
    losses = t[t < 0]
    p_win = len(wins) / n
    p_loss = len(losses) / n
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
    return p_win * avg_win - p_loss * avg_loss


def summary_report(returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Tổng hợp các metrics quan trọng nhất cho strategy report."""
    r = returns.dropna()
    if len(r) == 0:
        return {}
    return {
        "n_periods": len(r),
        "total_return": (1 + r).prod() - 1,
        "annual_return": r.mean() * periods_per_year,
        "annual_vol": r.std() * np.sqrt(periods_per_year),
        "sharpe": sharpe_ratio(r, periods_per_year),
        "sortino": sortino_ratio(r, periods_per_year),
        "max_drawdown": max_drawdown(r),
        "calmar": calmar_ratio(r, periods_per_year),
        "win_rate": (r > 0).sum() / len(r),
    }
