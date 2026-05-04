"""
src/risk.py — Risk management + position sizing.

Sẽ implement đầy đủ ở Chương 10:
    - Fixed fractional + 2% rule
    - Kelly Criterion (full + half-Kelly)
    - Volatility targeting
    - Portfolio heat management
    - Drawdown-based kill-switch
    - Stress test "all correlations → +1"
"""
import numpy as np


def fixed_fractional_size(
    account_equity: float,
    risk_per_trade: float = 0.02,
    stop_distance: float = 0.01,
) -> float:
    """
    Fixed fractional position sizing — quy tắc 2%.

    Args:
        account_equity: Tổng equity hiện tại (USD).
        risk_per_trade: % equity sẵn sàng mất mỗi lệnh (default 2%).
        stop_distance: % distance từ entry tới stop loss.

    Returns:
        Notional position size (USD).
    """
    risk_amount = account_equity * risk_per_trade
    return risk_amount / stop_distance


def kelly_fraction(p_win: float, payoff_ratio: float) -> float:
    """
    Kelly Criterion full fraction.

    Args:
        p_win: Probability of winning trade.
        payoff_ratio: avg_win / avg_loss.

    Returns:
        Kelly fraction (0 to 1). DÙNG HALF-KELLY trong thực tế.
    """
    p_loss = 1 - p_win
    return p_win - (p_loss / payoff_ratio) if payoff_ratio > 0 else 0.0


def half_kelly(p_win: float, payoff_ratio: float) -> float:
    """Half-Kelly — recommended cho retail vì Kelly thuần quá aggressive."""
    return kelly_fraction(p_win, payoff_ratio) * 0.5


def volatility_target_size(
    target_vol: float,
    realized_vol: float,
    notional_base: float = 1.0,
) -> float:
    """
    Volatility targeting — scale position để đạt target volatility.

    Args:
        target_vol: Target annualized volatility (vd: 0.15 = 15%).
        realized_vol: Realized volatility hiện tại.
        notional_base: Base notional khi realized_vol = target_vol.

    Returns:
        Adjusted notional size.
    """
    if realized_vol == 0:
        return 0.0
    return notional_base * (target_vol / realized_vol)
