"""
QuantCFD Chương 5 — Optimal f (Ralph Vince, 1990)

Optimal f là position sizing maximize Geometric Mean Holding Period Return (GHPR)
qua trades thực tế (không cần giả định Normal như Kelly).

Khác Kelly: Kelly cần biết p(win), avg_win, avg_loss. Optimal f scan f trực tiếp
trên trade-by-trade P&L history.

Vince's TWR (Terminal Wealth Relative):
    TWR(f) = ∏ (1 + f * (-pnl_i / worst_loss))

f optimal là f maximize TWR. Output f là % equity bet per trade.

CẢNH BÁO:
    - Optimal f ngắn hạn cực aggressive — drawdown khủng (60-80%)
    - Practitioners dùng "fractional optimal f": 0.25-0.5 × f_optimal
    - f phụ thuộc heavily vào worst_loss → sensitive với 1 outlier
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def twr(f: float, trade_returns: np.ndarray) -> float:
    """
    Terminal Wealth Relative theo công thức Vince.
    trade_returns: array P&L mỗi trade (USD hoặc %).
    """
    worst_loss = abs(min(trade_returns.min(), -1e-9))  # tránh chia 0
    return np.prod(1 + f * trade_returns / worst_loss)


def optimal_f(
    trade_returns: np.ndarray,
    f_min: float = 0.01,
    f_max: float = 0.99,
) -> dict:
    """Tìm f optimal. Return dict với f, twr, và full curve."""
    if len(trade_returns) < 10:
        raise ValueError(f"Cần ít nhất 10 trades, có {len(trade_returns)}")
    if (trade_returns >= 0).all():
        raise ValueError("Cần ít nhất 1 trade thua để tính worst_loss")

    # Optimization: maximize TWR ↔ minimize -TWR
    result = minimize_scalar(
        lambda f: -twr(f, trade_returns),
        bounds=(f_min, f_max),
        method='bounded',
    )
    f_opt = result.x

    # Full curve để visualize
    f_grid = np.linspace(f_min, f_max, 100)
    twr_curve = np.array([twr(f, trade_returns) for f in f_grid])

    return {
        'optimal_f': f_opt,
        'twr_at_optimal': twr(f_opt, trade_returns),
        'fractional_f_quarter': f_opt * 0.25,
        'fractional_f_half': f_opt * 0.5,
        'f_grid': f_grid,
        'twr_curve': twr_curve,
    }


def kelly_fraction(trade_returns: np.ndarray) -> float:
    """Kelly so sánh: f = p_win - p_loss/payoff."""
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    if len(wins) == 0 or len(losses) == 0:
        return 0
    p_win = len(wins) / len(trade_returns)
    p_loss = 1 - p_win
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    payoff = avg_win / avg_loss
    return p_win - p_loss / payoff


def demo():
    """Synthetic example: 100 trades với edge."""
    np.random.seed(42)
    # 60% win với avg +1%, 40% loss với avg -0.7%
    n_trades = 100
    is_win = np.random.random(n_trades) < 0.6
    pnl = np.where(
        is_win,
        np.random.normal(0.01, 0.005, n_trades),
        np.random.normal(-0.007, 0.003, n_trades),
    )

    print(f"Synthetic trade history: {n_trades} trades")
    print(f"Win rate:    {(pnl > 0).mean()*100:.1f}%")
    print(f"Avg win:     {pnl[pnl > 0].mean()*100:+.2f}%")
    print(f"Avg loss:    {pnl[pnl < 0].mean()*100:+.2f}%")
    print(f"Worst loss:  {pnl.min()*100:+.2f}%")

    print(f"\n--- Kelly ---")
    k = kelly_fraction(pnl)
    print(f"Kelly fraction: {k*100:.2f}% per trade")

    print(f"\n--- Optimal f (Vince) ---")
    res = optimal_f(pnl)
    print(f"Optimal f:           {res['optimal_f']*100:.2f}%")
    print(f"TWR at optimal:      {res['twr_at_optimal']:.3f}")
    print(f"Fractional f (1/4):  {res['fractional_f_quarter']*100:.2f}%  ← retail recommended")
    print(f"Fractional f (1/2):  {res['fractional_f_half']*100:.2f}%")

    print(f"\n--- Comparison ---")
    print(f"Kelly:        {k*100:.2f}%")
    print(f"Optimal f:    {res['optimal_f']*100:.2f}%")
    print(f"Half-Kelly:   {k*0.5*100:.2f}%  ← practitioners thường dùng")
    print(f"Quarter-f:    {res['fractional_f_quarter']*100:.2f}%  ← Vince book recommendation")


if __name__ == '__main__':
    demo()
