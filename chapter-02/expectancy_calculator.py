"""
QuantCFD - Chapter 2 - Expectancy Calculator
=============================================
Section 2.1: Expectancy là metric duy nhất quan trọng.

Hàm expectancy_report() nhận pd.Series các lệnh PnL (USD) và trả về dict
với 9 metrics quan trọng nhất: expectancy, win rate, payoff ratio,
profit factor, Kelly fraction.

Demo: 2 chiến lược (scalping win rate cao, trend win rate thấp) — chiến
lược thắng nhiều thực tế lại lỗ vì avg loss quá to.

Chạy:
    python chapter-02/expectancy_calculator.py
"""
import numpy as np
import pandas as pd


def expectancy_report(trades: pd.Series) -> dict:
    """
    Tính metrics quan trọng từ một series PnL.

    Args:
        trades: pd.Series chứa PnL (lãi/lỗ) của từng lệnh, USD.

    Returns:
        dict với:
            n_trades, win_rate, avg_win, avg_loss,
            expectancy, payoff_ratio, profit_factor,
            kelly_fraction, total_pnl
    """
    trades = trades.dropna()
    n = len(trades)
    if n == 0:
        return {}

    wins = trades[trades > 0]
    losses = trades[trades < 0]

    p_win = len(wins) / n
    p_loss = len(losses) / n
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0

    expectancy = p_win * avg_win - p_loss * avg_loss
    payoff = avg_win / avg_loss if avg_loss > 0 else float("inf")
    profit_factor = (
        wins.sum() / abs(losses.sum()) if losses.sum() < 0 else float("inf")
    )

    # Kelly fraction (sẽ học sâu ở Chương 10)
    # Công thức: f* = p - q/b, với p=p_win, q=p_loss, b=payoff
    kelly = p_win - (p_loss / payoff) if payoff > 0 and payoff != float("inf") else 0.0

    return {
        "n_trades": n,
        "win_rate": p_win,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "payoff_ratio": payoff,
        "profit_factor": profit_factor,
        "kelly_fraction": kelly,
        "total_pnl": trades.sum(),
    }


def print_report(name: str, report: dict) -> None:
    """In report đẹp, có color-coding cho expectancy âm/dương."""
    print(f"\n=== {name} ===")
    for k, v in report.items():
        if isinstance(v, float):
            if k in ("win_rate",):
                print(f"  {k:18s} = {v:>10.2%}")
            elif k == "expectancy" and v < 0:
                print(f"  {k:18s} = {v:>10.4f}  ⚠ ÂM")
            else:
                print(f"  {k:18s} = {v:>10.4f}")
        else:
            print(f"  {k:18s} = {v}")


def main() -> None:
    # Tạo 3 chiến lược ví dụ giống bảng ở Section 2.1
    np.random.seed(42)

    # A. Scalping: win rate 82%, avg win nhỏ, avg loss to (đặc trưng "cá ăn voi ăn lại")
    strategy_A = pd.Series(
        np.concatenate(
            [
                np.random.uniform(2, 14, 82),  # 82 lệnh thắng nhỏ
                np.random.uniform(-65, -25, 18),  # 18 lệnh thua to
            ]
        )
    )

    # B. Trend following: win rate 38%, avg win to, avg loss vừa
    strategy_B = pd.Series(
        np.concatenate(
            [
                np.random.uniform(150, 450, 38),  # 38 lệnh thắng to
                np.random.uniform(-130, -60, 62),  # 62 lệnh thua vừa
            ]
        )
    )

    # C. Hùng's breakout: win rate 78%, avg win nhỏ, avg loss CỰC to (gap risk)
    strategy_C = pd.Series(
        np.concatenate(
            [
                np.random.uniform(5, 18, 78),  # 78 lệnh thắng nhỏ
                np.random.uniform(-450, -300, 22),  # 22 lệnh thua to do gap
            ]
        )
    )

    print_report("A. Scalping range  (win 82%)", expectancy_report(strategy_A))
    print_report("B. Trend following (win 38%)", expectancy_report(strategy_B))
    print_report("C. Breakout (Hùng) (win 78%)", expectancy_report(strategy_C))

    print(
        "\nKết luận: Chiến lược B win rate THẤP NHẤT lại là chiến lược DUY NHẤT có "
        "expectancy dương. Win rate là cái bẫy tâm lý — đừng để nó lừa anh em."
    )


if __name__ == "__main__":
    main()
