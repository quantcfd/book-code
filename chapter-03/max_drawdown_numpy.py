"""
QuantCFD - Chapter 3 - Max Drawdown (numpy 4 lines)
=====================================================
Section 3.2: Tính max drawdown bằng 4 dòng numpy thuần.

Cùng task viết bằng VBA chiếm ~25 dòng. Đây là sức mạnh của vectorization.

Chạy:
    python chapter-03/max_drawdown_numpy.py
"""
import numpy as np


def max_drawdown(returns: np.ndarray) -> float:
    """
    Max drawdown từ một array returns. 4 dòng numpy thuần.

    Args:
        returns: array các returns hàng kỳ (vd: daily returns).

    Returns:
        Max drawdown dưới dạng số âm (vd: -0.18 = -18% drawdown).
    """
    equity = np.cumprod(1 + returns)  # đường equity
    running_max = np.maximum.accumulate(equity)  # đỉnh gần nhất
    drawdown = (equity - running_max) / running_max  # drawdown tại mỗi điểm
    return drawdown.min()  # max DD = min của series drawdown


def drawdown_series(returns: np.ndarray) -> np.ndarray:
    """Trả về cả series drawdown (cho việc plot/analyze)."""
    equity = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(equity)
    return (equity - running_max) / running_max


def main() -> None:
    np.random.seed(42)
    # Mô phỏng 1000 ngày returns ~ N(0.0005, 0.015)
    returns = np.random.normal(0.0005, 0.015, 1000)

    mdd = max_drawdown(returns)
    print(f"Max drawdown: {mdd:.2%}")

    # Phân tích thêm
    dd = drawdown_series(returns)
    avg_dd = dd[dd < 0].mean()  # drawdown trung bình khi đang DD
    n_dd_periods = (dd < -0.05).sum()  # số ngày DD > 5%

    print(f"Average DD (when in DD): {avg_dd:.2%}")
    print(f"Days with DD > 5%:       {n_dd_periods}/1000  ({n_dd_periods/10:.1f}%)")

    # So sánh số dòng
    print("\nĐây là 4 dòng code:")
    print('    equity = np.cumprod(1 + returns)')
    print('    running_max = np.maximum.accumulate(equity)')
    print('    drawdown = (equity - running_max) / running_max')
    print('    return drawdown.min()')
    print(
        "\nCùng task viết bằng VBA chiếm ~25 dòng (loop manual để tracking "
        "running max). Vectorization tiết kiệm thời gian + giảm lỗi off-by-one."
    )


if __name__ == "__main__":
    main()
