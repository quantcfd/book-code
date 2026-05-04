"""
QuantCFD - Chapter 3 - Vectorization Speed Test
=================================================
Section 3.2: Numpy — vì sao for-loop là kẻ thù.

Demo: tính returns từ 1 triệu giá, so sánh for-loop Python pure vs
numpy vectorized. Tỷ lệ kỳ vọng: 100-500x nhanh hơn.

Chạy:
    python chapter-03/speed_test.py
"""
import time

import numpy as np


def returns_python_loop(prices: np.ndarray) -> list:
    """Cách CHẬM — Python for-loop, giống Excel/VBA mindset."""
    returns = []
    for i in range(1, len(prices)):
        returns.append((prices[i] - prices[i - 1]) / prices[i - 1])
    return returns


def returns_numpy(prices: np.ndarray) -> np.ndarray:
    """Cách NHANH — vectorized với numpy."""
    return np.diff(prices) / prices[:-1]


def main() -> None:
    np.random.seed(42)
    n = 1_000_000
    prices = np.random.rand(n) * 100

    print(f"\nTest: tính returns từ {n:,} giá ngẫu nhiên")
    print("=" * 60)

    # For-loop
    start = time.time()
    rets_slow = returns_python_loop(prices)
    time_slow = time.time() - start

    # Numpy
    start = time.time()
    rets_fast = returns_numpy(prices)
    time_fast = time.time() - start

    print(f"For-loop Python:  {time_slow:8.4f} giây")
    print(f"Numpy vectorized: {time_fast:8.4f} giây")
    print(f"Tỷ lệ:           {time_slow / time_fast:8.0f}x nhanh hơn")
    print()

    # Verify giống nhau
    if np.allclose(rets_slow[:100], rets_fast[:100]):
        print("✓ Cả 2 cách cho cùng kết quả (verified 100 phần tử đầu).")

    print(
        "\nKhi backtest 5 năm × 5 instruments × 1-min bars (~6.5M rows), "
        "khác biệt là 30 phút (loop) vs 8 giây (vectorized). "
        "Đây không phải optimization sang chảnh — là khác biệt giữa "
        "'research được' và 'bỏ cuộc vì chậm quá'."
    )


if __name__ == "__main__":
    main()
