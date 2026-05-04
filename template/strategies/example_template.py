"""
strategies/example_template.py — Template cho một strategy file.

Mỗi strategy file nên expose một function `generate_signals(df)` trả về
pd.Series với position từ {-1, 0, +1} (hoặc fractional).

Example skeleton dưới đây — copy file này, đổi tên, thay logic của riêng anh em.
"""
import numpy as np
import pandas as pd


def generate_signals(df: pd.DataFrame) -> pd.Series:
    """
    Sinh tín hiệu giao dịch từ OHLCV DataFrame.

    Args:
        df: DataFrame phải có ít nhất cột 'Close'.

    Returns:
        pd.Series với same index as df. Values:
            +1 = long
             0 = flat
            -1 = short
    """
    # Ví dụ: MA crossover đơn giản
    ma_fast = df["Close"].rolling(20).mean()
    ma_slow = df["Close"].rolling(50).mean()

    # Signal tại close T → position vào ngày T+1 → CHỐNG LOOK-AHEAD
    raw_signal = np.where(ma_fast > ma_slow, 1, -1)
    position = pd.Series(raw_signal, index=df.index).shift(1)

    return position.fillna(0)
