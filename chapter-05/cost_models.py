"""
QuantCFD - Chapter 5 - Cost Models
====================================
Section 5.3: CFD-specific cost components.

Three cost types:
    1. Spread: session-aware (24-element profile per instrument)
    2. Swap: overnight financing, 3x on Wednesday
    3. Slippage: base + ATR-based component

Reuse trong engine ở backtest_engine.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================================
#  SPREAD PROFILES — đo bằng pip-equivalent units
# ============================================================================
# Mỗi profile là 24-element array, index = giờ UTC.
# Số liệu lấy từ broker IC Markets (ECN). Adjust ±20% cho broker khác.

EURUSD_SPREAD_BY_HOUR = np.array(
    [
        # 00-07: Asia + early London
        1.5, 1.8, 2.0, 2.2, 2.0, 1.5, 1.0, 0.5,
        # 08-15: London + NY overlap (cheapest)
        0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.4, 0.5,
        # 16-23: NY + rollover spike at 21
        0.5, 0.6, 0.7, 0.8, 1.2, 4.5, 3.0, 2.0,
    ]
)

# XAUUSD: pip = 0.01 ($), spread đo bằng cents
XAUUSD_SPREAD_BY_HOUR = np.array(
    [
        35, 38, 40, 38, 32, 25, 18, 15,
        13, 12, 12, 13, 15, 15, 15, 18,
        18, 20, 22, 25, 30, 80, 60, 45,
    ]
) / 100.0  # convert cents → dollars

# US500 indices CFD: spread đo bằng index points
US500_SPREAD_BY_HOUR = np.array(
    [
        1.5, 1.8, 2.0, 2.0, 1.8, 1.5, 1.0, 0.7,
        0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.5, 0.5,
        0.5, 0.6, 0.8, 1.0, 1.5, 4.0, 3.0, 2.0,
    ]
)


def get_spread_per_bar(
    timestamps: pd.DatetimeIndex,
    profile: np.ndarray = EURUSD_SPREAD_BY_HOUR,
) -> np.ndarray:
    """
    Trả về spread cho mỗi timestamp dựa trên giờ UTC.

    Args:
        timestamps: DatetimeIndex (must be UTC).
        profile: 24-element array, index=hour UTC.

    Returns:
        np.ndarray cùng length với timestamps.
    """
    if len(profile) != 24:
        raise ValueError(f"Profile phải có 24 elements, got {len(profile)}")
    hours = timestamps.hour.values
    return profile[hours]


# ============================================================================
#  SWAP RATES — % per year
# ============================================================================
# Source: typical retail broker rates 2024.
# Long âm = pay carry. Short dương = receive carry. Crypto thường cả 2 phía âm.

DEFAULT_SWAP_RATES = {
    "EURUSD": {"long": -2.5, "short": +1.8},
    "GBPUSD": {"long": -1.0, "short": +0.5},
    "USDJPY": {"long": +1.5, "short": -2.0},
    "AUDJPY": {"long": +3.5, "short": -4.0},  # carry trade favorite
    "XAUUSD": {"long": -3.0, "short": +1.5},
    "US500":  {"long": -4.5, "short":  0.0},
    "GER40":  {"long": -3.5, "short":  0.0},
    "JP225":  {"long": -1.0, "short":  0.0},
    "BTCUSD": {"long": -8.0, "short": -2.0},  # crypto: cả 2 phía âm
}


def calculate_swap_per_bar(
    positions: pd.Series,
    bar_timestamps: pd.DatetimeIndex,
    swap_long_pct: float,
    swap_short_pct: float,
    bars_per_day: int = 1,
) -> pd.Series:
    """
    Tính swap charge cho mỗi bar.

    Logic:
        - Khi position != 0, mỗi bar trả 1/bars_per_day daily swap.
        - Nếu bar nằm vào thứ Tư (UTC weekday=2), nhân 3.

    Args:
        positions: Series of {-1, 0, +1} hoặc fractional, index=datetime.
        bar_timestamps: DatetimeIndex của bars.
        swap_long_pct: % per year, dấu âm = cost (vd -2.5 cho EURUSD long).
        swap_short_pct: % per year cho short side.
        bars_per_day: 1 = daily bars; 24 = hourly bars; 1440 = M1 bars.

    Returns:
        pd.Series, value = swap RETURN (dấu dương = receive, âm = pay).
        Engine sẽ ADD trực tiếp vào net returns.
    """
    is_long = positions > 0
    is_short = positions < 0

    # Annual % → per-bar fraction
    daily_long = swap_long_pct / 100.0 / 365.0
    daily_short = swap_short_pct / 100.0 / 365.0
    per_bar_long = daily_long / bars_per_day
    per_bar_short = daily_short / bars_per_day

    rate = pd.Series(0.0, index=positions.index)
    rate.loc[is_long] = per_bar_long
    rate.loc[is_short] = per_bar_short

    # Wednesday triple
    is_wednesday = pd.Series(
        bar_timestamps.weekday == 2, index=positions.index
    )
    multiplier = pd.Series(
        np.where(is_wednesday, 3.0, 1.0), index=positions.index
    )

    # Multiply by |position| (cho fractional sizing)
    return rate * multiplier * positions.abs()


# ============================================================================
#  SLIPPAGE MODEL — base + ATR-linear
# ============================================================================
def calculate_slippage_pips(
    atr_pips: pd.Series,
    base_pips: float = 0.2,
    atr_multiplier: float = 0.05,
) -> pd.Series:
    """
    Slippage = base + atr_multiplier × ATR (đo bằng pips).

    Args:
        atr_pips: Series of ATR values (đo bằng pips), không phải %.
        base_pips: floor slippage khi market quiet.
        atr_multiplier: % của ATR cộng vào slippage.

    Returns:
        Series slippage (pips) cùng length.
    """
    return base_pips + atr_multiplier * atr_pips


def calculate_atr_pips(
    df: pd.DataFrame,
    period: int = 14,
    pip_size: float = 1e-4,
) -> pd.Series:
    """
    Average True Range, output đo bằng pips (chia pip_size).

    Args:
        df: DataFrame phải có cột High, Low, Close.
        period: số kỳ tính ATR (default 14, theo Wilder).
        pip_size: 1e-4 cho EURUSD-like, 1e-2 cho XAUUSD.

    Returns:
        Series ATR đo bằng pips.
    """
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Wilder's smoothing = EMA với α = 1/period
    atr = true_range.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr / pip_size
