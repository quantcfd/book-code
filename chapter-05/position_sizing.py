"""
QuantCFD - Chapter 5 - Position Sizing
==========================================
Section 5.6: Convert position {-1,0,+1} thành lot size cụ thể.

Three approaches:
    1. fixed_lot — luôn 1 lot. Tệ nhất.
    2. fixed_fractional — risk N% equity. Default cho Chương 5-9.
    3. vol_target — scale để đạt target volatility. Chương 10 deep-dive.
"""
from __future__ import annotations

import numpy as np


# ============================================================================
#  PIP VALUES — USD per pip per 1.0 standard lot
# ============================================================================
# 1 standard lot:
#   Forex major: 100,000 unit base currency → pip value $10/pip
#   XAUUSD: 100 oz → pip value $1/pip ($0.01 move per oz)
#   US500: 1 contract → $1/index point
#   BTCUSD CFD: 1 BTC → $1/$1 move

DEFAULT_PIP_VALUES = {
    "EURUSD": 10.0,
    "GBPUSD": 10.0,
    "USDJPY": 10.0,  # gần đúng — JPY pairs có conversion
    "AUDJPY": 10.0,
    "XAUUSD": 1.0,
    "US500":  1.0,
    "GER40":  1.0,
    "JP225":  1.0,
    "BTCUSD": 1.0,
}


def fixed_fractional_lot(
    equity_usd: float,
    stop_distance_pips: float,
    pip_value_per_lot: float,
    risk_pct: float = 0.02,
) -> float:
    """
    Lot size sao cho stop loss = risk_pct × equity (quy tắc 2%).

    Ví dụ:
        EURUSD, equity = $10k, risk 2% = $200, stop 50 pips,
        pip_value = $10/pip/lot → lot = 200 / (50 × 10) = 0.4 lot

    Args:
        equity_usd: equity hiện tại USD.
        stop_distance_pips: khoảng cách entry → stop, đo bằng pips.
        pip_value_per_lot: USD per pip per 1.0 lot (xem DEFAULT_PIP_VALUES).
        risk_pct: % equity sẵn sàng mất mỗi lệnh (default 2%).

    Returns:
        Lot size (có thể fractional). Min 0.01 lot, max 100 lot.
    """
    if stop_distance_pips <= 0:
        return 0.0

    risk_amount = equity_usd * risk_pct
    lot = risk_amount / (stop_distance_pips * pip_value_per_lot)

    # Sanity bounds
    return float(np.clip(lot, 0.01, 100.0))


def vol_target_lot(
    equity_usd: float,
    target_annual_vol: float,
    realized_atr_pct: float,
    pip_value_per_lot: float,
    pip_size: float,
    price: float,
    periods_per_year: int = 252,
) -> float:
    """
    Volatility-target sizing: scale position để strategy có target annualized vol.

    Sẽ học chi tiết ở Chương 10. Đặt placeholder ở đây để engine có thể dùng.

    Args:
        equity_usd: equity hiện tại.
        target_annual_vol: vol mục tiêu (vd 0.15 = 15% annualized).
        realized_atr_pct: ATR / Price (vd 0.005 = 0.5% daily ATR).
        pip_value_per_lot: USD per pip per lot.
        pip_size: 1e-4 cho EURUSD, 1e-2 cho XAU/US500.
        price: giá hiện tại.
        periods_per_year: 252 cho daily, 365 cho crypto.

    Returns:
        Lot size để đạt target vol.
    """
    if realized_atr_pct <= 0:
        return 0.0

    target_daily_vol = target_annual_vol / np.sqrt(periods_per_year)
    # Notional cần thiết để daily P&L vol = target × equity
    notional_target = equity_usd * (target_daily_vol / realized_atr_pct)
    # Convert notional → lot
    notional_per_lot = price * pip_value_per_lot / pip_size
    lot = notional_target / notional_per_lot
    return float(np.clip(lot, 0.01, 100.0))


def fixed_lot(lot_size: float = 0.1) -> float:
    """Trả về lot cố định bất kể equity. Tệ nhất — chỉ dùng test."""
    return float(lot_size)
