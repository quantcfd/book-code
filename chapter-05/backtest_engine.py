"""
QuantCFD - Chapter 5 - Vectorized Backtest Engine
====================================================
Section 5.4: Core engine với CFD-specific costs.

Pipeline:
    Data → Signal → Position → Execution(+costs) → P&L → Metrics

Engine handle 4 cost components:
    - Spread (session-aware, 24h profile)
    - Swap (3x Wednesday)
    - Slippage (base + ATR-linear)
    - Optional: commission per trade

Vectorized = không for-loop. Backtest 5 năm M1 data trong < 5 giây.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

# Import cost models nếu run as part of repo
try:
    from .cost_models import (
        get_spread_per_bar,
        calculate_atr_pips,
        calculate_slippage_pips,
        calculate_swap_per_bar,
    )
except ImportError:
    # Standalone fallback — import từ same directory
    from cost_models import (
        get_spread_per_bar,
        calculate_atr_pips,
        calculate_slippage_pips,
        calculate_swap_per_bar,
    )


# ============================================================================
#  RESULT CONTAINER
# ============================================================================
@dataclass
class BacktestResult:
    """Container cho output của một backtest run."""

    equity:      pd.Series
    returns:     pd.Series
    positions:   pd.Series
    spread_cost: pd.Series
    swap_cost:   pd.Series
    slip_cost:   pd.Series
    n_trades:    int
    initial_capital: float = 10_000.0
    metadata:    dict = field(default_factory=dict)

    @property
    def total_return(self) -> float:
        return float(self.equity.iloc[-1] / self.equity.iloc[0] - 1)

    @property
    def total_cost(self) -> float:
        return float(
            self.spread_cost.sum()
            + self.swap_cost.sum()
            + self.slip_cost.sum()
        )

    def stats(self, periods_per_year: int = 252) -> dict:
        """Tính các metrics cơ bản. Đầy đủ ở Chương 6 (metrics.py)."""
        r = self.returns.dropna()
        if len(r) == 0 or r.std() == 0:
            return {}

        sharpe = r.mean() / r.std() * np.sqrt(periods_per_year)
        cum = (1 + r).cumprod()
        max_dd = float((cum / cum.cummax() - 1).min())

        return {
            "total_return":  self.total_return,
            "annual_return": float(r.mean() * periods_per_year),
            "annual_vol":    float(r.std() * np.sqrt(periods_per_year)),
            "sharpe":        float(sharpe),
            "max_drawdown":  max_dd,
            "n_trades":      self.n_trades,
            "total_cost":    self.total_cost,
            "cost_drag":     self.total_cost / self.initial_capital,
        }

    def equity_curve_df(self) -> pd.DataFrame:
        """DataFrame để plot dễ hơn."""
        return pd.DataFrame(
            {
                "equity": self.equity,
                "position": self.positions,
                "returns": self.returns,
                "spread_cost": self.spread_cost,
                "swap_cost": self.swap_cost,
                "slip_cost": self.slip_cost,
            }
        )


# ============================================================================
#  CORE ENGINE
# ============================================================================
def run_backtest(
    df: pd.DataFrame,
    signal_fn: Callable[[pd.DataFrame], pd.Series],
    *,
    initial_capital: float = 10_000.0,
    pip_size: float = 1e-4,
    spread_profile: Optional[np.ndarray] = None,
    swap_long_pct: float = 0.0,
    swap_short_pct: float = 0.0,
    slippage_base_pips: float = 0.2,
    slippage_atr_mult: float = 0.05,
    atr_period: int = 14,
    bars_per_day: int = 1,
    enable_asserts: bool = True,
) -> BacktestResult:
    """
    Vectorized backtest engine với CFD-specific costs.

    Args:
        df: DataFrame index=datetime UTC, cột Open/High/Low/Close (Volume optional).
        signal_fn: function df -> pd.Series of {-1, 0, +1} cùng index.
        initial_capital: vốn USD ban đầu.
        pip_size: 1e-4 cho EUR/USD-like, 1e-2 cho XAU/USD và US500.
        spread_profile: 24-element array (giờ UTC) chứa spread (pips).
                        None = dùng spread cố định 0.5 pip.
        swap_long_pct: swap rate % per year cho long (vd -2.5 cho EURUSD).
        swap_short_pct: swap rate % per year cho short.
        slippage_base_pips: base slippage khi market quiet.
        slippage_atr_mult: % ATR cộng vào slippage.
        atr_period: period tính ATR (default 14).
        bars_per_day: 1 cho daily, 24 cho hourly, 1440 cho M1.
        enable_asserts: True = check sanity sau mỗi run.

    Returns:
        BacktestResult với equity curve + breakdown costs.

    Raises:
        AssertionError nếu engine detect bug (vd cost âm, position vô lý).
    """
    df = df.copy()

    # Đảm bảo index là DatetimeIndex UTC
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index phải là DatetimeIndex")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # ---- 1. Generate signal + shift để chống look-ahead ----
    raw_signal = signal_fn(df).fillna(0)
    df["position"] = raw_signal.shift(1).fillna(0)

    # ---- 2. Tính returns + ATR ----
    df["mkt_ret"] = df["Close"].pct_change().fillna(0)
    df["atr_pips"] = calculate_atr_pips(df, period=atr_period, pip_size=pip_size)

    # ---- 3. Spread cost mỗi khi đổi position ----
    if spread_profile is None:
        spread_pips_per_bar = np.full(len(df), 0.5)
    else:
        spread_pips_per_bar = get_spread_per_bar(df.index, spread_profile)

    pos_change = df["position"].diff().abs().fillna(0)
    # Cost as fraction of price = spread_in_price / price
    df["spread_cost"] = (
        spread_pips_per_bar * pip_size / df["Close"]
    ) * pos_change

    # ---- 4. Slippage mỗi khi vào/ra lệnh ----
    slip_pips = calculate_slippage_pips(
        df["atr_pips"].fillna(0),
        base_pips=slippage_base_pips,
        atr_multiplier=slippage_atr_mult,
    )
    df["slip_cost"] = (slip_pips * pip_size / df["Close"]) * pos_change

    # ---- 5. Swap charge mỗi bar khi giữ position ----
    df["swap_return"] = calculate_swap_per_bar(
        df["position"],
        df.index,
        swap_long_pct=swap_long_pct,
        swap_short_pct=swap_short_pct,
        bars_per_day=bars_per_day,
    )

    # ---- 6. Strategy returns (đã trừ all costs) ----
    df["gross_ret"] = df["position"] * df["mkt_ret"]
    df["net_ret"] = (
        df["gross_ret"]
        - df["spread_cost"]
        - df["slip_cost"]
        + df["swap_return"]  # swap can be + (receive) or - (pay)
    )

    # ---- 7. Equity curve ----
    df["equity"] = initial_capital * (1 + df["net_ret"]).cumprod()

    # ---- 8. Đếm trades round-trip ----
    n_changes = int((pos_change > 0).sum())
    n_trades = max(1, n_changes // 2)  # mỗi round-trip = 2 changes

    # ---- 9. Sanity asserts ----
    if enable_asserts:
        assert df["equity"].iloc[-1] > 0, "Equity âm — bug position sizing?"
        assert df["spread_cost"].sum() >= 0, "Spread cost không thể âm"
        assert df["slip_cost"].sum() >= 0, "Slippage không thể âm"
        assert (
            df["net_ret"].abs().max() < 1.0
        ), "Returns > 100% trong 1 bar — bug data?"

    # Convert swap_return → swap_cost (positive = cost)
    return BacktestResult(
        equity=df["equity"],
        returns=df["net_ret"],
        positions=df["position"],
        spread_cost=df["spread_cost"],
        swap_cost=-df["swap_return"],  # report positive when paying
        slip_cost=df["slip_cost"],
        n_trades=n_trades,
        initial_capital=initial_capital,
        metadata={
            "pip_size": pip_size,
            "spread_profile": (
                "session-aware" if spread_profile is not None else "flat 0.5"
            ),
            "swap_long_pct": swap_long_pct,
            "swap_short_pct": swap_short_pct,
        },
    )


# ============================================================================
#  CLI / DEMO
# ============================================================================
if __name__ == "__main__":
    print(
        "Đây là module engine. Chạy demo_xauusd_macross.py để xem ví dụ "
        "đầy đủ với data thật."
    )
