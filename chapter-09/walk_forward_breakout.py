"""
QuantCFD — Chương 9.17
Walk-Forward Analysis cho Vol Breakout

WFA cho Keltner breakout strategy.
IS = 2 năm, OOS = 6 tháng, step = 3 tháng.
Optimize 2 params (EMA period, ATR multiplier).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from itertools import product


def compute_atr_for_wfa(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def backtest_keltner(
    df: pd.DataFrame,
    ema_period: int,
    atr_mult: float,
    cost: float = 0.0008,
    periods_per_year: int = 252,
) -> float:
    """Single Keltner backtest, returns annualized Sharpe."""
    df = df.copy()
    df["ema"] = df["close"].ewm(span=ema_period, adjust=False).mean()
    df["atr"] = compute_atr_for_wfa(df, 14)
    df["upper"] = (df["ema"] + atr_mult * df["atr"]).shift(1)
    df["lower"] = (df["ema"] - atr_mult * df["atr"]).shift(1)
    df["mid"] = df["ema"].shift(1)

    position = 0
    positions = []
    for i in range(len(df)):
        row = df.iloc[i]
        if pd.isna(row["upper"]):
            positions.append(0)
            continue
        if position == 0:
            if row["close"] > row["upper"]:
                position = 1
            elif row["close"] < row["lower"]:
                position = -1
        elif position == 1 and row["close"] < row["mid"]:
            position = 0
        elif position == -1 and row["close"] > row["mid"]:
            position = 0
        positions.append(position)

    df["pos"] = positions
    df["ret"] = df["close"].pct_change()
    df["pos_change"] = df["pos"].diff().abs().fillna(0)
    df["strat_ret"] = df["pos"] * df["ret"] - df["pos_change"] * cost
    clean = df["strat_ret"].dropna()

    if len(clean) < 30 or clean.std() == 0:
        return -999

    return (clean.mean() / clean.std()) * np.sqrt(periods_per_year)


def walk_forward_keltner(
    df: pd.DataFrame,
    is_months: int = 24,
    oos_months: int = 6,
    step_months: int = 3,
    ema_grid=(10, 20, 30, 50),
    mult_grid=(1.5, 2.0, 2.5),
    cost: float = 0.0008,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    Walk-forward analysis for Keltner breakout strategy.

    Args:
        df: Price data with 'close', 'high', 'low' columns.
        is_months: In-sample window in months.
        oos_months: Out-of-sample window.
        step_months: Step between windows.
        ema_grid: EMA period values to test.
        mult_grid: ATR multiplier values.

    Returns:
        DataFrame of WFA results per window.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    start = df.index[0]
    end = df.index[-1]

    results = []
    is_start = start
    window_id = 0

    while True:
        is_end = is_start + pd.DateOffset(months=is_months)
        oos_end = is_end + pd.DateOffset(months=oos_months)

        if oos_end > end:
            break

        is_data = df.loc[is_start:is_end]
        oos_data = df.loc[is_end:oos_end]

        if len(is_data) < 100 or len(oos_data) < 30:
            is_start = is_start + pd.DateOffset(months=step_months)
            window_id += 1
            continue

        # Optimize on IS
        best_sharpe = -999
        best_params = None
        for ema_p, mult in product(ema_grid, mult_grid):
            sharpe = backtest_keltner(
                is_data, ema_p, mult, cost, periods_per_year
            )
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (ema_p, mult)

        if best_params is None:
            is_start = is_start + pd.DateOffset(months=step_months)
            window_id += 1
            continue

        # Test on OOS
        oos_sharpe = backtest_keltner(
            oos_data, best_params[0], best_params[1], cost, periods_per_year
        )
        wfe = oos_sharpe / best_sharpe if best_sharpe > 0 else 0

        results.append({
            "window": window_id,
            "is_period": f"{is_start.date()}_to_{is_end.date()}",
            "oos_period": f"{is_end.date()}_to_{oos_end.date()}",
            "best_ema": best_params[0],
            "best_mult": best_params[1],
            "is_sharpe": best_sharpe,
            "oos_sharpe": oos_sharpe,
            "wfe": wfe,
        })

        is_start = is_start + pd.DateOffset(months=step_months)
        window_id += 1

    return pd.DataFrame(results)


def wfa_verdict(wfa_df: pd.DataFrame) -> dict:
    """GO/CAUTION/NO-GO verdict from WFA results."""
    if len(wfa_df) == 0:
        return {"verdict": "NO_DATA"}

    avg_oos = wfa_df["oos_sharpe"].mean()
    median_oos = wfa_df["oos_sharpe"].median()
    pct_pos = (wfa_df["oos_sharpe"] > 0).mean() * 100
    avg_wfe = wfa_df["wfe"].mean()

    n_consecutive_neg = max_consecutive_neg = 0
    for s in wfa_df["oos_sharpe"]:
        if s < 0:
            n_consecutive_neg += 1
            max_consecutive_neg = max(max_consecutive_neg, n_consecutive_neg)
        else:
            n_consecutive_neg = 0

    if avg_oos >= 0.5 and pct_pos >= 60 and avg_wfe >= 0.5 and max_consecutive_neg < 2:
        verdict = "GO — strategy robust"
    elif avg_oos >= 0.3 and pct_pos >= 50 and max_consecutive_neg < 3:
        verdict = "PROCEED WITH CAUTION — marginal edge"
    else:
        verdict = "NO-GO — insufficient edge"

    return {
        "n_windows": len(wfa_df),
        "avg_oos_sharpe": avg_oos,
        "median_oos_sharpe": median_oos,
        "pct_positive": pct_pos,
        "avg_wfe": avg_wfe,
        "max_consecutive_negative": max_consecutive_neg,
        "verdict": verdict,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("WFA — Keltner Breakout (XAUUSD H4 synthetic)")
    print("=" * 70)

    np.random.seed(42)
    n = 7 * 252
    dates = pd.date_range("2018-01-01", periods=n, freq="D")

    rets = np.zeros(n)
    for i in range(0, n, 252):
        end = min(i + 252, n)
        drift = np.random.choice([-0.0002, 0, 0.0003])
        rets[i:end] = np.random.randn(end - i) * 0.012 + drift

    closes = 1500 * np.exp(np.cumsum(rets))
    daily_vol = np.abs(rets) + 0.005
    highs = closes * (1 + daily_vol)
    lows = closes * (1 - daily_vol)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]

    df = pd.DataFrame({
        "open": opens,
        "high": np.maximum(np.maximum(opens, closes), highs),
        "low": np.minimum(np.minimum(opens, closes), lows),
        "close": closes,
    }, index=dates)

    print(f"\nData: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} bars)")
    print("\nRunning WFA (30-60 seconds)...")

    wfa = walk_forward_keltner(
        df,
        is_months=24, oos_months=6, step_months=3,
        ema_grid=(10, 20, 30),
        mult_grid=(1.5, 2.0, 2.5),
    )

    print(f"\n{len(wfa)} windows generated\n")
    if len(wfa) > 0:
        print(wfa[
            ["window", "oos_period", "best_ema", "best_mult",
             "is_sharpe", "oos_sharpe", "wfe"]
        ].round(3).to_string(index=False))

        print(f"\n{'─' * 70}")
        verdict = wfa_verdict(wfa)
        print("WFA Summary:")
        for k, v in verdict.items():
            if isinstance(v, float):
                print(f"  {k:30s}: {v:.3f}")
            else:
                print(f"  {k:30s}: {v}")
