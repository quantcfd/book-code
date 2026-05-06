"""
QuantCFD — Chương 8.13.5
Walk-Forward Analysis for Mean Reversion

MR-specific WFA differences from trend:
- IS shorter (2-3 năm vs 3-5 năm) because more trades
- Optimize fewer params (curse of dimensionality with high turnover)
- Step shorter (3-6 months vs 1 year)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from itertools import product


def backtest_bb_mr(
    df: pd.DataFrame,
    period: int,
    std_mult: float,
    cost: float = 0.0008,
) -> float:
    """
    Bollinger Bands MR backtest. Returns Sharpe.

    Long when close < lower band, exit at middle.
    Short when close > upper band, exit at middle.
    """
    out = df.copy()
    sma = out["close"].rolling(period).mean()
    std = out["close"].rolling(period).std()
    out["upper"] = (sma + std_mult * std).shift(1)
    out["lower"] = (sma - std_mult * std).shift(1)
    out["mid"] = sma.shift(1)

    position = 0
    positions = []
    for i in range(len(out)):
        row = out.iloc[i]
        if pd.isna(row["lower"]):
            positions.append(0)
            continue
        price = row["close"]

        if position == 0:
            if price < row["lower"]:
                position = 1
            elif price > row["upper"]:
                position = -1
        elif position == 1:
            if price >= row["mid"] or price < row["lower"] * 0.92:
                position = 0
        elif position == -1:
            if price <= row["mid"] or price > row["upper"] * 1.08:
                position = 0

        positions.append(position)

    out["position"] = positions
    out["asset_return"] = out["close"].pct_change()
    out["pos_change"] = pd.Series(positions, index=out.index).diff().abs().fillna(0)
    out["strat_return"] = (
        out["position"] * out["asset_return"] - out["pos_change"] * cost
    )
    out_clean = out.dropna()

    if len(out_clean) < 30 or out_clean["strat_return"].std() == 0:
        return -999

    sharpe = (
        out_clean["strat_return"].mean() / out_clean["strat_return"].std()
    ) * np.sqrt(252)
    return sharpe


def walk_forward_mr_bb(
    df: pd.DataFrame,
    is_months: int = 24,
    oos_months: int = 6,
    step_months: int = 3,
    period_grid=(15, 20, 25, 30),
    std_grid=(1.5, 2.0, 2.5),
    cost: float = 0.0008,
) -> pd.DataFrame:
    """
    Walk-forward analysis for Bollinger Bands MR strategy.

    Args:
        df: Price data with 'close' column.
        is_months: In-sample window in months.
        oos_months: Out-of-sample window in months.
        step_months: Step size between windows.
        period_grid: BB period values to test.
        std_grid: BB std multiplier values to test.
        cost: Round-trip cost per trade.

    Returns:
        DataFrame with columns: window, is_period, oos_period,
        best_period, best_std, is_sharpe, oos_sharpe, wfe.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    start = df.index[0]
    end = df.index[-1]
    results = []
    window_id = 0
    is_start = start

    while True:
        is_end = is_start + pd.DateOffset(months=is_months)
        oos_start = is_end
        oos_end = oos_start + pd.DateOffset(months=oos_months)

        if oos_end > end:
            break

        is_data = df[is_start:is_end]
        oos_data = df[oos_start:oos_end]

        if len(is_data) < 100 or len(oos_data) < 30:
            is_start = is_start + pd.DateOffset(months=step_months)
            window_id += 1
            continue

        best_sharpe_is = -999
        best_params = None
        for period, std_mult in product(period_grid, std_grid):
            sharpe = backtest_bb_mr(is_data, period, std_mult, cost)
            if sharpe > best_sharpe_is:
                best_sharpe_is = sharpe
                best_params = (period, std_mult)

        if best_params is None:
            is_start = is_start + pd.DateOffset(months=step_months)
            window_id += 1
            continue

        oos_sharpe = backtest_bb_mr(
            oos_data, best_params[0], best_params[1], cost
        )

        wfe = (
            oos_sharpe / best_sharpe_is
            if best_sharpe_is > 0 else 0
        )

        results.append({
            "window": window_id,
            "is_period": f"{is_start.year}-{is_end.year}",
            "oos_period": f"{oos_start.strftime('%Y-%m')}_to_{oos_end.strftime('%Y-%m')}",
            "best_period": best_params[0],
            "best_std": best_params[1],
            "is_sharpe": best_sharpe_is,
            "oos_sharpe": oos_sharpe,
            "wfe": wfe,
        })

        is_start = is_start + pd.DateOffset(months=step_months)
        window_id += 1

    return pd.DataFrame(results)


def deployment_verdict_mr(wfa: pd.DataFrame) -> dict:
    """
    GO/NO-GO verdict for MR strategy based on WFA.

    Thresholds:
    - GO: avg OOS Sharpe ≥ 0.5, % positive ≥ 60%, WFE ≥ 0.5
    - CAUTION: avg OOS Sharpe ≥ 0.3, % positive ≥ 50%
    - NO-GO: any below thresholds
    """
    if len(wfa) == 0:
        return {"verdict": "NO_DATA"}

    avg_oos = wfa["oos_sharpe"].mean()
    median_oos = wfa["oos_sharpe"].median()
    pct_pos = (wfa["oos_sharpe"] > 0).mean() * 100
    avg_wfe = wfa["wfe"].mean()
    n_consecutive_neg = 0
    max_consecutive_neg = 0
    for s in wfa["oos_sharpe"]:
        if s < 0:
            n_consecutive_neg += 1
            max_consecutive_neg = max(max_consecutive_neg, n_consecutive_neg)
        else:
            n_consecutive_neg = 0

    if (
        avg_oos >= 0.5
        and pct_pos >= 60
        and avg_wfe >= 0.5
        and max_consecutive_neg < 2
    ):
        verdict = "GO — strategy robust"
    elif avg_oos >= 0.3 and pct_pos >= 50 and max_consecutive_neg < 3:
        verdict = "PROCEED WITH CAUTION — marginal edge"
    else:
        verdict = "NO-GO — insufficient edge"

    return {
        "n_windows": len(wfa),
        "avg_oos_sharpe": avg_oos,
        "median_oos_sharpe": median_oos,
        "pct_positive": pct_pos,
        "avg_wfe": avg_wfe,
        "max_consecutive_negative": max_consecutive_neg,
        "verdict": verdict,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Walk-Forward Analysis — Bollinger Bands MR")
    print("=" * 70)

    # Generate longer synthetic data with regime changes
    np.random.seed(42)
    n = 7 * 252  # 7 years
    dates = pd.date_range("2018-01-01", periods=n, freq="D")

    # Mix of MR and trending periods
    returns = np.zeros(n)
    for i in range(0, n, 252):
        end = min(i + 252, n)
        regime_drift = np.random.choice([-0.0002, 0.0001, 0.0005])
        returns[i:end] = np.random.randn(end - i) * 0.012 + regime_drift

    prices = 2000 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({"close": prices}, index=dates)

    print(f"\nData: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} bars)")
    print(f"Price range: {df['close'].min():.0f} → {df['close'].max():.0f}")

    print("\nRunning WFA (this takes 30-60 seconds)...")
    wfa = walk_forward_mr_bb(
        df,
        is_months=24,
        oos_months=6,
        step_months=3,
        period_grid=(15, 20, 25, 30),
        std_grid=(1.5, 2.0, 2.5),
    )

    print(f"\n{len(wfa)} windows generated")
    print()
    print(wfa[
        ["window", "oos_period", "best_period", "best_std",
         "is_sharpe", "oos_sharpe", "wfe"]
    ].round(3).to_string(index=False))

    print("\n" + "─" * 70)
    summary = deployment_verdict_mr(wfa)
    print("WFA Summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.3f}")
        else:
            print(f"  {k:30s}: {v}")
