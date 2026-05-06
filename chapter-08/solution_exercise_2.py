"""
QuantCFD — Chương 8, Bài tập 2
Pairs Trading XAU-XAG với cointegration test (90 phút)

Yêu cầu:
- Test cointegration XAUUSD vs XAGUSD daily
- Compute hedge ratio
- Trade Z-score của spread
- Compute metrics
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def cointegration_test(y: pd.Series, x: pd.Series, use_logs: bool = True) -> dict:
    """
    Engle-Granger cointegration test (manual implementation).

    Step 1: Log transform (default — real cointegration is on log prices).
    Step 2: OLS log(y) ~ log(x) to get hedge ratio and residuals.
    Step 3: Compute AR(1) coefficient of residuals as MR proxy.

    For more rigorous test, use statsmodels.tsa.stattools.coint.
    """
    common = y.dropna().index.intersection(x.dropna().index)
    y_arr = y.loc[common].values
    x_arr = x.loc[common].values

    if len(y_arr) < 100:
        return {"error": "insufficient data"}

    if use_logs:
        y_arr = np.log(y_arr)
        x_arr = np.log(x_arr)

    # Hedge ratio via OLS (no intercept for simplicity)
    x_centered = x_arr - x_arr.mean()
    y_centered = y_arr - y_arr.mean()
    hedge_ratio = (
        np.sum(x_centered * y_centered) / np.sum(x_centered ** 2)
    )

    # Residuals (spread)
    intercept = y_arr.mean() - hedge_ratio * x_arr.mean()
    spread = y_arr - hedge_ratio * x_arr - intercept

    # Lag-1 autocorrelation of spread (MR proxy)
    if len(spread) < 2:
        return {"error": "insufficient data"}

    spread_lag = spread[:-1]
    spread_curr = spread[1:]
    var_lag = np.var(spread_lag)
    if var_lag <= 0:
        ar1_coef = 1.0
    else:
        # AR(1): spread_t = phi * spread_{t-1} + noise
        ar1_coef = np.cov(spread_lag, spread_curr)[0, 1] / var_lag

    # Half-life from AR(1)
    if ar1_coef >= 1.0 or ar1_coef <= 0:
        half_life = np.inf
    else:
        half_life = -np.log(2) / np.log(ar1_coef)

    # Cointegrated heuristic: AR(1) < 0.95 and half-life finite/reasonable
    cointegrated = (0 < ar1_coef < 0.95) and (3 < half_life < 100)

    return {
        "cointegrated": cointegrated,
        "ar1_coefficient": ar1_coef,
        "half_life": half_life,
        "hedge_ratio": hedge_ratio,
        "intercept": intercept,
        "spread_std": np.std(spread),
        "use_logs": use_logs,
    }


def pairs_strategy(
    y_series: pd.Series,
    x_series: pd.Series,
    hedge_ratio: float,
    lookback: int = 60,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_stop: float = 4.0,
) -> pd.DataFrame:
    """
    Pairs trading on Z-score of spread.

    Spread = y - hedge_ratio * x
    Long spread (long y, short x) when Z << 0
    Short spread (short y, long x) when Z >> 0
    """
    common = y_series.dropna().index.intersection(x_series.dropna().index)
    y = y_series.loc[common]
    x = x_series.loc[common]

    out = pd.DataFrame({"y": y, "x": x})
    out["spread"] = y - hedge_ratio * x
    out["mean"] = out["spread"].rolling(lookback).mean().shift(1)
    out["std"] = out["spread"].rolling(lookback).std().shift(1)
    out["zscore"] = (out["spread"] - out["mean"]) / out["std"]

    position = 0
    positions = []
    for i in range(len(out)):
        z = out["zscore"].iloc[i]
        if pd.isna(z):
            positions.append(0)
            continue
        if position == 0:
            if z < -z_entry:
                position = 1
            elif z > z_entry:
                position = -1
        elif position == 1:
            if z >= -z_exit or z < -z_stop:
                position = 0
        elif position == -1:
            if z <= z_exit or z > z_stop:
                position = 0
        positions.append(position)

    out["position"] = positions
    out["spread_change"] = out["spread"].diff()
    out["pnl"] = out["position"].shift(1) * out["spread_change"]

    return out


if __name__ == "__main__":
    np.random.seed(42)
    n = 1500
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    # Construct cointegrated pair via AR(1) spread approach.
    # Spread follows AR(1) with phi=0.92 → half-life ≈ 8 bars.
    # log(xau) = hedge_ratio * log(xag) + spread + intercept
    common_factor = np.cumsum(np.random.randn(n) * 0.012)

    # Generate AR(1) spread with controlled persistence
    phi = 0.92  # → half-life = -ln(2)/ln(0.92) ≈ 8.3 bars
    sigma_eta = 0.015  # innovation std
    spread_true = np.zeros(n)
    for i in range(1, n):
        spread_true[i] = phi * spread_true[i-1] + sigma_eta * np.random.randn()

    # Build cointegrated pair
    xag_log = np.log(18) + common_factor
    hedge_ratio_true = 1.05
    xau_log = np.log(1500) + hedge_ratio_true * common_factor + spread_true

    xau = np.exp(xau_log)
    xag = np.exp(xag_log)

    xau_s = pd.Series(xau, index=dates)
    xag_s = pd.Series(xag, index=dates)

    print("=" * 60)
    print("Bài tập 2 — Pairs Trading XAU-XAG")
    print("=" * 60)

    # Test cointegration
    print("\n--- Cointegration Test (manual Engle-Granger) ---")
    test_result = cointegration_test(xau_s, xag_s)
    print(f"  Cointegrated:    {test_result['cointegrated']}")
    print(f"  AR(1) coef:      {test_result['ar1_coefficient']:.4f}")
    print(f"  Half-life:       {test_result['half_life']:.1f} bars")
    print(f"  Hedge ratio:     {test_result['hedge_ratio']:.3f}")
    print(f"  Spread std:      {test_result['spread_std']:.2f}")

    if not test_result["cointegrated"]:
        print("\n⚠ Not cointegrated. Skipping strategy backtest.")
    else:
        # Run strategy
        print("\n--- Strategy Backtest ---")
        result = pairs_strategy(
            xau_s, xag_s,
            hedge_ratio=test_result["hedge_ratio"],
            lookback=60, z_entry=2.0, z_exit=0.5,
        )

        n_trades = int((result["position"].diff().abs() > 0).sum() / 2)
        cumulative = result["pnl"].cumsum()
        sharpe = (
            result["pnl"].mean() / result["pnl"].std()
            * np.sqrt(252)
            if result["pnl"].std() > 0 else 0
        )

        print(f"  Total trades:    {n_trades}")
        print(f"  Sharpe:          {sharpe:.3f}")
        print(f"  Final spread $:  {cumulative.iloc[-1]:.2f}")
        print(f"  Max DD spread:   "
              f"{(cumulative - cumulative.cummax()).min():.2f}")

    print("\nLessons:")
    print("  - Engle-Granger test cho 2 series, Johansen cho 3+ series")
    print("  - Hedge ratio từ OLS regression — XAU per unit XAG")
    print("  - Z-score lookback nên ≥ 2× half-life của spread")
    print("  - Stop loss Z=4 — vượt là cointegration broken, exit")
