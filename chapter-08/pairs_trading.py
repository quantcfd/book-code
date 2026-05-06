"""
Pairs trading strategy with cointegration.
QuantCFD Chapter 8.6
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def test_cointegration(
    series_a: pd.Series,
    series_b: pd.Series,
    significance: float = 0.05,
) -> dict:
    """
    Test cointegration via simplified Engle-Granger.

    Method:
    1. OLS A ~ B → residuals (spread)
    2. Compute AR(1) of spread → if phi < 0.95 with finite half-life, cointegrated
    
    For rigorous test, use statsmodels.tsa.stattools.coint when available.

    Returns dict with:
        test_statistic: AR(1) coefficient
        p_value: approximate (1 - phi)
        cointegrated: boolean
    """
    a = pd.Series(series_a).dropna()
    b = pd.Series(series_b).dropna()
    common_idx = a.index.intersection(b.index)
    a = a.loc[common_idx].values
    b = b.loc[common_idx].values

    if len(a) < 50:
        return {
            "test_statistic": np.nan,
            "p_value": 1.0,
            "cointegrated": False,
        }

    # OLS hedge ratio
    b_centered = b - b.mean()
    a_centered = a - a.mean()
    var_b = np.sum(b_centered ** 2)
    if var_b <= 0:
        return {
            "test_statistic": np.nan, "p_value": 1.0, "cointegrated": False,
        }
    beta = np.sum(a_centered * b_centered) / var_b
    intercept = a.mean() - beta * b.mean()
    spread = a - beta * b - intercept

    # AR(1) of spread
    spread_lag = spread[:-1]
    spread_curr = spread[1:]
    var_lag = np.var(spread_lag)
    if var_lag <= 0:
        return {
            "test_statistic": 1.0, "p_value": 1.0, "cointegrated": False,
        }
    phi = np.cov(spread_lag, spread_curr)[0, 1] / var_lag

    cointegrated = (0 < phi < 0.95)
    p_approx = max(0.0, min(1.0, phi))  # rough proxy

    return {
        "test_statistic": phi,
        "p_value": p_approx,
        "cointegrated": cointegrated,
    }


def compute_hedge_ratio(series_a: pd.Series, series_b: pd.Series) -> float:
    """
    Compute hedge ratio via OLS regression.
    A_t = α + β·B_t + ε
    """
    common_idx = series_a.index.intersection(series_b.index)
    a = series_a.loc[common_idx].values
    b = series_b.loc[common_idx].values

    b_centered = b - b.mean()
    a_centered = a - a.mean()
    var_b = np.sum(b_centered ** 2)
    if var_b <= 0:
        return 1.0
    beta = np.sum(a_centered * b_centered) / var_b
    return float(beta)


def pairs_trading_signals(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    lookback: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.DataFrame:
    """
    Pairs trading strategy with rolling hedge ratio and Z-score of spread.

    Returns DataFrame with:
        spread, hedge_ratio, zscore, spread_signal, pos_a, pos_b

    Long spread (signal=1): long A, short hedge_ratio*B
    Short spread (signal=-1): short A, long hedge_ratio*B
    """
    df = pd.DataFrame({
        "price_a": df_a["close"],
        "price_b": df_b["close"],
    }).dropna()

    # Simple hedge ratio: ratio of std (a proxy when correlation ~ 1)
    df["hedge_ratio"] = (
        df["price_a"].rolling(lookback).corr(df["price_b"])
        * df["price_a"].rolling(lookback).std()
        / df["price_b"].rolling(lookback).std()
    )

    df["spread"] = df["price_a"] - df["hedge_ratio"] * df["price_b"]
    spread_mean = df["spread"].rolling(lookback).mean()
    spread_std = df["spread"].rolling(lookback).std()
    df["zscore"] = (df["spread"] - spread_mean) / spread_std

    n = len(df)
    position = np.zeros(n, dtype=int)
    for i in range(lookback + 1, n):
        prev_pos = position[i - 1]
        z = df["zscore"].iloc[i]
        if pd.isna(z):
            position[i] = prev_pos
            continue

        if prev_pos == 0:
            if z < -entry_z:
                position[i] = 1
            elif z > entry_z:
                position[i] = -1
        elif prev_pos == 1:
            position[i] = 0 if z >= -exit_z else 1
        elif prev_pos == -1:
            position[i] = 0 if z <= exit_z else -1

    df["spread_signal"] = position
    df["spread_signal"] = df["spread_signal"].shift(1)
    df["pos_a"] = df["spread_signal"]
    df["pos_b"] = -df["spread_signal"] * df["hedge_ratio"]
    return df


def cfd_pair_sizing(
    account_equity: float,
    risk_per_trade: float,
    spread_std: float,
    instrument_a_value_per_unit: float,
    instrument_b_value_per_unit: float,
    hedge_ratio: float,
) -> tuple[float, float]:
    """
    Compute CFD pair position size accounting for different contract specs.

    Returns: (size_a, size_b) in lots/contracts of each instrument.
    """
    risk_usd = account_equity * risk_per_trade
    # 2-sigma move on spread = stop level
    stop_distance_usd = 2 * spread_std * instrument_a_value_per_unit
    if stop_distance_usd == 0:
        return 0.0, 0.0

    size_a = risk_usd / stop_distance_usd
    size_b = size_a * hedge_ratio * (instrument_a_value_per_unit / instrument_b_value_per_unit)
    return size_a, size_b


if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    common = np.cumsum(np.random.randn(n) * 0.01)
    noise_a = np.random.randn(n) * 0.05
    noise_b = np.random.randn(n) * 0.05
    series_a = pd.Series(2000 + common + noise_a, name="A")
    series_b = pd.Series(20 + 0.01 * common + 0.1 * noise_b, name="B")

    coint_result = test_cointegration(series_a, series_b)
    print(f"Cointegration test:")
    print(f"  p-value: {coint_result['p_value']:.4f}")
    print(f"  cointegrated: {coint_result['cointegrated']}")

    hr = compute_hedge_ratio(series_a, series_b)
    print(f"\nHedge ratio: {hr:.2f}")

    df_a = pd.DataFrame({"close": series_a})
    df_b = pd.DataFrame({"close": series_b})
    df_a.index = pd.date_range("2020-01-01", periods=n, freq="D")
    df_b.index = df_a.index

    result = pairs_trading_signals(df_a, df_b, lookback=60, entry_z=2.0)
    print(f"\nPairs trading signals:")
    print(result["spread_signal"].value_counts())
