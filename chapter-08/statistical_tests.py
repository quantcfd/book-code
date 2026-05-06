"""
QuantCFD — Chương 8.5.5
Statistical tests for mean reversion validation

3 essential tests trước khi deploy MR strategy:
1. Hurst exponent — < 0.5 = mean-reverting
2. Augmented Dickey-Fuller (ADF) — formal stationarity test
3. Half-life — calibrate strategy params
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def hurst_exponent(price_series: pd.Series, lags=None) -> float:
    """
    Compute Hurst exponent via R/S analysis (rescaled range).

    Interpretation:
    - H = 0.5: random walk (Brownian motion)
    - H < 0.5: mean-reverting (anti-persistent)
    - H > 0.5: trending (persistent)

    Args:
        price_series: Time series of prices.
        lags: Range of lags to test (default 2-100).

    Returns:
        Hurst exponent (float between 0 and 1).
    """
    if lags is None:
        lags = range(2, 100)

    series_clean = price_series.dropna()
    if len(series_clean) < max(lags) + 10:
        return np.nan

    log_returns = np.log(series_clean).diff().dropna()
    tau = []
    for lag in lags:
        d = log_returns.rolling(lag).sum().dropna()
        if len(d) < 2:
            return np.nan
        tau.append(np.std(d.values))

    log_lags = np.log(list(lags))
    log_tau = np.log(tau)
    if not np.all(np.isfinite(log_tau)):
        return np.nan
    H = np.polyfit(log_lags, log_tau, 1)[0]
    return H


def adf_test(series: pd.Series, sig_level: float = 0.05) -> dict:
    """
    Augmented Dickey-Fuller test for stationarity / mean reversion.

    H0 (null): unit root present → non-stationary → NOT mean-reverting
    H1: no unit root → stationary → mean-reverting
    Reject H0 if p_value < sig_level.

    Args:
        series: Time series to test.
        sig_level: Significance level (default 0.05).

    Returns:
        Dict with test_stat, p_value, critical_values, conclusion.
    """
    series_clean = series.dropna()
    if len(series_clean) < 100:
        return {"error": "insufficient data (need 100+ points)"}

    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series_clean, autolag="AIC")
        test_stat, p_value = result[0], result[1]
        critical_values = result[4]
        is_stationary = p_value < sig_level

        return {
            "test_stat": test_stat,
            "p_value": p_value,
            "critical_1pct": critical_values["1%"],
            "critical_5pct": critical_values["5%"],
            "critical_10pct": critical_values["10%"],
            "is_stationary": is_stationary,
            "conclusion": ("MEAN-REVERTING" if is_stationary
                           else "RANDOM-WALK / TRENDING"),
            "method": "statsmodels.adfuller",
        }
    except ImportError:
        # Fallback: AR(1) coefficient based heuristic
        x = series_clean.values
        x_lag = x[:-1]
        x_curr = x[1:]
        var_lag = np.var(x_lag)
        if var_lag <= 0:
            phi = 1.0
        else:
            phi = np.cov(x_lag, x_curr)[0, 1] / var_lag

        # Heuristic: phi < 0.95 with finite half-life → "stationary"
        is_stationary = (0 < phi < 0.95)
        # Approximate p-value (rough): closer phi to 1 = higher p
        p_approx = max(0.001, min(1.0, phi))

        return {
            "test_stat": phi,
            "p_value": p_approx,
            "is_stationary": is_stationary,
            "conclusion": ("MEAN-REVERTING (AR1-based)" if is_stationary
                           else "RANDOM-WALK / TRENDING"),
            "method": "AR(1) fallback (statsmodels not available)",
        }


def compute_half_life(series: pd.Series) -> float:
    """
    Half-life of mean reversion via OLS regression.

    Model: dx_t = -lambda * (x_t - mean) + noise
    half_life = ln(2) / lambda

    Args:
        series: Mean-reverting series (e.g. spread, Z-score).

    Returns:
        Half-life in bars. inf if not mean-reverting.
    """
    series_clean = series.dropna()
    if len(series_clean) < 30:
        return np.nan

    lagged = series_clean.shift(1).dropna()
    delta = (series_clean - lagged).dropna()
    lagged = lagged.loc[delta.index]

    mean_val = lagged.mean()
    lagged_centered = (lagged - mean_val).values
    delta_values = delta.values

    # OLS: delta = -lambda * lagged_centered (no intercept)
    if np.sum(lagged_centered ** 2) <= 0:
        return np.inf

    lam = -np.sum(lagged_centered * delta_values) / np.sum(lagged_centered ** 2)

    if lam <= 0:
        return np.inf  # not mean-reverting

    half_life = np.log(2) / lam
    return half_life


def mr_validation_report(series: pd.Series, name: str = "series") -> dict:
    """
    Run all 3 tests and produce verdict.

    Args:
        series: Series to validate.
        name: Display name.

    Returns:
        Dict with test results and overall verdict.
    """
    H = hurst_exponent(series)
    adf = adf_test(series)
    hl = compute_half_life(series)

    # Determine verdict
    pass_hurst = (not np.isnan(H)) and (H < 0.5)
    pass_adf = adf.get("is_stationary", False)
    valid_hl = (not np.isnan(hl)) and (3 < hl < 50)

    pass_count = sum([pass_hurst, pass_adf, valid_hl])

    if pass_count == 3:
        verdict = "STRONG_MR — deploy with confidence"
    elif pass_count == 2:
        verdict = "MARGINAL_MR — proceed with caution + filters"
    else:
        verdict = "NOT_MR — do not deploy MR strategy here"

    return {
        "name": name,
        "hurst": H,
        "hurst_pass": pass_hurst,
        "adf_p_value": adf.get("p_value"),
        "adf_pass": pass_adf,
        "half_life_bars": hl,
        "half_life_valid": valid_hl,
        "tests_passed": f"{pass_count}/3",
        "verdict": verdict,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Mean Reversion Validation Tests")
    print("=" * 70)

    np.random.seed(42)
    n = 1000

    # Test 1: True mean-reverting (Ornstein-Uhlenbeck process)
    mu, theta, sigma = 0.0, 0.1, 1.0
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i-1] + theta * (mu - x[i-1]) + sigma * np.random.randn()
    mr_series = pd.Series(x, index=pd.date_range("2020-01-01", periods=n, freq="D"))

    # Test 2: Random walk
    rw = pd.Series(np.cumsum(np.random.randn(n)),
                   index=pd.date_range("2020-01-01", periods=n, freq="D"))

    # Test 3: Trending (geometric Brownian motion with drift)
    drift = 0.001
    trend = pd.Series(np.cumsum(np.random.randn(n) * 0.01 + drift),
                      index=pd.date_range("2020-01-01", periods=n, freq="D"))
    trend_prices = 100 * np.exp(trend)

    test_cases = [
        ("Mean-reverting (OU)", mr_series),
        ("Random walk", rw),
        ("Trending (drift)", pd.Series(trend_prices.values, index=trend.index)),
    ]

    for name, series in test_cases:
        print(f"\n{'─' * 70}")
        print(f"Series: {name}")
        print(f"{'─' * 70}")
        report = mr_validation_report(series, name)
        print(f"  Hurst:        {report['hurst']:.3f}  "
              f"({'✓' if report['hurst_pass'] else '✗'} pass)")
        if report['adf_p_value'] is not None:
            print(f"  ADF p-value:  {report['adf_p_value']:.4f}  "
                  f"({'✓' if report['adf_pass'] else '✗'} pass)")
        if not np.isnan(report['half_life_bars']) and np.isfinite(report['half_life_bars']):
            print(f"  Half-life:    {report['half_life_bars']:.1f} bars  "
                  f"({'✓' if report['half_life_valid'] else '✗'} valid)")
        else:
            print(f"  Half-life:    inf/nan (not MR)")
        print(f"  Tests passed: {report['tests_passed']}")
        print(f"  Verdict:      {report['verdict']}")
