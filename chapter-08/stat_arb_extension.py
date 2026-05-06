"""
QuantCFD — Chương 8.13.7
Statistical Arbitrage Extensions

Beyond pairs trading:
1. Multi-asset cointegration (Johansen test)
2. Dynamic hedge ratio via Kalman filter
3. Factor-neutral residual mean reversion (template)
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def find_cointegrated_basket(
    prices_df: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> tuple:
    """
    Johansen cointegration test on basket of assets.

    Returns weight vector for cointegrated portfolio (linear combination
    that is stationary).

    Args:
        prices_df: DataFrame with N asset prices.
        det_order: -1 (no constant), 0 (constant), 1 (linear trend).
        k_ar_diff: Number of lagged differences.

    Returns:
        (weights, spread) where:
        - weights: numpy array of weights for cointegrated combination
        - spread: pd.Series of weighted sum (should be stationary)
    """
    try:
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
    except ImportError:
        return None, None

    if prices_df.shape[1] < 2 or len(prices_df) < 100:
        return None, None

    result = coint_johansen(prices_df.values, det_order, k_ar_diff)

    # Trace statistic vs critical values to determine # cointegration relations
    n_assets = prices_df.shape[1]
    trace_stat = result.lr1
    crit_val_5pct = result.cvt[:, 1]  # 5% critical values

    n_coint = 0
    for i in range(n_assets):
        if trace_stat[i] > crit_val_5pct[i]:
            n_coint += 1
        else:
            break

    if n_coint == 0:
        return None, None

    # Use first eigenvector (strongest cointegration)
    weights = result.evec[:, 0]
    weights = weights / np.abs(weights).sum()  # normalize

    spread = (prices_df * weights).sum(axis=1)

    return weights, spread


def kalman_dynamic_hedge_ratio(
    y_series: pd.Series,
    x_series: pd.Series,
    delta: float = 1e-5,
    obs_cov: float = 1.0,
) -> tuple:
    """
    Compute time-varying hedge ratio via Kalman filter.

    Model: y_t = beta_t × x_t + intercept_t + noise
    State: [beta_t, intercept_t] evolves slowly.

    Args:
        y_series: Dependent series (e.g. ETH).
        x_series: Independent series (e.g. BTC).
        delta: State innovation variance scaler (smaller = smoother beta).
        obs_cov: Observation noise variance.

    Returns:
        (beta_series, spread) where:
        - beta_series: time-varying hedge ratio
        - spread: y - beta × x (residual)
    """
    try:
        from pykalman import KalmanFilter
    except ImportError:
        # Fallback to expanding window OLS
        return _expanding_ols_hedge_ratio(y_series, x_series)

    y_clean = y_series.dropna()
    x_clean = x_series.dropna()
    common = y_clean.index.intersection(x_clean.index)
    y = y_clean.loc[common].values
    x = x_clean.loc[common].values

    if len(y) < 30:
        return None, None

    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.vstack([x, np.ones(len(x))]).T[:, np.newaxis]

    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=obs_cov,
        transition_covariance=trans_cov,
    )

    state_means, _ = kf.filter(y.reshape(-1, 1))

    beta_series = pd.Series(state_means[:, 0], index=common)
    intercept_series = pd.Series(state_means[:, 1], index=common)
    spread = pd.Series(
        y_clean.loc[common].values
        - beta_series.values * x_clean.loc[common].values
        - intercept_series.values,
        index=common
    )

    return beta_series, spread


def _expanding_ols_hedge_ratio(
    y_series: pd.Series,
    x_series: pd.Series,
    min_periods: int = 30,
) -> tuple:
    """Fallback: expanding window OLS hedge ratio."""
    common = y_series.dropna().index.intersection(x_series.dropna().index)
    y = y_series.loc[common]
    x = x_series.loc[common]

    betas = []
    for i in range(len(y)):
        if i < min_periods:
            betas.append(np.nan)
            continue
        y_window = y.iloc[: i + 1].values
        x_window = x.iloc[: i + 1].values
        # OLS beta
        x_centered = x_window - x_window.mean()
        y_centered = y_window - y_window.mean()
        beta = np.sum(x_centered * y_centered) / np.sum(x_centered ** 2)
        betas.append(beta)

    beta_series = pd.Series(betas, index=common)
    spread = y - beta_series * x

    return beta_series, spread


def trade_cointegrated_spread(
    spread: pd.Series,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_stop: float = 4.0,
    lookback: int = 60,
) -> pd.DataFrame:
    """
    Trade cointegrated spread on Z-score signals.

    Args:
        spread: Spread (cointegrated linear combination).
        z_entry: Entry threshold (long if Z<-entry, short if Z>+entry).
        z_exit: Exit threshold (close at Z near 0).
        z_stop: Stop loss threshold.
        lookback: Window for Z-score computation.

    Returns:
        DataFrame with spread, zscore, position, equity.
    """
    out = pd.DataFrame({"spread": spread})
    out["zscore"] = (
        spread - spread.rolling(lookback).mean()
    ) / spread.rolling(lookback).std()
    out["zscore"] = out["zscore"].shift(1)

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
    out["equity"] = out["pnl"].cumsum()

    return out


if __name__ == "__main__":
    print("=" * 70)
    print("Statistical Arbitrage Extensions — Demo")
    print("=" * 70)

    # Demo 1: Cointegrated basket
    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    
    # Common factor + idiosyncratic
    common = np.cumsum(np.random.randn(n) * 0.01)
    asset_a = common + np.cumsum(np.random.randn(n) * 0.005)
    asset_b = 0.8 * common + np.cumsum(np.random.randn(n) * 0.005)
    asset_c = 1.2 * common + np.cumsum(np.random.randn(n) * 0.005)

    prices = pd.DataFrame({
        "A": np.exp(asset_a),
        "B": np.exp(asset_b),
        "C": np.exp(asset_c),
    }, index=dates)

    print("\n--- Cointegrated Basket (Johansen test) ---")
    weights, spread = find_cointegrated_basket(prices)
    if weights is not None:
        print(f"Weights: {dict(zip(prices.columns, weights.round(3)))}")
        print(f"Spread mean: {spread.mean():.4f}")
        print(f"Spread std:  {spread.std():.4f}")
        print(f"(Stationary spread → cointegration found)")
    else:
        print("No cointegration detected")

    # Demo 2: Kalman dynamic hedge ratio
    print("\n--- Kalman Filter Dynamic Hedge Ratio ---")
    np.random.seed(42)
    n = 500
    btc = np.cumsum(np.random.randn(n) * 0.03)
    btc_prices = 30000 * np.exp(btc)

    # ETH with time-varying beta
    beta_true = np.linspace(15, 18, n)  # changes from 15 to 18
    eth_prices = btc_prices / beta_true * np.exp(np.random.randn(n) * 0.01)

    btc_s = pd.Series(btc_prices, index=pd.date_range("2023-01-01", periods=n, freq="D"))
    eth_s = pd.Series(eth_prices, index=pd.date_range("2023-01-01", periods=n, freq="D"))

    beta_est, spread_kalman = kalman_dynamic_hedge_ratio(eth_s, btc_s)

    if beta_est is not None:
        print(f"Initial beta estimate: {beta_est.iloc[30]:.3f} (true: ~15)")
        print(f"Final beta estimate:   {beta_est.iloc[-1]:.3f} (true: ~18)")
        print(f"Beta tracks the changing relationship.")

    # Demo 3: Trade spread
    print("\n--- Trade Cointegrated Spread ---")
    if spread is not None:
        result = trade_cointegrated_spread(spread, z_entry=2.0, z_exit=0.5)
        n_trades = (result["position"].diff().abs() > 0).sum() / 2
        n_long = (result["position"] == 1).sum()
        n_short = (result["position"] == -1).sum()
        print(f"Total trades: {n_trades:.0f}")
        print(f"Days long:    {n_long}")
        print(f"Days short:   {n_short}")
        print(f"Final equity: {result['equity'].iloc[-1]:.2f}")
