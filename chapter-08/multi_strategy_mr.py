"""
QuantCFD — Chương 8.9 + 8.10
Multi-Strategy MR Portfolio

Combine 4 MR strategies trên 1 portfolio:
- Bollinger Bands (single asset volatility-based)
- RSI extreme (sentiment-based)
- Z-score (statistical-based)
- Pairs trading (cointegration-based)

Diversification: nếu 1 strategy DD trong crisis, others có thể OK.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def bb_strategy_returns(
    df: pd.DataFrame, period: int = 20, std_mult: float = 2.0,
    cost: float = 0.0008,
) -> pd.Series:
    """Bollinger Bands MR strategy returns."""
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    upper = (sma + std_mult * std).shift(1)
    lower = (sma - std_mult * std).shift(1)
    mid = sma.shift(1)

    position = 0
    positions = []
    for i in range(len(df)):
        if pd.isna(lower.iloc[i]):
            positions.append(0)
            continue
        price = df["close"].iloc[i]
        if position == 0:
            if price < lower.iloc[i]:
                position = 1
            elif price > upper.iloc[i]:
                position = -1
        elif position == 1:
            if price >= mid.iloc[i]:
                position = 0
        elif position == -1:
            if price <= mid.iloc[i]:
                position = 0
        positions.append(position)

    pos_series = pd.Series(positions, index=df.index)
    returns = df["close"].pct_change()
    pos_change = pos_series.diff().abs().fillna(0)
    return pos_series * returns - pos_change * cost


def rsi_strategy_returns(
    df: pd.DataFrame, period: int = 14, oversold: float = 30,
    overbought: float = 70, cost: float = 0.0008,
) -> pd.Series:
    """RSI extreme MR strategy returns."""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi_shifted = rsi.shift(1)

    position = 0
    positions = []
    for i in range(len(df)):
        r = rsi_shifted.iloc[i]
        if pd.isna(r):
            positions.append(0)
            continue
        if position == 0:
            if r < oversold:
                position = 1
            elif r > overbought:
                position = -1
        elif position == 1:
            if r > 50:
                position = 0
        elif position == -1:
            if r < 50:
                position = 0
        positions.append(position)

    pos_series = pd.Series(positions, index=df.index)
    returns = df["close"].pct_change()
    pos_change = pos_series.diff().abs().fillna(0)
    return pos_series * returns - pos_change * cost


def zscore_strategy_returns(
    df: pd.DataFrame, lookback: int = 20, z_entry: float = 2.0,
    z_exit: float = 0.5, cost: float = 0.0008,
) -> pd.Series:
    """Z-score MR strategy returns."""
    sma = df["close"].rolling(lookback).mean()
    std = df["close"].rolling(lookback).std()
    z = ((df["close"] - sma) / std).shift(1)

    position = 0
    positions = []
    for i in range(len(df)):
        zs = z.iloc[i]
        if pd.isna(zs):
            positions.append(0)
            continue
        if position == 0:
            if zs < -z_entry:
                position = 1
            elif zs > z_entry:
                position = -1
        elif position == 1:
            if zs >= -z_exit:
                position = 0
        elif position == -1:
            if zs <= z_exit:
                position = 0
        positions.append(position)

    pos_series = pd.Series(positions, index=df.index)
    returns = df["close"].pct_change()
    pos_change = pos_series.diff().abs().fillna(0)
    return pos_series * returns - pos_change * cost


def pairs_strategy_returns(
    a: pd.Series, b: pd.Series, lookback: int = 60,
    z_entry: float = 2.0, z_exit: float = 0.5,
    cost: float = 0.0010,  # 2 legs = 2x cost
) -> pd.Series:
    """Pairs trading on log ratio MR."""
    common = a.dropna().index.intersection(b.dropna().index)
    a_clean = a.loc[common]
    b_clean = b.loc[common]
    log_ratio = np.log(a_clean / b_clean)

    z = ((log_ratio - log_ratio.rolling(lookback).mean()) /
         log_ratio.rolling(lookback).std()).shift(1)

    position = 0
    positions = []
    for i in range(len(log_ratio)):
        zs = z.iloc[i]
        if pd.isna(zs):
            positions.append(0)
            continue
        if position == 0:
            if zs < -z_entry:
                position = 1
            elif zs > z_entry:
                position = -1
        elif position == 1:
            if zs >= -z_exit:
                position = 0
        elif position == -1:
            if zs <= z_exit:
                position = 0
        positions.append(position)

    pos_series = pd.Series(positions, index=common)
    ratio_change = log_ratio.diff()
    pos_change = pos_series.diff().abs().fillna(0)
    return pos_series.shift(1) * ratio_change - pos_change * cost


def combine_mr_strategies(
    strategy_returns: dict,
    weight_method: str = "equal",
    rolling_vol_lookback: int = 60,
) -> dict:
    """
    Combine MR strategies into portfolio.

    Args:
        strategy_returns: Dict {name: pd.Series of strategy returns}.
        weight_method: 'equal' or 'inv_vol' (inverse volatility weighting).

    Returns:
        Dict with portfolio_returns, weights_history, sharpe, cagr, max_dd,
        per_strategy_metrics.
    """
    df = pd.DataFrame(strategy_returns).dropna(how="all").fillna(0)

    if weight_method == "equal":
        n = len(df.columns)
        weights = pd.DataFrame(1/n, index=df.index, columns=df.columns)
    elif weight_method == "inv_vol":
        rolling_vol = df.rolling(rolling_vol_lookback).std()
        inv_vol = 1 / rolling_vol.replace(0, np.nan)
        weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0)
    else:
        raise ValueError(f"Unknown weight_method: {weight_method}")

    portfolio_returns = (df * weights).sum(axis=1)

    # Per-strategy metrics
    per_strategy = {}
    for name in df.columns:
        s = df[name].dropna()
        if len(s) > 30 and s.std() > 0:
            per_strategy[name] = {
                "sharpe": (s.mean() / s.std()) * np.sqrt(252),
                "cagr": (1 + s.mean()) ** 252 - 1,
                "max_dd": ((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min(),
            }
        else:
            per_strategy[name] = {"sharpe": np.nan, "cagr": np.nan, "max_dd": np.nan}

    # Portfolio metrics
    pr = portfolio_returns.dropna()
    if len(pr) > 30 and pr.std() > 0:
        portfolio_sharpe = (pr.mean() / pr.std()) * np.sqrt(252)
        portfolio_cagr = (1 + pr.mean()) ** 252 - 1
        equity = (1 + pr).cumprod()
        portfolio_max_dd = (equity / equity.cummax() - 1).min()
    else:
        portfolio_sharpe = portfolio_cagr = portfolio_max_dd = np.nan
        equity = None

    # Correlation matrix
    corr = df.corr()

    return {
        "portfolio_returns": portfolio_returns,
        "portfolio_sharpe": portfolio_sharpe,
        "portfolio_cagr": portfolio_cagr,
        "portfolio_max_dd": portfolio_max_dd,
        "per_strategy": per_strategy,
        "correlation_matrix": corr,
        "equity_curve": equity,
        "weights_history": weights,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Multi-Strategy MR Portfolio")
    print("=" * 70)

    np.random.seed(42)
    dates = pd.date_range("2018-01-01", "2024-12-31", freq="D")
    n = len(dates)

    # Synthetic XAUUSD-like with regimes
    returns = np.zeros(n)
    for i in range(0, n, 252):
        end = min(i + 252, n)
        drift = np.random.choice([-0.0002, 0, 0.0003])
        returns[i:end] = np.random.randn(end - i) * 0.012 + drift
    prices = 1500 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({"close": prices}, index=dates)

    # Synthetic XAGUSD (correlated)
    returns_b = returns + np.random.randn(n) * 0.003
    prices_b = 18 * np.exp(np.cumsum(returns_b))
    df_b = pd.DataFrame({"close": prices_b}, index=dates)

    # Run 4 strategies
    print("\nRunning 4 MR strategies...")
    bb_ret = bb_strategy_returns(df)
    rsi_ret = rsi_strategy_returns(df)
    z_ret = zscore_strategy_returns(df)
    pairs_ret = pairs_strategy_returns(df["close"], df_b["close"])

    strategies = {
        "BB(20,2)": bb_ret,
        "RSI(14)": rsi_ret,
        "Z-score(20)": z_ret,
        "Pairs XAU-XAG": pairs_ret,
    }

    # Combine — equal weight
    print("\n--- Equal-weight portfolio ---")
    result_eq = combine_mr_strategies(strategies, weight_method="equal")

    print("\nPer-strategy metrics:")
    for name, m in result_eq["per_strategy"].items():
        print(f"  {name:18s}: Sharpe={m['sharpe']:6.3f}  "
              f"CAGR={m['cagr']*100:6.2f}%  DD={m['max_dd']*100:6.2f}%")

    print(f"\nPortfolio (equal-weight):")
    print(f"  Sharpe:   {result_eq['portfolio_sharpe']:.3f}")
    print(f"  CAGR:     {result_eq['portfolio_cagr']*100:.2f}%")
    print(f"  Max DD:   {result_eq['portfolio_max_dd']*100:.2f}%")

    avg_single = np.mean([
        m["sharpe"] for m in result_eq["per_strategy"].values()
        if not np.isnan(m["sharpe"])
    ])
    print(f"\n  Avg single-strategy Sharpe: {avg_single:.3f}")
    print(f"  Diversification benefit:    {result_eq['portfolio_sharpe'] - avg_single:+.3f}")

    # Inverse-vol weighting
    print("\n--- Inverse-volatility weighted portfolio ---")
    result_iv = combine_mr_strategies(strategies, weight_method="inv_vol")
    print(f"  Sharpe:   {result_iv['portfolio_sharpe']:.3f}")
    print(f"  CAGR:     {result_iv['portfolio_cagr']*100:.2f}%")
    print(f"  Max DD:   {result_iv['portfolio_max_dd']*100:.2f}%")

    print(f"\nCorrelation matrix:")
    print(result_eq["correlation_matrix"].round(2))
    print(f"\nIdeal MR portfolio: avg correlation < 0.3 across strategies.")
