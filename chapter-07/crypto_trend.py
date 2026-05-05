"""
QuantCFD — Chương 7
Crypto Trend Strategy — vol-targeted với weekend handling

Reference: Ch7.11.3.
- BTCUSD/ETHUSD only (high liquidity)
- Vol targeting (essential cho crypto extreme vol)
- Weekend cost buffer
- Funding rate awareness
"""

import numpy as np
import pandas as pd


def crypto_trend_strategy(df: pd.DataFrame, fast: int = 20, slow: int = 50,
                          atr_period: int = 14, target_vol: float = 0.10,
                          weekend_mult: float = 2.0,
                          long_funding_drag: float = 0.22,
                          short_funding_boost: float = 0.11,
                          base_cost: float = 0.0015) -> pd.DataFrame:
    """
    Crypto-specific trend strategy.
    
    Parameters
    ----------
    target_vol : float
        Annual target vol of strategy returns (default 10%).
    weekend_mult : float
        Cost multiplier for trades during weekend (default 2x).
    long_funding_drag : float
        Annual funding cost for perpetual long (default 22% bull market).
    short_funding_boost : float
        Annual funding income for perpetual short (default 11%).
    base_cost : float
        Round-trip transaction cost as fraction (default 0.15%).
    
    Returns
    -------
    pd.DataFrame with columns:
        ma_fast, ma_slow, signal, leverage, position,
        gross_return, funding_drag, tx_cost, net_return, equity.
    """
    out = df.copy()
    
    # MA signals
    out["ma_fast"] = out["close"].ewm(span=fast, adjust=False).mean()
    out["ma_slow"] = out["close"].ewm(span=slow, adjust=False).mean()
    out["signal"] = (out["ma_fast"] > out["ma_slow"]).astype(int).shift(1)
    
    # Realized vol for vol targeting
    out["asset_return"] = out["close"].pct_change()
    out["realized_vol"] = out["asset_return"].rolling(30).std() * np.sqrt(365)
    
    # Vol-targeted leverage (crypto uses 365-day year)
    out["leverage"] = (target_vol / out["realized_vol"].clip(0.20, 2.0))
    out["leverage"] = out["leverage"].clip(0.1, 3.0)
    
    # Weekend filter (close positions Friday close, reopen Monday)
    out["dow"] = out.index.dayofweek
    out["is_weekend"] = ((out["dow"] == 5) | (out["dow"] == 6)).astype(int)
    out["effective_signal"] = out["signal"] * (1 - out["is_weekend"])
    
    # Position
    out["position"] = out["effective_signal"] * out["leverage"]
    
    # Returns
    out["gross_return"] = out["position"] * out["asset_return"]
    
    # Funding cost
    daily_long_drag = long_funding_drag / 365
    daily_short_boost = short_funding_boost / 365
    out["funding_drag"] = np.where(
        out["position"] > 0, -daily_long_drag * out["position"].abs(),
        np.where(out["position"] < 0, +daily_short_boost * out["position"].abs(), 0)
    )
    
    # Transaction cost when position changes
    out["pos_change"] = out["position"].diff().abs().fillna(0)
    cost_multiplier = np.where(out["is_weekend"] == 1, weekend_mult, 1.0)
    out["tx_cost"] = out["pos_change"] * base_cost * cost_multiplier
    
    # Net return
    out["net_return"] = out["gross_return"] + out["funding_drag"] - out["tx_cost"]
    
    # Equity curve
    out["equity"] = (1 + out["net_return"]).cumprod()
    
    return out


def naive_trend_strategy(df: pd.DataFrame, fast: int = 20, slow: int = 50,
                         cost: float = 0.0015) -> pd.DataFrame:
    """Naive trend strategy without crypto-specific adjustments (for comparison)."""
    out = df.copy()
    out["ma_fast"] = out["close"].ewm(span=fast, adjust=False).mean()
    out["ma_slow"] = out["close"].ewm(span=slow, adjust=False).mean()
    out["signal"] = (out["ma_fast"] > out["ma_slow"]).astype(int).shift(1)
    out["asset_return"] = out["close"].pct_change()
    out["pos_change"] = out["signal"].diff().abs().fillna(0)
    out["net_return"] = out["signal"] * out["asset_return"] - out["pos_change"] * cost
    out["equity"] = (1 + out["net_return"]).cumprod()
    return out


def compute_metrics(equity: pd.Series, periods_per_year: int = 365) -> dict:
    """Compute Sharpe, CAGR, MaxDD from equity curve."""
    returns = equity.pct_change().dropna()
    if len(returns) < 30 or returns.std() == 0:
        return {"sharpe": np.nan, "cagr": np.nan, "max_dd": np.nan,
                "calmar": np.nan, "total_return": 0}
    
    sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)
    n_years = len(returns) / periods_per_year
    cagr = equity.iloc[-1] ** (1 / n_years) - 1 if equity.iloc[-1] > 0 else np.nan
    max_dd = (equity / equity.cummax() - 1).min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.inf
    total_return = equity.iloc[-1] - 1
    
    return {
        "sharpe": sharpe, "cagr": cagr, "max_dd": max_dd,
        "calmar": calmar, "total_return": total_return,
    }


if __name__ == "__main__":
    # Synthetic BTCUSD-like data
    np.random.seed(42)
    dates = pd.date_range("2018-01-01", "2024-12-31", freq="D")
    
    # Bull-bear-recovery cycle
    n = len(dates)
    regimes = [(0, n//4, 0.0010, 0.030),    # bull
               (n//4, n//2, -0.0008, 0.040),  # crash
               (n//2, 3*n//4, 0.0002, 0.025), # winter
               (3*n//4, n, 0.0008, 0.025)]    # recovery
    returns = np.zeros(n)
    for s, e, drift, vol in regimes:
        returns[s:e] = np.random.randn(e-s) * vol + drift
    prices = 10000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({"close": prices}, index=dates)
    
    print("=" * 60)
    print("Crypto Trend Strategy — BTC simulation")
    print("=" * 60)
    
    # Buy and hold
    bh_equity = df["close"] / df["close"].iloc[0]
    bh_metrics = compute_metrics(bh_equity, periods_per_year=365)
    
    # Naive trend
    naive_df = naive_trend_strategy(df)
    naive_metrics = compute_metrics(naive_df["equity"], periods_per_year=365)
    
    # Crypto-specific trend
    crypto_df = crypto_trend_strategy(df)
    crypto_metrics = compute_metrics(crypto_df["equity"], periods_per_year=365)
    
    print(f"\n{'Strategy':<25} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'Calmar':>8}")
    print("-" * 60)
    
    for name, m in [("Buy & hold BTC", bh_metrics),
                    ("Naive trend", naive_metrics),
                    ("Crypto-specific trend", crypto_metrics)]:
        print(f"{name:<25} {m['cagr']*100:>7.2f}% {m['sharpe']:>8.3f} "
              f"{m['max_dd']*100:>7.1f}% {m['calmar']:>8.2f}")
    
    print(f"\nKey insight: vol targeting + weekend awareness improves Calmar")
    print(f"from {naive_metrics['calmar']:.2f} → {crypto_metrics['calmar']:.2f}")
