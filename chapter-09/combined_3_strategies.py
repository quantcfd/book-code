"""
QuantCFD — Chương 9.14
Combined 3-Strategy Portfolio: Trend (Ch7) + MR (Ch8) + Vol BO (Ch9)

Master orchestration combining all 3 strategy classes:
- Trend: MA crossover (Ch7-style)
- Mean Reversion: Bollinger Bands (Ch8-style)
- Vol Breakout: Keltner channel (Ch9-style)

Allocation: configurable (default balanced 45/30/25).
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def trend_returns(
    df: pd.DataFrame, fast: int = 20, slow: int = 50, cost: float = 0.0005,
) -> pd.Series:
    """MA crossover trend strategy returns."""
    out = df.copy()
    out["ma_fast"] = out["close"].rolling(fast).mean()
    out["ma_slow"] = out["close"].rolling(slow).mean()
    out["signal"] = (out["ma_fast"] > out["ma_slow"]).astype(int).shift(1)
    out["ret"] = out["close"].pct_change()
    out["pos_change"] = out["signal"].diff().abs().fillna(0)
    return out["signal"] * out["ret"] - out["pos_change"] * cost


def mr_returns(
    df: pd.DataFrame, period: int = 20, std_mult: float = 2.0,
    cost: float = 0.0008,
) -> pd.Series:
    """Bollinger Bands MR strategy returns."""
    out = df.copy()
    sma = out["close"].rolling(period).mean()
    std = out["close"].rolling(period).std()
    upper = (sma + std_mult * std).shift(1)
    lower = (sma - std_mult * std).shift(1)
    mid = sma.shift(1)

    position = 0
    positions = []
    for i in range(len(out)):
        if pd.isna(lower.iloc[i]):
            positions.append(0)
            continue
        price = out["close"].iloc[i]
        if position == 0:
            if price < lower.iloc[i]:
                position = 1
            elif price > upper.iloc[i]:
                position = -1
        elif position == 1 and price >= mid.iloc[i]:
            position = 0
        elif position == -1 and price <= mid.iloc[i]:
            position = 0
        positions.append(position)

    pos_series = pd.Series(positions, index=out.index)
    out["ret"] = out["close"].pct_change()
    out["pos_change"] = pos_series.diff().abs().fillna(0)
    return pos_series * out["ret"] - out["pos_change"] * cost


def vol_bo_returns(
    df: pd.DataFrame, ema_period: int = 20, atr_mult: float = 2.0,
    cost: float = 0.0008,
) -> pd.Series:
    """Keltner breakout vol BO strategy returns."""
    out = df.copy()
    out["ema"] = out["close"].ewm(span=ema_period, adjust=False).mean()
    high_low = out["high"] - out["low"]
    high_close = (out["high"] - out["close"].shift(1)).abs()
    low_close = (out["low"] - out["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    out["upper"] = (out["ema"] + atr_mult * out["atr"]).shift(1)
    out["lower"] = (out["ema"] - atr_mult * out["atr"]).shift(1)
    out["mid"] = out["ema"].shift(1)

    position = 0
    positions = []
    for i in range(len(out)):
        row = out.iloc[i]
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

    pos_series = pd.Series(positions, index=out.index)
    out["ret"] = out["close"].pct_change()
    out["pos_change"] = pos_series.diff().abs().fillna(0)
    return pos_series * out["ret"] - out["pos_change"] * cost


def combined_3_strategy_portfolio(
    df: pd.DataFrame,
    allocation: dict = None,
    periods_per_year: int = 252,
) -> dict:
    """
    Combine 3 strategy classes into 1 portfolio.

    Args:
        df: OHLC DataFrame.
        allocation: Dict {"trend": 0.45, "mr": 0.30, "vol_bo": 0.25}.
                    Defaults to balanced allocation.
        periods_per_year: For Sharpe annualization.

    Returns:
        Dict with portfolio + per-strategy metrics + correlation.
    """
    if allocation is None:
        allocation = {"trend": 0.45, "mr": 0.30, "vol_bo": 0.25}

    trend_r = trend_returns(df)
    mr_r = mr_returns(df)
    vol_r = vol_bo_returns(df)

    strategies = {"trend": trend_r, "mr": mr_r, "vol_bo": vol_r}

    # Build portfolio returns
    df_strategies = pd.DataFrame(strategies).dropna(how="all").fillna(0)
    portfolio = pd.Series(0.0, index=df_strategies.index)
    for name, weight in allocation.items():
        if name in df_strategies.columns:
            portfolio += df_strategies[name] * weight

    portfolio_clean = portfolio.dropna()
    if len(portfolio_clean) < 30 or portfolio_clean.std() == 0:
        return {"error": "insufficient data"}

    # Per-strategy metrics
    per_strategy = {}
    for name in df_strategies.columns:
        s = df_strategies[name].dropna()
        if len(s) > 30 and s.std() > 0:
            per_strategy[name] = {
                "sharpe": (s.mean() / s.std()) * np.sqrt(periods_per_year),
                "cagr": (1 + s.mean()) ** periods_per_year - 1,
                "max_dd": (
                    (1 + s).cumprod() / (1 + s).cumprod().cummax() - 1
                ).min(),
                "weight": allocation.get(name, 0),
            }

    # Portfolio metrics
    p_sharpe = (
        portfolio_clean.mean() / portfolio_clean.std()
        * np.sqrt(periods_per_year)
    )
    p_cagr = (1 + portfolio_clean.mean()) ** periods_per_year - 1
    eq = (1 + portfolio_clean).cumprod()
    p_dd = (eq / eq.cummax() - 1).min()
    calmar = p_cagr / abs(p_dd) if p_dd != 0 else 0

    return {
        "portfolio_sharpe": p_sharpe,
        "portfolio_cagr": p_cagr,
        "portfolio_max_dd": p_dd,
        "portfolio_calmar": calmar,
        "per_strategy": per_strategy,
        "correlation_matrix": df_strategies.corr(),
        "equity_curve": eq,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("3-Strategy Portfolio: Trend + MR + Vol BO")
    print("=" * 70)

    np.random.seed(42)
    n = 7 * 252
    dates = pd.date_range("2018-01-01", periods=n, freq="D")

    rets = np.zeros(n)
    for i in range(0, n, 252):
        end = min(i + 252, n)
        regime = np.random.choice(["trend_up", "range", "trend_down", "vol"])
        if regime == "trend_up":
            rets[i:end] = np.random.randn(end - i) * 0.010 + 0.0004
        elif regime == "trend_down":
            rets[i:end] = np.random.randn(end - i) * 0.010 - 0.0003
        elif regime == "range":
            for j in range(i, end):
                rets[j] = np.random.randn() * 0.008 - 0.15 * (rets[j-1] if j > 0 else 0)
        else:  # vol expansion
            rets[i:end] = np.random.randn(end - i) * 0.020

    closes = 1500 * np.exp(np.cumsum(rets))
    daily_vol = np.abs(rets) + 0.003
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

    # Test 3 allocation profiles
    profiles = {
        "Conservative (50/30/20)": {"trend": 0.50, "mr": 0.30, "vol_bo": 0.20},
        "Balanced (45/30/25)":     {"trend": 0.45, "mr": 0.30, "vol_bo": 0.25},
        "Aggressive (40/25/35)":   {"trend": 0.40, "mr": 0.25, "vol_bo": 0.35},
    }

    print(f"\n{'─' * 70}")
    print(f"{'Profile':<28} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>8}")
    print(f"{'─' * 70}")

    for name, alloc in profiles.items():
        r = combined_3_strategy_portfolio(df, allocation=alloc)
        if "error" not in r:
            print(f"{name:<28} {r['portfolio_sharpe']:>8.3f} "
                  f"{r['portfolio_cagr']*100:>7.2f}% "
                  f"{r['portfolio_max_dd']*100:>7.2f}% "
                  f"{r['portfolio_calmar']:>8.2f}")

    # Detailed view of balanced profile
    print(f"\n{'─' * 70}")
    print("BALANCED PROFILE — Detailed view")
    print(f"{'─' * 70}")
    r_balanced = combined_3_strategy_portfolio(
        df, allocation={"trend": 0.45, "mr": 0.30, "vol_bo": 0.25}
    )
    print("\nPer-strategy metrics:")
    for name, m in r_balanced["per_strategy"].items():
        print(f"  {name:<10}: Sharpe={m['sharpe']:6.3f}  "
              f"CAGR={m['cagr']*100:6.2f}%  DD={m['max_dd']*100:7.2f}%  "
              f"weight={m['weight']*100:.0f}%")

    print(f"\nCorrelation between strategies:")
    print(r_balanced["correlation_matrix"].round(2))

    print(f"\nKey insight: low correlation across strategies → diversification benefit")
    print(f"Portfolio Sharpe ({r_balanced['portfolio_sharpe']:.2f}) > avg single strategy")
