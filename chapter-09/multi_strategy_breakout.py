"""
QuantCFD — Chương 9.12
Multi-Strategy Vol Breakout Portfolio

Combine 4 vol breakout strategies parallel trên multiple instruments:
- ORB (intraday, indices/commodities)
- NR7 (daily, all markets)
- Keltner (FX, crypto)
- BB Squeeze (crypto)
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def keltner_returns(
    df: pd.DataFrame,
    ema_period: int = 20, atr_period: int = 14, atr_mult: float = 2.0,
    cost: float = 0.0008,
) -> pd.Series:
    """Keltner breakout strategy returns."""
    out = df.copy()
    out["ema"] = out["close"].ewm(span=ema_period, adjust=False).mean()

    high_low = out["high"] - out["low"]
    high_close = (out["high"] - out["close"].shift(1)).abs()
    low_close = (out["low"] - out["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["atr"] = tr.ewm(alpha=1 / atr_period, adjust=False).mean()
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


def nr7_returns(
    df: pd.DataFrame, lookback: int = 7, target_mult: float = 1.5,
) -> pd.Series:
    """NR7 strategy returns (treats each trade as 1 bar exposure)."""
    out = df.copy()
    out["range"] = out["high"] - out["low"]
    nr7 = (
        (out["range"] == out["range"].rolling(lookback).min())
        & (out["range"].rolling(lookback).count() == lookback)
    )

    out["setup_high"] = out["high"].shift(1).where(nr7.shift(1))
    out["setup_low"] = out["low"].shift(1).where(nr7.shift(1))
    out["setup_range"] = out["range"].shift(1).where(nr7.shift(1))

    # Per-day returns from breakout
    daily_ret = pd.Series(0.0, index=out.index)
    for idx, row in out.iterrows():
        if pd.isna(row["setup_high"]):
            continue
        if row["high"] > row["setup_high"]:
            entry = row["setup_high"]
            target = entry + target_mult * row["setup_range"]
            stop = row["setup_low"]
            if row["low"] <= stop:
                trade_pnl = (stop - entry) / entry
            elif row["high"] >= target:
                trade_pnl = (target - entry) / entry
            else:
                trade_pnl = (row["close"] - entry) / entry
            daily_ret.loc[idx] = trade_pnl
        elif row["low"] < row["setup_low"]:
            entry = row["setup_low"]
            target = entry - target_mult * row["setup_range"]
            stop = row["setup_high"]
            if row["high"] >= stop:
                trade_pnl = (entry - stop) / entry
            elif row["low"] <= target:
                trade_pnl = (entry - target) / entry
            else:
                trade_pnl = (entry - row["close"]) / entry
            daily_ret.loc[idx] = trade_pnl
    return daily_ret


def bb_squeeze_returns(
    df: pd.DataFrame,
    bb_period: int = 20, kc_mult: float = 1.5,
    cost: float = 0.0008,
    max_holding: int = 20,
) -> pd.Series:
    """BB squeeze strategy returns."""
    out = df.copy()
    sma = out["close"].rolling(bb_period).mean()
    std = out["close"].rolling(bb_period).std()
    out["bb_upper"] = sma + 2 * std
    out["bb_lower"] = sma - 2 * std

    high_low = out["high"] - out["low"]
    high_close = (out["high"] - out["close"].shift(1)).abs()
    low_close = (out["low"] - out["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / bb_period, adjust=False).mean()
    ema = out["close"].ewm(span=bb_period, adjust=False).mean()
    out["kc_upper"] = ema + kc_mult * atr
    out["kc_lower"] = ema - kc_mult * atr

    out["in_squeeze"] = (
        (out["bb_upper"] < out["kc_upper"])
        & (out["bb_lower"] > out["kc_lower"])
    )
    out["release"] = (
        out["in_squeeze"].shift(1) & (~out["in_squeeze"])
    )
    out["momentum"] = out["close"] - out["close"].shift(10)

    position = 0
    bars_held = 0
    positions = []
    for i in range(len(out)):
        if position == 0:
            if out["release"].iloc[i] and not pd.isna(out["momentum"].iloc[i]):
                if out["momentum"].iloc[i] > 0:
                    position = 1
                    bars_held = 1
                elif out["momentum"].iloc[i] < 0:
                    position = -1
                    bars_held = 1
        else:
            bars_held += 1
            mid = (out["bb_upper"].iloc[i] + out["bb_lower"].iloc[i]) / 2
            if (
                bars_held >= max_holding
                or (position == 1 and out["close"].iloc[i] < mid)
                or (position == -1 and out["close"].iloc[i] > mid)
            ):
                position = 0
                bars_held = 0
        positions.append(position)

    pos_series = pd.Series(positions, index=out.index)
    out["ret"] = out["close"].pct_change()
    out["pos_change"] = pos_series.diff().abs().fillna(0)
    return pos_series * out["ret"] - out["pos_change"] * cost


def combine_breakout_strategies(
    strategy_returns: dict,
    weights: dict = None,
    periods_per_year: int = 252,
) -> dict:
    """
    Combine multiple breakout strategy returns into portfolio.

    Args:
        strategy_returns: Dict of {name: return_series}.
        weights: Dict of {name: weight}. None → equal-weight.

    Returns:
        Dict with portfolio metrics.
    """
    df = pd.DataFrame(strategy_returns).dropna(how="all").fillna(0)

    if weights is None:
        weights = {name: 1.0 / len(df.columns) for name in df.columns}

    # Build weighted portfolio returns
    portfolio_ret = pd.Series(0.0, index=df.index)
    for name in df.columns:
        portfolio_ret += df[name] * weights.get(name, 0)

    portfolio_ret_clean = portfolio_ret.dropna()
    if len(portfolio_ret_clean) < 30 or portfolio_ret_clean.std() == 0:
        return {"error": "insufficient data"}

    # Per-strategy metrics
    per_strategy = {}
    for name in df.columns:
        s = df[name].dropna()
        if len(s) > 30 and s.std() > 0:
            per_strategy[name] = {
                "sharpe": (s.mean() / s.std()) * np.sqrt(periods_per_year),
                "cagr": (1 + s.mean()) ** periods_per_year - 1,
                "max_dd": (
                    (1 + s).cumprod() / (1 + s).cumprod().cummax() - 1
                ).min(),
            }
        else:
            per_strategy[name] = {"sharpe": np.nan, "cagr": np.nan, "max_dd": np.nan}

    # Portfolio metrics
    sharpe = (
        portfolio_ret_clean.mean() / portfolio_ret_clean.std()
        * np.sqrt(periods_per_year)
    )
    cagr = (1 + portfolio_ret_clean.mean()) ** periods_per_year - 1
    eq = (1 + portfolio_ret_clean).cumprod()
    max_dd = (eq / eq.cummax() - 1).min()

    correlation = df.corr()

    return {
        "portfolio_sharpe": sharpe,
        "portfolio_cagr": cagr,
        "portfolio_max_dd": max_dd,
        "per_strategy": per_strategy,
        "correlation_matrix": correlation,
        "equity_curve": eq,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Multi-Strategy Vol Breakout Portfolio — Demo")
    print("=" * 70)

    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
    n = len(dates)

    # Synthetic data with regime mix
    rets = np.zeros(n)
    for i in range(0, n, 200):
        end = min(i + 200, n)
        rets[i:end] = np.random.randn(end - i) * 0.012

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

    # Run 3 strategies
    print("\nRunning 3 vol breakout strategies...")
    keltner_r = keltner_returns(df, ema_period=20, atr_mult=2.0)
    nr7_r = nr7_returns(df, lookback=7)
    squeeze_r = bb_squeeze_returns(df, bb_period=20, kc_mult=1.5)

    strategies = {
        "Keltner(20,2)": keltner_r,
        "NR7": nr7_r,
        "BB Squeeze": squeeze_r,
    }

    print("\n--- Equal-weight portfolio ---")
    result = combine_breakout_strategies(strategies)

    print("\nPer-strategy:")
    for name, m in result["per_strategy"].items():
        print(f"  {name:<20}: Sharpe={m['sharpe']:6.3f}  "
              f"CAGR={m['cagr']*100:6.2f}%  DD={m['max_dd']*100:7.2f}%")

    print(f"\nCombined portfolio:")
    print(f"  Sharpe:   {result['portfolio_sharpe']:.3f}")
    print(f"  CAGR:     {result['portfolio_cagr']*100:.2f}%")
    print(f"  Max DD:   {result['portfolio_max_dd']*100:.2f}%")

    avg_single = np.mean([
        m["sharpe"] for m in result["per_strategy"].values()
        if not np.isnan(m["sharpe"])
    ])
    benefit = result["portfolio_sharpe"] - avg_single
    print(f"\n  Avg single Sharpe: {avg_single:.3f}")
    print(f"  Diversification:   {benefit:+.3f}")

    print(f"\nCorrelation matrix:")
    print(result["correlation_matrix"].round(2))
    print(f"\nIdeal portfolio: avg correlation < 0.3")
