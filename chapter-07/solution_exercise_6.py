"""
QuantCFD — Chương 7, Bài tập 6 (BONUS)
Walk-forward MA params (180 phút)

Yêu cầu:
- Implement walk-forward analysis cho MA crossover
- 3-year IS, 1-year OOS, step 1 year
- Test trên 3 instruments (XAUUSD, EURUSD, BTCUSD synthetic)
- Compute WFE, verdict
- Identify which instruments suitable cho deployment
"""

import numpy as np
import pandas as pd
from itertools import product


def backtest_ma(df, fast, slow, cost=0.0005, periods_per_year=252):
    """Single MA backtest."""
    out = df.copy()
    out["ma_fast"] = out["close"].rolling(fast).mean()
    out["ma_slow"] = out["close"].rolling(slow).mean()
    out["signal"] = (out["ma_fast"] > out["ma_slow"]).astype(int).shift(1)
    out["ret"] = out["close"].pct_change()
    out["pos_change"] = out["signal"].diff().abs().fillna(0)
    out["strat_ret"] = out["signal"] * out["ret"] - out["pos_change"] * cost
    out = out.dropna()
    
    if len(out) < 30 or out["strat_ret"].std() == 0:
        return -999
    return (out["strat_ret"].mean() / out["strat_ret"].std()) * np.sqrt(periods_per_year)


def walk_forward_full(df, is_years=3, oos_years=1, step_years=1,
                      fast_grid=(10, 15, 20, 25, 30),
                      slow_grid=(40, 50, 60, 80, 100),
                      periods_per_year=252):
    """Walk-forward analysis returning DataFrame of windows."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    start = df.index[0]
    end = df.index[-1]
    results = []
    window_id = 0
    
    is_start = start
    while True:
        is_end = is_start + pd.DateOffset(years=is_years)
        oos_start = is_end
        oos_end = oos_start + pd.DateOffset(years=oos_years)
        if oos_end > end:
            break
        
        is_data = df[is_start:is_end]
        oos_data = df[oos_start:oos_end]
        
        if len(is_data) < 100:
            is_start = is_start + pd.DateOffset(years=step_years)
            window_id += 1
            continue
        
        best_sharpe_is = -999
        best = None
        for f, s in product(fast_grid, slow_grid):
            if f >= s:
                continue
            sh = backtest_ma(is_data, f, s, periods_per_year=periods_per_year)
            if sh > best_sharpe_is:
                best_sharpe_is = sh
                best = (f, s)
        
        if best is None:
            is_start = is_start + pd.DateOffset(years=step_years)
            window_id += 1
            continue
        
        oos_sharpe = backtest_ma(oos_data, best[0], best[1],
                                 periods_per_year=periods_per_year)
        wfe = oos_sharpe / best_sharpe_is if best_sharpe_is > 0 else 0
        
        results.append({
            "window": window_id,
            "is_period": f"{is_start.year}-{is_end.year}",
            "oos_period": f"{oos_start.year}-{oos_end.year}",
            "best_fast": best[0], "best_slow": best[1],
            "is_sharpe": best_sharpe_is,
            "oos_sharpe": oos_sharpe,
            "wfe": wfe,
        })
        is_start = is_start + pd.DateOffset(years=step_years)
        window_id += 1
    
    return pd.DataFrame(results)


def deployment_verdict(wfa: pd.DataFrame) -> str:
    """Determine GO/NO-GO for live deployment."""
    if len(wfa) == 0:
        return "NO_DATA"
    
    avg_oos = wfa["oos_sharpe"].mean()
    pct_pos = (wfa["oos_sharpe"] > 0).mean() * 100
    avg_wfe = wfa["wfe"].mean()
    
    if avg_oos >= 0.5 and pct_pos >= 60 and avg_wfe >= 0.5:
        return "GO — strong edge across windows"
    elif avg_oos >= 0.3 and pct_pos >= 50:
        return "PROCEED WITH CAUTION — marginal edge, monitor closely"
    else:
        return "NO-GO — insufficient OOS performance"


def generate_synthetic_data(asset: str, dates: pd.DatetimeIndex,
                             seed: int = 42) -> pd.DataFrame:
    """Generate asset-specific synthetic data."""
    np.random.seed(seed)
    
    profiles = {
        "XAUUSD": {"vol": 0.012, "drift": 0.0003, "regime_strength": 0.4},
        "EURUSD": {"vol": 0.006, "drift": 0.0001, "regime_strength": 0.1},
        "BTCUSD": {"vol": 0.040, "drift": 0.0010, "regime_strength": 0.7},
    }
    
    p = profiles.get(asset, profiles["XAUUSD"])
    n = len(dates)
    
    # Add regime-switching trend component
    base_returns = np.random.randn(n) * p["vol"] + p["drift"]
    
    # Add momentum: each 250 days, momentum builds
    momentum = np.zeros(n)
    for i in range(0, n, 250):
        end = min(i + 250, n)
        regime_drift = np.random.choice([-1, 0, 1]) * p["regime_strength"] * 0.001
        momentum[i:end] = regime_drift
    
    returns = base_returns + momentum
    prices = 100 * np.exp(np.cumsum(returns))
    
    return pd.DataFrame({"close": prices}, index=dates)


if __name__ == "__main__":
    dates = pd.date_range("2014-01-01", "2024-12-31", freq="D")
    
    print("=" * 70)
    print("Bài tập 6 (BONUS) — Walk-Forward Analysis 3 instruments")
    print("=" * 70)
    
    for asset, seed in [("XAUUSD", 42), ("EURUSD", 100), ("BTCUSD", 7)]:
        df = generate_synthetic_data(asset, dates, seed=seed)
        
        ppy = 365 if asset == "BTCUSD" else 252
        wfa = walk_forward_full(df, periods_per_year=ppy)
        
        print(f"\n{'─' * 70}")
        print(f"Asset: {asset}")
        print(f"{'─' * 70}")
        print(wfa[["window", "oos_period", "best_fast", "best_slow",
                   "is_sharpe", "oos_sharpe", "wfe"]].round(2).to_string(index=False))
        
        avg_oos = wfa["oos_sharpe"].mean()
        median_oos = wfa["oos_sharpe"].median()
        pct_pos = (wfa["oos_sharpe"] > 0).mean() * 100
        avg_wfe = wfa["wfe"].mean()
        
        print(f"\nSummary {asset}:")
        print(f"  Avg OOS Sharpe:      {avg_oos:.3f}")
        print(f"  Median OOS Sharpe:   {median_oos:.3f}")
        print(f"  % positive windows:  {pct_pos:.0f}%")
        print(f"  Avg WFE:             {avg_wfe:.3f}")
        print(f"  Verdict:             {deployment_verdict(wfa)}")
    
    print("\n" + "=" * 70)
    print("LESSONS:")
    print("=" * 70)
    print("  - Best params CHANGE qua mỗi window — không có 'magic params'")
    print("  - High Hurst assets (BTC) tend to have higher avg OOS Sharpe")
    print("  - Random-walk-like assets (EUR/USD synthetic) often fail WFA")
    print("  - WFE > 0.5 = strategy không overfit nghiêm trọng")
    print("  - % windows positive > 60% = robust edge across regimes")
