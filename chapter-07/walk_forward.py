"""
QuantCFD — Chương 7
Walk-Forward Analysis

Reference: Ch7.4.5.
Validate strategy bằng rolling in-sample / out-of-sample windows.
Tính Walk-Forward Efficiency (WFE) = OOS Sharpe / IS Sharpe.
"""

import numpy as np
import pandas as pd
from itertools import product


def backtest_ma_simple(df: pd.DataFrame, fast: int, slow: int,
                       cost: float = 0.0005) -> dict:
    """Simple MA crossover backtest. Returns Sharpe, CAGR, MaxDD."""
    out = df.copy()
    out["ma_fast"] = out["close"].rolling(fast).mean()
    out["ma_slow"] = out["close"].rolling(slow).mean()
    out["signal"] = (out["ma_fast"] > out["ma_slow"]).astype(int).shift(1)
    out["ret"] = out["close"].pct_change()
    out["pos_change"] = out["signal"].diff().abs().fillna(0)
    out["strat_ret"] = out["signal"] * out["ret"] - out["pos_change"] * cost
    out = out.dropna()
    
    if len(out) < 30 or out["strat_ret"].std() == 0:
        return {"sharpe": -999, "cagr": 0, "max_dd": 0}
    
    sharpe = (out["strat_ret"].mean() / out["strat_ret"].std()) * np.sqrt(252)
    cagr = (1 + out["strat_ret"].mean()) ** 252 - 1
    eq = (1 + out["strat_ret"]).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return {"sharpe": sharpe, "cagr": cagr, "max_dd": dd}


def walk_forward_ma(df: pd.DataFrame, is_years: int = 3, oos_years: int = 1,
                    step_years: int = 1,
                    fast_grid=(10, 15, 20, 25, 30),
                    slow_grid=(40, 50, 60, 80, 100),
                    cost: float = 0.0005) -> pd.DataFrame:
    """
    Walk-forward analysis cho MA crossover.
    
    Returns DataFrame with columns:
        window, is_period, oos_period, best_params,
        is_sharpe, oos_sharpe, oos_cagr, oos_dd, wfe.
    """
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
        
        if len(is_data) < 100 or len(oos_data) < 30:
            is_start = is_start + pd.DateOffset(years=step_years)
            window_id += 1
            continue
        
        # Grid search on IS
        best_sharpe_is = -999
        best_params = None
        for fast, slow in product(fast_grid, slow_grid):
            if fast >= slow:
                continue
            r = backtest_ma_simple(is_data, fast, slow, cost)
            if r["sharpe"] > best_sharpe_is:
                best_sharpe_is = r["sharpe"]
                best_params = (fast, slow)
        
        if best_params is None:
            is_start = is_start + pd.DateOffset(years=step_years)
            window_id += 1
            continue
        
        # Apply on OOS
        oos_result = backtest_ma_simple(oos_data, best_params[0], best_params[1], cost)
        
        wfe = oos_result["sharpe"] / best_sharpe_is if best_sharpe_is > 0 else 0
        
        results.append({
            "window": window_id,
            "is_period": f"{is_start.year}-{is_end.year}",
            "oos_period": f"{oos_start.year}-{oos_end.year}",
            "best_fast": best_params[0],
            "best_slow": best_params[1],
            "is_sharpe": best_sharpe_is,
            "oos_sharpe": oos_result["sharpe"],
            "oos_cagr": oos_result["cagr"],
            "oos_dd": oos_result["max_dd"],
            "wfe": wfe,
        })
        
        is_start = is_start + pd.DateOffset(years=step_years)
        window_id += 1
    
    return pd.DataFrame(results)


def wfa_summary(wfa_results: pd.DataFrame) -> dict:
    """Summary stats from WFA results."""
    if len(wfa_results) == 0:
        return {"verdict": "NO_DATA"}
    
    avg_oos_sharpe = wfa_results["oos_sharpe"].mean()
    median_oos_sharpe = wfa_results["oos_sharpe"].median()
    pct_positive = (wfa_results["oos_sharpe"] > 0).mean() * 100
    avg_wfe = wfa_results["wfe"].mean()
    
    # Verdict
    if avg_oos_sharpe < 0.3 or pct_positive < 60 or avg_wfe < 0.4:
        verdict = "DO_NOT_DEPLOY"
        reason = []
        if avg_oos_sharpe < 0.3:
            reason.append(f"avg OOS Sharpe {avg_oos_sharpe:.2f} < 0.3")
        if pct_positive < 60:
            reason.append(f"only {pct_positive:.0f}% windows positive")
        if avg_wfe < 0.4:
            reason.append(f"WFE {avg_wfe:.2f} < 0.4 (overfitting)")
    else:
        verdict = "PROCEED_WITH_CAUTION"
        reason = ["passes minimum thresholds"]
    
    return {
        "n_windows": len(wfa_results),
        "avg_oos_sharpe": avg_oos_sharpe,
        "median_oos_sharpe": median_oos_sharpe,
        "pct_positive_windows": pct_positive,
        "avg_wfe": avg_wfe,
        "verdict": verdict,
        "reason": "; ".join(reason),
    }


if __name__ == "__main__":
    # Generate longer synthetic data for WFA
    np.random.seed(42)
    dates = pd.date_range("2014-01-01", "2024-12-31", freq="D")
    
    # Mix of trending and choppy periods
    regime_changes = [0, 500, 1100, 1800, 2500, 3200, len(dates)]
    regime_drifts = [0.0008, -0.0003, 0.0005, 0.0002, -0.0002, 0.0010]
    returns = np.zeros(len(dates))
    for i in range(len(regime_changes) - 1):
        s, e = regime_changes[i], regime_changes[i+1]
        returns[s:e] = np.random.randn(e-s) * 0.012 + regime_drifts[i]
    
    prices = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({"close": prices}, index=dates)
    
    print("=" * 60)
    print("Walk-Forward Analysis — MA Crossover")
    print("=" * 60)
    print(f"Data: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} bars)")
    
    results = walk_forward_ma(df, is_years=3, oos_years=1, step_years=1)
    
    print(f"\n{len(results)} windows generated")
    print(results[["window", "is_period", "oos_period", "best_fast", "best_slow",
                   "is_sharpe", "oos_sharpe", "wfe"]].round(2).to_string(index=False))
    
    print("\n" + "-" * 60)
    summary = wfa_summary(results)
    print("WFA Summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
