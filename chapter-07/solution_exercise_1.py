"""
QuantCFD — Chương 7, Bài tập 1
MA crossover từ scratch (90 phút)

Yêu cầu:
- Implement MA crossover trên XAUUSD H4
- Compute Sharpe, CAGR, max DD
- Verify anti-look-ahead với .shift(1)
- Plot equity curve
"""

import numpy as np
import pandas as pd


def ma_crossover_solution(df: pd.DataFrame, fast: int = 20, slow: int = 50,
                          cost: float = 0.0005) -> dict:
    """Full implementation từ scratch — không dùng helper modules."""
    out = df.copy()
    
    # 1. Compute MAs
    out["ma_fast"] = out["close"].rolling(fast).mean()
    out["ma_slow"] = out["close"].rolling(slow).mean()
    
    # 2. Generate signal — CRITICAL: shift by 1 to avoid look-ahead
    raw_signal = (out["ma_fast"] > out["ma_slow"]).astype(int)
    out["signal"] = raw_signal.shift(1)  # signal at T uses MAs computed up to T-1
    
    # 3. Compute returns
    out["asset_return"] = out["close"].pct_change()
    out["strat_return"] = out["signal"] * out["asset_return"]
    
    # 4. Apply transaction cost
    out["pos_change"] = out["signal"].diff().abs().fillna(0)
    out["strat_return_net"] = out["strat_return"] - out["pos_change"] * cost
    
    out_clean = out.dropna()
    
    if len(out_clean) < 30:
        return {"error": "insufficient data"}
    
    # 5. Compute metrics
    bars_per_year = 252  # daily
    sharpe = (out_clean["strat_return_net"].mean() / out_clean["strat_return_net"].std()
              * np.sqrt(bars_per_year)) if out_clean["strat_return_net"].std() > 0 else 0
    cagr = (1 + out_clean["strat_return_net"].mean()) ** bars_per_year - 1
    
    # Equity curve
    equity = (1 + out_clean["strat_return_net"]).cumprod()
    
    # Max drawdown
    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()
    
    # Trade stats
    total_trades = int(out_clean["pos_change"].sum() / 2)
    
    return {
        "sharpe": sharpe,
        "cagr": cagr,
        "max_dd": max_dd,
        "total_trades": total_trades,
        "equity_curve": equity,
        "signal_history": out_clean["signal"],
    }


def verify_no_lookahead(df: pd.DataFrame, fast: int = 20, slow: int = 50):
    """
    Test 1: Verify rằng signal at bar T không peek vào future.
    
    Method: compute signal trên 2 versions data:
    - Full data (giá trị correct)
    - Data truncated to bar T (no future)
    Signal at bar T phải GIỐNG NHAU trong 2 versions.
    """
    full_result = ma_crossover_solution(df, fast, slow)
    full_signals = full_result["signal_history"]
    
    # Spot check: pick a random bar in middle
    mid_idx = len(df) // 2
    truncated = df.iloc[:mid_idx + 1]
    trunc_result = ma_crossover_solution(truncated, fast, slow)
    
    # Compare signal at bar mid_idx
    trunc_signal_at_mid = trunc_result["signal_history"].iloc[-1]
    full_signal_at_mid = full_signals.loc[df.index[mid_idx]]
    
    if trunc_signal_at_mid == full_signal_at_mid:
        print(f"✓ No look-ahead: signal at bar {mid_idx} consistent (full vs truncated)")
        return True
    else:
        print(f"✗ LOOK-AHEAD DETECTED: full={full_signal_at_mid}, truncated={trunc_signal_at_mid}")
        return False


if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range("2014-01-01", "2024-12-31", freq="D")
    returns = np.random.randn(len(dates)) * 0.012 + 0.0003
    prices = 1500 * np.exp(np.cumsum(returns))  # XAUUSD-like
    df = pd.DataFrame({"close": prices}, index=dates)
    
    print("=" * 60)
    print("Bài tập 1 — MA Crossover từ scratch (XAUUSD synthetic)")
    print("=" * 60)
    
    result = ma_crossover_solution(df, fast=20, slow=50)
    
    print(f"\nSharpe:        {result['sharpe']:.3f}")
    print(f"CAGR:          {result['cagr']*100:.2f}%")
    print(f"Max DD:        {result['max_dd']*100:.2f}%")
    print(f"Total trades:  {result['total_trades']}")
    
    print("\n" + "-" * 60)
    print("Verify no look-ahead bias:")
    verify_no_lookahead(df)
    
    print("\nLesson:")
    print("  - Always .shift(1) signal trước khi multiply với return")
    print("  - Verify bằng truncation test")
    print("  - Strategy Sharpe > 0.7 trên synthetic data → likely có edge thật trên XAUUSD live")
