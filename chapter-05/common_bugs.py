"""
QuantCFD Chương 5 — 5 common engine bugs với fix

Mỗi bug đi kèm:
    1. Ví dụ code BUGGY
    2. Cách phát hiện
    3. Code FIX
    4. Test verify

Run: python common_bugs.py
"""
import numpy as np
import pandas as pd


# ============================================================
# BUG 1: Look-ahead trong indicator (ATR dùng future bar)
# ============================================================
def bug1_atr_lookahead():
    print("=== BUG 1: ATR look-ahead ===")
    df = pd.DataFrame({
        'high': [101, 102, 103, 104, 105],
        'low':  [99, 100, 101, 102, 103],
        'close':[100, 101, 102, 103, 104],
    })

    # BUG: dùng ATR rolling không shift → bar t dùng giá close của bar t
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_buggy = tr.rolling(3).mean()
    print("ATR_buggy at t (dùng cả high/low của t — leak!):")
    print(atr_buggy)

    # FIX: shift ATR thêm 1 bar trước khi dùng làm signal
    atr_safe = tr.rolling(3).mean().shift(1)
    print("\nATR_safe (đã shift, chỉ dùng data <= t-1 cho decision tại t):")
    print(atr_safe)


# ============================================================
# BUG 2: Position off-by-one (signal không shift)
# ============================================================
def bug2_position_off_by_one():
    print("\n=== BUG 2: Position off-by-one ===")
    df = pd.DataFrame({
        'close': [100, 102, 99, 101, 103, 105],
    })
    signals = pd.Series([0, 1, 1, -1, -1, 0])

    # BUG: position lấy signal cùng bar — như biết tương lai
    returns = df['close'].pct_change()
    pnl_buggy = signals * returns       # SAI
    print(f"PnL buggy total: {pnl_buggy.sum():.4f}")

    # FIX: signal generate tại close bar t → execute tại open bar t+1
    pnl_safe = signals.shift(1) * returns
    print(f"PnL safe total:  {pnl_safe.sum():.4f}")

    print(f"Sai số: {(pnl_buggy.sum() - pnl_safe.sum())/abs(pnl_safe.sum())*100:.1f}% inflate")


# ============================================================
# BUG 3: Annualization wrong (crypto dùng 252 thay vì 365)
# ============================================================
def bug3_annualization():
    print("\n=== BUG 3: Annualization wrong ===")
    np.random.seed(42)
    daily_returns = pd.Series(np.random.normal(0.001, 0.02, 365))

    sharpe_wrong = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    sharpe_correct = daily_returns.mean() / daily_returns.std() * np.sqrt(365)

    print(f"Crypto strategy daily returns: μ={daily_returns.mean()*100:.2f}%, σ={daily_returns.std()*100:.2f}%")
    print(f"Sharpe wrong (×√252):    {sharpe_wrong:.2f}")
    print(f"Sharpe correct (×√365):  {sharpe_correct:.2f}")
    print(f"Inflation: {(sharpe_correct - sharpe_wrong)/sharpe_wrong*100:.1f}%")
    print(f"\nQuy tắc:")
    print(f"  Stocks/forex (5d/wk):   periods_per_year = 252")
    print(f"  Crypto (24/7):          periods_per_year = 365")
    print(f"  H1 stocks:              periods_per_year = 252 × 6.5 = 1638")
    print(f"  H1 crypto:              periods_per_year = 365 × 24 = 8760")
    print(f"  M1 forex:               periods_per_year = 252 × 24 × 60 ≈ 363,000")


# ============================================================
# BUG 4: Double-counting stop loss
# ============================================================
def bug4_double_count_stop():
    print("\n=== BUG 4: Double-counting stop loss ===")
    # Buggy: stop trigger → close position → ALSO record next bar PnL với stop level
    # Realistic engines should set position=0 immediately after stop, no further PnL.

    df = pd.DataFrame({
        'close': [100, 99, 95, 96, 97],   # Drop 5% bar 2, recover slightly
        'low':   [99, 98, 94, 95.5, 96.5],
    })
    entry = 100
    stop_price = 97   # 3% stop loss

    # BUG: position tracking tiếp tục sau khi stop hit
    pos_buggy = pd.Series([1, 1, 0, 0, 0])  # OK, set 0 sau bar 2
    # Nhưng PnL tính:
    pnl_buggy_bar2 = (df['close'].iloc[2] - df['close'].iloc[1]) / df['close'].iloc[1]
    # Bar 2 close = 95, low = 94, stop = 97 → đáng lẽ exit at 97, không 95!
    print(f"Bar 2 close = {df['close'].iloc[2]}, low = {df['low'].iloc[2]}, stop = {stop_price}")
    print(f"PnL nếu mark-to-close (BUG): {pnl_buggy_bar2*100:.2f}%")

    # FIX: khi low <= stop trong bar, exit at stop_price exactly
    if df['low'].iloc[2] <= stop_price <= df['close'].iloc[1]:
        pnl_fix_bar2 = (stop_price - df['close'].iloc[1]) / df['close'].iloc[1]
    else:
        pnl_fix_bar2 = pnl_buggy_bar2
    print(f"PnL fix (exit at stop price): {pnl_fix_bar2*100:.2f}%")

    # Slippage realistic: exit at stop_price - slippage
    slippage = 0.05
    pnl_realistic = (stop_price - slippage - df['close'].iloc[1]) / df['close'].iloc[1]
    print(f"PnL realistic (stop + 5cent slippage): {pnl_realistic*100:.2f}%")


# ============================================================
# BUG 5: Fillna với forward-fill mask leak
# ============================================================
def bug5_ffill_leak():
    print("\n=== BUG 5: Forward-fill leak ===")
    # Buggy: dùng ffill cho returns / signals → fill NaN bằng giá trị FUTURE
    # khi data có gap (weekend, holiday)

    df = pd.DataFrame({
        'close': [100, 101, np.nan, np.nan, 110, 111],
    })

    # BUG: bfill (back-fill) — leak future
    df_bfill = df.copy()
    df_bfill['close_filled'] = df_bfill['close'].bfill()
    print("BUG: bfill làm bar 2-3 dùng giá bar 4 (110):")
    print(df_bfill)

    # FIX: ffill OK vì chỉ dùng past, hoặc forward fill rồi mask out các bar không tradeable
    df_safe = df.copy()
    df_safe['close_filled'] = df_safe['close'].ffill()
    df_safe['tradeable'] = df_safe['close'].notna()
    print("\nFIX: ffill + tradeable mask:")
    print(df_safe)
    print("\nSignals chỉ generate khi tradeable=True. Position carry forward khi gap.")


# ============================================================
# Run tất cả
# ============================================================
if __name__ == '__main__':
    bug1_atr_lookahead()
    bug2_position_off_by_one()
    bug3_annualization()
    bug4_double_count_stop()
    bug5_ffill_leak()
    print("\n" + "="*60)
    print("Quy tắc tổng quát:")
    print("  1. Mọi indicator → shift(1) trước khi làm signal")
    print("  2. Mọi signal → shift(1) trước khi tính PnL")
    print("  3. Crypto annualize = √365, không √252")
    print("  4. Stop loss phải exit at stop level, không close")
    print("  5. Tránh bfill, dùng ffill + tradeable mask")
