"""
QuantCFD - Chapter 2 - Bài 3 Solution
======================================
Lời giải Bài tập 3 / Chương 2: Tìm look-ahead bug trong code RSI strategy.

Code gốc có 2 bug chính (và 1 bug phụ về RSI calculation).

Chạy:
    python chapter-02/solution_exercise_3.py
"""
import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================================
#                          CODE GỐC (CÓ BUG)
# ============================================================================
def buggy_rsi_strategy() -> pd.DataFrame:
    """Bản code gốc trong sách — có 2 bug, sẽ phân tích bên dưới."""
    df = yf.download(
        "BTC-USD",
        start="2022-01-01",
        end="2024-12-31",
        progress=False,
        auto_adjust=True,
    )

    # Tính RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - 100 / (1 + rs)

    # Vào lệnh
    df["signal"] = 0
    df.loc[df["rsi"] < 30, "signal"] = 1  # long
    df.loc[df["rsi"] > 70, "signal"] = -1  # short

    # ❌ BUG 1: signal ngày T cho ngày T (look-ahead)
    df["ret"] = df["Close"].pct_change()
    df["pnl"] = df["signal"] * df["ret"]

    # ❌ BUG 2: cap PnL theo PnL thực tế đã xảy ra
    df.loc[df["pnl"] < -0.03, "pnl"] = -0.03

    return df


# ============================================================================
#                       CODE ĐÚNG (ĐÃ FIX 2 BUG)
# ============================================================================
def fixed_rsi_strategy() -> pd.DataFrame:
    """
    Đã fix cả 2 bug:
      Bug 1: shift(1) signal trước khi áp dụng vào returns ngày sau.
      Bug 2: stop loss thực hiện đúng — dựa trên giá intraday low/high
             (đơn giản hoá: dùng Low của ngày cho long, High cho short),
             không phải PnL đã được tính ra rồi cap.
    """
    df = yf.download(
        "BTC-USD",
        start="2022-01-01",
        end="2024-12-31",
        progress=False,
        auto_adjust=True,
    )

    # Tính RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - 100 / (1 + rs)

    # Generate signal từ RSI ngày T
    df["raw_signal"] = 0
    df.loc[df["rsi"] < 30, "raw_signal"] = 1
    df.loc[df["rsi"] > 70, "raw_signal"] = -1

    # ✅ FIX BUG 1: shift signal 1 ngày
    # Logic: RSI tính từ close ngày T → ta chỉ có thể vào lệnh ngày T+1 mở cửa
    # → return chiến lược nhận được là return từ T+1 đến T+2 (close-to-close).
    df["position"] = df["raw_signal"].shift(1)

    # ✅ FIX BUG 2: stop loss bằng cách check intraday low/high
    # Cho long: nếu Low của ngày < entry × (1 - 0.03), giả định stop chạm
    # Cho short: nếu High của ngày > entry × (1 + 0.03), giả định stop chạm
    # Đơn giản hoá: assume entry = open của ngày T+1 = close của ngày T (rough).

    # PnL đầy đủ (chưa stop)
    df["full_ret"] = df["Close"].pct_change()
    df["raw_pnl"] = df["position"] * df["full_ret"]

    # Tính loss capping ĐÚNG dựa trên Low/High intraday vs entry
    # Entry giả định = Close của ngày trước (open ~ close của T-1)
    entry_price = df["Close"].shift(1)
    long_intraday_low = (df["Low"] - entry_price) / entry_price  # âm
    short_intraday_high = -(df["High"] - entry_price) / entry_price  # âm cho short

    # Nếu position long và low ngày đó < -3% so entry → stop chạm tại -3%
    long_stop_hit = (df["position"] == 1) & (long_intraday_low < -0.03)
    short_stop_hit = (df["position"] == -1) & (short_intraday_high < -0.03)

    df["pnl"] = df["raw_pnl"]
    df.loc[long_stop_hit, "pnl"] = -0.03
    df.loc[short_stop_hit, "pnl"] = -0.03

    return df


# ============================================================================
#                              SO SÁNH
# ============================================================================
def compare_results() -> None:
    print("\n" + "=" * 75)
    print(" SO SÁNH: BUGGY vs FIXED RSI STRATEGY (BTC-USD, 2022-2024)")
    print("=" * 75)

    buggy = buggy_rsi_strategy()
    fixed = fixed_rsi_strategy()

    # BTC trade 24/7 → dùng 365
    pyear = 365

    def report(df, label):
        pnl = df["pnl"].dropna()
        if len(pnl) == 0:
            print(f"\n{label}: No data")
            return
        sharpe = pnl.mean() / pnl.std() * np.sqrt(pyear)
        cum = (1 + pnl).cumprod()
        total = cum.iloc[-1] - 1
        max_dd = (cum / cum.cummax() - 1).min()
        n_trades = (df["signal"].diff().abs() if "signal" in df else df["position"].diff().abs())
        n_trades = int(n_trades.sum() / 2) if n_trades is not None else None

        print(f"\n{label}:")
        print(f"  Sharpe:        {sharpe:+.2f}")
        print(f"  Total return:  {total:+.2%}")
        print(f"  Max drawdown:  {max_dd:+.2%}")

    report(buggy, "❌ BUGGY (code gốc trong sách)")
    report(fixed, "✅ FIXED (đã sửa 2 bug)")

    print(
        "\nGiải thích 2 bugs:\n"
        "\n  BUG 1 — Look-ahead bias trong signal application:\n"
        "      df['pnl'] = df['signal'] * df['ret']\n"
        "      ↑ Signal tính từ close ngày T, nhưng nhân với return ngày T\n"
        "        (close T-1 → close T). Như vậy tại close T, ta đã 'biết'\n"
        "        signal và 'biết' return — không thể trong realtime.\n"
        "      Fix: df['position'] = df['signal'].shift(1)\n"
        "\n  BUG 2 — Stop loss áp dụng sai cách:\n"
        "      df.loc[df['pnl'] < -0.03, 'pnl'] = -0.03\n"
        "      ↑ Cap PnL DỰA TRÊN PnL thực tế đã xảy ra. Như vậy ta đang\n"
        "        dùng kết quả ngày để override — không thể trong realtime.\n"
        "      Trong thực tế, stop loss phải triggered khi giá CHẠM stop,\n"
        "        không phải khi PnL đã chạm mức nào đó.\n"
        "      Fix: dùng Low (cho long) / High (cho short) intraday để\n"
        "        check liệu stop có bị chạm.\n"
        "\n  Bug phụ — RSI calculation:\n"
        "      Code dùng SMA cho gain/loss. RSI gốc của Wilder dùng EMA\n"
        "      với α=1/14. Không phải bug nghiêm trọng — chỉ là biến thể.\n"
    )


if __name__ == "__main__":
    compare_results()
