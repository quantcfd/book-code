"""
QuantCFD - Chapter 3 - Pandas Patterns
========================================
Section 3.3: 10 thao tác pandas chiếm 80% workflow của quant retail.

Mỗi pattern là một section trong code dưới đây. Đọc xong file này,
anh em đủ pandas để làm 80% bài toán quant cơ bản.

Chạy:
    python chapter-03/pandas_patterns.py
"""
import numpy as np
import pandas as pd
import yfinance as yf


def main() -> None:
    # Load data 5 năm Gold (proxy XAUUSD CFD)
    df = yf.download(
        "GC=F",
        start="2020-01-01",
        end="2024-12-31",
        progress=False,
        auto_adjust=True,
    )

    print(f"\nLoaded {len(df)} rows of GC=F data\n")

    # ------------------------------------------------------------------
    # 1. pct_change — % thay đổi
    # ------------------------------------------------------------------
    df["Return"] = df["Close"].pct_change()

    # ------------------------------------------------------------------
    # 2. rolling — moving statistics
    # ------------------------------------------------------------------
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["Vol20"] = df["Return"].rolling(20).std() * np.sqrt(252)

    # ------------------------------------------------------------------
    # 3. shift — dịch theo thời gian (CHỐNG LOOK-AHEAD!)
    # ------------------------------------------------------------------
    df["MA20_lag1"] = df["MA20"].shift(1)  # MA của ngày hôm qua

    # ------------------------------------------------------------------
    # 4. np.where — tạo signal có điều kiện
    # ------------------------------------------------------------------
    df["Signal"] = np.where(df["MA20"] > df["MA50"], 1, -1)
    df["Position"] = df["Signal"].shift(1)  # Vào lệnh ngày T+1, KHÔNG T

    # ------------------------------------------------------------------
    # 5. cumprod — equity curve
    # ------------------------------------------------------------------
    df["StratRet"] = df["Position"] * df["Return"]
    df["Equity"] = (1 + df["StratRet"]).cumprod()

    # ------------------------------------------------------------------
    # 6. resample — đổi tần suất daily → weekly
    # ------------------------------------------------------------------
    weekly_close = df["Close"].resample("W").last()
    print(f"Weekly closes (5 cuối cùng):")
    print(weekly_close.tail())

    # ------------------------------------------------------------------
    # 7. groupby — gom nhóm theo tháng
    # ------------------------------------------------------------------
    monthly_return = (
        df["Return"]
        .groupby(df.index.to_period("M"))
        .apply(lambda x: (1 + x).prod() - 1)
    )
    print(f"\nMonthly returns (5 cuối cùng):")
    print(monthly_return.tail().apply(lambda x: f"{x:+.2%}"))

    # ------------------------------------------------------------------
    # 8. dropna — bỏ NaN (sau rolling/shift sinh ra NaN)
    # ------------------------------------------------------------------
    df_clean = df.dropna()
    print(f"\nBefore dropna: {len(df)} rows. After: {len(df_clean)} rows.")

    # ------------------------------------------------------------------
    # 9. filter có điều kiện
    # ------------------------------------------------------------------
    high_vol_days = df[df["Vol20"] > 0.30]
    print(f"\nDays with annual vol > 30%: {len(high_vol_days)}")

    # ------------------------------------------------------------------
    # 10. Tính max drawdown trên equity curve
    # ------------------------------------------------------------------
    rolling_max = df["Equity"].cummax()
    drawdown = (df["Equity"] - rolling_max) / rolling_max
    print(f"\nStrategy max drawdown: {drawdown.min():.2%}")

    # ------------------------------------------------------------------
    # Tổng kết
    # ------------------------------------------------------------------
    print(f"\nFinal equity (start = 1.0): {df_clean['Equity'].iloc[-1]:.4f}")
    print(f"Total return: {(df_clean['Equity'].iloc[-1] - 1):+.2%}")

    sharpe = (
        df_clean["StratRet"].mean()
        / df_clean["StratRet"].std()
        * np.sqrt(252)
    )
    print(f"Annualized Sharpe: {sharpe:+.2f}")

    print("\nLast 5 rows of strategy DataFrame:")
    cols = ["Close", "MA20", "MA50", "Position", "StratRet", "Equity"]
    print(df_clean[cols].tail())


if __name__ == "__main__":
    main()
