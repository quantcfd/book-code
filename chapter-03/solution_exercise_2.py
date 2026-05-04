"""
QuantCFD - Chapter 3 - Bài 2 Solution
=======================================
Lời giải Bài tập 2 / Chương 3: Pandas 10 Functions Drill.

Tải XAUUSD 5 năm và trả lời 10 câu hỏi, mỗi câu dùng đúng 1 function
trong list 10 functions của Section 3.3.

Chạy:
    python chapter-03/solution_exercise_2.py
"""
import numpy as np
import pandas as pd
import yfinance as yf


def main() -> None:
    # Load data XAUUSD 5 năm (proxy: Gold futures GC=F)
    gold = yf.download(
        "GC=F",
        start="2020-01-01",
        end="2024-12-31",
        progress=False,
        auto_adjust=True,
    )

    print(f"\nLoaded {len(gold)} rows of GC=F data\n")

    # ------------------------------------------------------------------
    # 1. Tính daily return
    # ------------------------------------------------------------------
    gold["Return"] = gold["Close"].pct_change()
    print(f"1. Daily return (5 cuối): "
          f"{gold['Return'].tail().apply(lambda x: f'{x:+.4%}').tolist()}")

    # ------------------------------------------------------------------
    # 2. Moving average 20 ngày
    # ------------------------------------------------------------------
    gold["MA20"] = gold["Close"].rolling(20).mean()
    print(f"\n2. MA20 (5 cuối): {gold['MA20'].tail().apply(lambda x: f'{x:.2f}').tolist()}")

    # ------------------------------------------------------------------
    # 3. MA20 của hôm trước (chống look-ahead)
    # ------------------------------------------------------------------
    gold["MA20_yesterday"] = gold["MA20"].shift(1)
    diff = (gold["MA20"] - gold["MA20_yesterday"]).dropna()
    print(f"\n3. MA20_yesterday created. Mean diff (today - yesterday): {diff.mean():.4f}")

    # ------------------------------------------------------------------
    # 4. Đổi data từ daily → monthly (giá close ngày cuối tháng)
    # ------------------------------------------------------------------
    monthly = gold["Close"].resample("M").last()
    print(f"\n4. Monthly close (5 tháng cuối):")
    for date, val in monthly.tail().items():
        print(f"   {date.strftime('%Y-%m'):10s}  {val:.2f}")

    # ------------------------------------------------------------------
    # 5. Bỏ tất cả row có NaN
    # ------------------------------------------------------------------
    before = len(gold)
    gold_clean = gold.dropna()
    after = len(gold_clean)
    print(f"\n5. Dropna: {before} → {after} rows ({before - after} rows dropped)")

    # ------------------------------------------------------------------
    # 6. Equity curve buy-and-hold (cumprod)
    # ------------------------------------------------------------------
    gold_clean = gold_clean.copy()
    gold_clean["Equity"] = (1 + gold_clean["Return"]).cumprod()
    final_equity = gold_clean["Equity"].iloc[-1]
    print(f"\n6. Buy-and-hold equity from 2020-2024: "
          f"{final_equity:.4f}  ({(final_equity - 1):+.2%})")

    # ------------------------------------------------------------------
    # 7. Tổng return mỗi năm (groupby)
    # ------------------------------------------------------------------
    yearly_return = gold["Return"].groupby(gold.index.year).apply(
        lambda x: (1 + x).prod() - 1
    )
    print(f"\n7. Annual returns:")
    for year, ret in yearly_return.items():
        print(f"   {year}  {ret:+.2%}")

    # ------------------------------------------------------------------
    # 8. Merge với DXY data, tính correlation
    # ------------------------------------------------------------------
    dxy = yf.download(
        "DX-Y.NYB",
        start="2020-01-01",
        end="2024-12-31",
        progress=False,
        auto_adjust=True,
    )
    dxy_returns = dxy["Close"].pct_change().rename("DXY_Return")

    # Merge bằng index align
    merged = pd.concat([gold["Return"].rename("Gold_Return"), dxy_returns], axis=1).dropna()
    correlation = merged["Gold_Return"].corr(merged["DXY_Return"])
    print(f"\n8. Gold vs DXY correlation (2020-2024): {correlation:+.3f}")
    print(f"   (Kỳ vọng âm — vàng định giá USD, USD mạnh = vàng giảm)")

    # ------------------------------------------------------------------
    # 9. Rank top 5 tháng tốt nhất
    # ------------------------------------------------------------------
    monthly_returns = gold["Return"].resample("M").apply(
        lambda x: (1 + x).prod() - 1
    )
    top5_months = monthly_returns.nlargest(5)
    print(f"\n9. Top 5 best months for Gold (2020-2024):")
    for date, ret in top5_months.items():
        print(f"   {date.strftime('%Y-%m'):10s}  {ret:+.2%}")

    # ------------------------------------------------------------------
    # 10. Đánh dấu high vol vs low vol (np.where)
    # ------------------------------------------------------------------
    gold["Vol20"] = gold["Return"].rolling(20).std() * np.sqrt(252)
    gold["VolRegime"] = np.where(gold["Vol20"] > 0.25, "high vol", "low vol")

    regime_counts = gold["VolRegime"].value_counts()
    print(f"\n10. Volatility regime classification:")
    for regime, count in regime_counts.items():
        pct = count / len(gold) * 100
        print(f"    {regime:10s}  {count} days  ({pct:.1f}%)")

    print("\n" + "=" * 70)
    print(" Đã dùng đủ 10 functions: pct_change, rolling, shift, resample,")
    print(" dropna, cumprod, groupby, merge (qua concat), nlargest, where")
    print("=" * 70)


if __name__ == "__main__":
    main()
