"""
QuantCFD - Chapter 2 - Look-ahead Bug Detector
================================================
Section 2.6: Phát hiện look-ahead bias.

Demo: cùng một chiến lược MA crossover, có và không có look-ahead bug.
Khác biệt thực tế: Sharpe có thể bị phóng đại lên +1.5 đến +2.5 do bug.

Chạy:
    python chapter-02/lookahead_bug_detector.py
"""
import numpy as np
import pandas as pd
import yfinance as yf


def quick_stats(pnl: pd.Series, label: str, periods_per_year: int = 252) -> dict:
    """Tính nhanh Sharpe + total return cho một series PnL."""
    pnl = pnl.dropna()
    if len(pnl) == 0:
        print(f"  {label:25s}  No data")
        return {}

    sharpe = pnl.mean() / pnl.std() * np.sqrt(periods_per_year)
    cum = (1 + pnl).cumprod()
    total_ret = cum.iloc[-1] - 1
    print(
        f"  {label:25s}  Sharpe={sharpe:+6.2f}  "
        f"Total Return={total_ret:+8.2%}"
    )
    return {"sharpe": sharpe, "total_return": total_ret}


def main() -> None:
    df = yf.download(
        "GC=F",
        start="2020-01-01",
        end="2024-12-31",
        progress=False,
        auto_adjust=True,
    )

    # MA crossover signal
    df["ma_fast"] = df["Close"].rolling(20).mean()
    df["ma_slow"] = df["Close"].rolling(50).mean()
    df["signal"] = (df["ma_fast"] > df["ma_slow"]).astype(int)
    df["ret"] = df["Close"].pct_change()

    # === Phiên bản BUG (look-ahead) ===
    # Position ngày T dùng signal ngày T (signal tính từ close ngày T)
    # → Implicit assumption: ta vào lệnh tại close ngày T với signal vừa generated
    # → Nhưng close ngày T chỉ biết SAU khi nó đã đóng cửa → BUG
    df["pos_buggy"] = df["signal"]
    df["pnl_buggy"] = df["pos_buggy"] * df["ret"]

    # === Phiên bản ĐÚNG ===
    # Signal tính từ close ngày T → vào lệnh ngày T+1 → return ngày T+1
    # Cách đúng: shift signal 1 chu kỳ
    df["pos_correct"] = df["signal"].shift(1)
    df["pnl_correct"] = df["pos_correct"] * df["ret"]

    print("\n" + "=" * 70)
    print("DEMO: Cùng chiến lược MA(20,50) trên Gold 2020-2024")
    print("=" * 70)

    buggy_stats = quick_stats(df["pnl_buggy"], "Có look-ahead bug")
    correct_stats = quick_stats(df["pnl_correct"], "Đã shift(1) đúng")

    if buggy_stats and correct_stats:
        sharpe_inflation = buggy_stats["sharpe"] - correct_stats["sharpe"]
        return_inflation = (
            buggy_stats["total_return"] - correct_stats["total_return"]
        )
        print(f"\n{'='*70}")
        print(f"Mức độ phóng đại do look-ahead bug:")
        print(f"  Sharpe inflation:        +{sharpe_inflation:.2f}")
        print(f"  Total return inflation:  +{return_inflation:.2%}")
        print(f"{'='*70}")

    print(
        "\nQuy tắc vàng: trước khi tin bất kỳ backtest nào (kể cả tự code), "
        "grep code tìm '.shift('. Nếu không có shift → gần như chắc chắn "
        "có look-ahead bug. Nếu có shift → đếm xem shift đúng chỗ chưa."
    )


if __name__ == "__main__":
    main()
