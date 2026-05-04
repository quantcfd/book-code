"""
QuantCFD - Chapter 4 - Contract Rollover Detector
====================================================
Section 4.4: Phát hiện gap không phải do thị trường mà do contract roll.

CFD trên indices/commodity dựa trên futures. Mỗi 1-3 tháng có roll →
giá có thể "gap" 0.5-2% mà không phản ánh move thực.

Function này flag các ngày khả nghi để anh em manual check.

Chạy:
    python chapter-04/rollover_detector.py
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf


def detect_rollover_gaps(
    df: pd.DataFrame,
    threshold: float = 0.005,
    open_col: str = "Open",
    close_col: str = "Close",
) -> pd.Series:
    """
    Phát hiện ngày có overnight gap > threshold (default 0.5%).

    Args:
        df: DataFrame với cột Open, Close, index=date.
        threshold: % gap để flag (0.005 = 0.5%).
        open_col, close_col: tên cột.

    Returns:
        Series với index=date, value=overnight_change_pct (chỉ những ngày flagged).
    """
    overnight_change = df[open_col] / df[close_col].shift(1) - 1
    return overnight_change[abs(overnight_change) > threshold]


def classify_gaps(
    df: pd.DataFrame,
    rollover_dates: pd.Series,
    intraday_threshold: float = 0.005,
) -> pd.DataFrame:
    """
    Phân loại các gap thành "có thể là rollover" vs "move thị trường thực".

    Logic: nếu intraday range nhỏ so với overnight gap → khả năng cao là rollover.
           Nếu intraday range cũng lớn → có thể move thực do news.

    Returns:
        DataFrame với columns: date, gap_pct, intraday_range_pct, likely_cause.
    """
    rollover_df = df.loc[rollover_dates.index].copy()
    rollover_df["gap_pct"] = rollover_dates
    rollover_df["intraday_range_pct"] = (
        (rollover_df["High"] - rollover_df["Low"]) / rollover_df["Close"]
    )
    rollover_df["likely_cause"] = rollover_df.apply(
        lambda row: (
            "ROLLOVER (intraday quiet)"
            if abs(row["gap_pct"]) > 2 * row["intraday_range_pct"]
            else "MARKET MOVE (intraday active)"
        ),
        axis=1,
    )
    return rollover_df[["gap_pct", "intraday_range_pct", "likely_cause"]]


def main() -> None:
    print("Loading US500 (ES futures proxy) data...")
    spx = yf.download(
        "ES=F",
        start="2020-01-01",
        end="2024-12-31",
        progress=False,
        auto_adjust=True,
    )
    print(f"Loaded {len(spx)} days")

    # Detect gaps > 0.5% overnight
    rollover_dates = detect_rollover_gaps(spx, threshold=0.005)
    print(f"\nDetected {len(rollover_dates)} potential rollover gaps (>0.5%)")

    if len(rollover_dates) > 0:
        # Classify
        classified = classify_gaps(spx, rollover_dates)
        print("\nTop 10 largest gaps with classification:")
        top = classified.reindex(
            classified["gap_pct"].abs().sort_values(ascending=False).index
        ).head(10)
        for date, row in top.iterrows():
            print(
                f"  {date.strftime('%Y-%m-%d')}  "
                f"gap={row['gap_pct']:+.3%}  "
                f"intraday_range={row['intraday_range_pct']:.3%}  "
                f"→ {row['likely_cause']}"
            )

    print(
        "\nLưu ý: rollover detection là heuristic — không 100% accurate. "
        "Cách robust nhất là dùng continuous-adjusted feed từ TwelveData "
        "(symbol như 'ES1!') hoặc broker đã pre-adjust."
    )


if __name__ == "__main__":
    main()
