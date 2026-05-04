"""
QuantCFD - Chapter 3 - Ten Lines Pandas Replace 100 Lines VBA
================================================================
Section 3.5: 5 task quant retail thường làm trên Excel + VBA, viết
lại bằng pandas. Mỗi task ở dạng "trước-sau" để minh hoạ sức mạnh.

Chạy:
    python chapter-03/ten_lines_pandas.py
"""
import numpy as np
import pandas as pd
import yfinance as yf


def setup_data() -> pd.DataFrame:
    """Tải 5 năm Gold + tính returns, position, returns chiến lược."""
    df = yf.download(
        "GC=F",
        start="2020-01-01",
        end="2024-12-31",
        progress=False,
        auto_adjust=True,
    )
    df["Return"] = df["Close"].pct_change()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["Position"] = np.where(df["MA20"] > df["MA50"], 1, -1)
    df["Position"] = df["Position"].shift(1)
    df["StratRet"] = df["Position"] * df["Return"]
    return df


def task_1_rolling_sharpe(df: pd.DataFrame) -> pd.Series:
    """
    Task 1: Rolling 60-day Sharpe ratio.
    VBA: ~25 dòng. Pandas: 2 dòng.
    """
    window = 60
    rolling_sharpe = (
        df["Return"].rolling(window).mean()
        / df["Return"].rolling(window).std()
        * np.sqrt(252)
    )
    return rolling_sharpe


def task_2_filter_extreme_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Task 2: Lọc ngày có return > 2% hoặc < -3% trong tháng 6/2024.
    VBA: AutoFilter + custom criteria + copy. Pandas: 1 dòng.
    """
    mask = (
        (df.index.year == 2024)
        & (df.index.month == 6)
        & ((df["Return"] > 0.02) | (df["Return"] < -0.03))
    )
    return df[mask]


def task_3_weekly_pnl(df: pd.DataFrame) -> pd.Series:
    """
    Task 3: Tổng PnL theo tuần.
    VBA: pivot table + date grouping. Pandas: 1 dòng.
    """
    return df["StratRet"].resample("W").sum()


def task_4_longest_streak(df: pd.DataFrame) -> dict:
    """
    Task 4: Tìm chuỗi thắng/thua dài nhất.
    VBA: vòng for đếm consecutive ~30 dòng, dễ off-by-one. Pandas: 4 dòng.
    """
    wins = (df["StratRet"] > 0).astype(int)
    losses = (df["StratRet"] < 0).astype(int)

    # Identify groups of consecutive 1s (or 0s)
    win_groups = (wins != wins.shift()).cumsum()
    loss_groups = (losses != losses.shift()).cumsum()

    win_streaks = wins.groupby(win_groups).sum()
    loss_streaks = losses.groupby(loss_groups).sum()

    return {
        "longest_win_streak": int(win_streaks.max()),
        "longest_loss_streak": int(loss_streaks.max()),
    }


def task_5_top_months(df: pd.DataFrame, top_n: int = 5) -> pd.Series:
    """
    Task 5: Rank top N tháng tốt nhất theo total return.
    VBA: nhiều bước với pivot + sort. Pandas: 2 dòng.
    """
    monthly_ret = df["Return"].resample("M").apply(lambda x: (1 + x).prod() - 1)
    return monthly_ret.nlargest(top_n)


def task_6_cross_sectional_rank(top_n: int = 3) -> pd.Series:
    """
    Task 6 (bonus): Rank top N instruments theo Sharpe 6 tháng gần nhất.
    Đây là task cross-sectional ranking — phổ biến trong momentum strategies.
    """
    tickers = {
        "BTC": "BTC-USD",
        "ETH": "ETH-USD",
        "GOLD": "GC=F",
        "US500": "ES=F",
        "DXY": "DX-Y.NYB",
        "OIL": "CL=F",
    }

    rets = pd.DataFrame()
    for name, ticker in tickers.items():
        rets[name] = yf.download(
            ticker,
            start="2024-01-01",
            end="2024-12-31",
            progress=False,
            auto_adjust=True,
        )["Close"].pct_change()

    recent = rets.tail(126)  # ~6 tháng
    sharpes = recent.mean() / recent.std() * np.sqrt(252)
    return sharpes.nlargest(top_n)


def main() -> None:
    print("\nLoading data...")
    df = setup_data()
    print(f"Loaded {len(df)} rows of GC=F data")

    print("\n" + "=" * 70)
    print(" TASK 1: Rolling 60-day Sharpe (2 dòng pandas vs ~25 dòng VBA)")
    print("=" * 70)
    rolling_sharpe = task_1_rolling_sharpe(df)
    print(f"Rolling Sharpe (5 cuối):")
    print(rolling_sharpe.dropna().tail().apply(lambda x: f"{x:+.3f}"))

    print("\n" + "=" * 70)
    print(" TASK 2: Lọc ngày extreme tháng 6/2024 (1 dòng vs ~10 thao tác Excel)")
    print("=" * 70)
    extreme = task_2_filter_extreme_days(df)
    if len(extreme) > 0:
        print(f"Found {len(extreme)} extreme days:")
        print(extreme[["Close", "Return"]].apply(
            lambda x: x.apply(lambda v: f"{v:.4f}") if x.name == "Return" else x
        ))
    else:
        print("Không có ngày nào extreme trong tháng 6/2024.")

    print("\n" + "=" * 70)
    print(" TASK 3: Tổng PnL theo tuần (1 dòng vs pivot table)")
    print("=" * 70)
    weekly = task_3_weekly_pnl(df)
    print(f"Weekly PnL (5 tuần cuối):")
    print(weekly.tail().apply(lambda x: f"{x:+.4%}"))

    print("\n" + "=" * 70)
    print(" TASK 4: Chuỗi thắng/thua dài nhất (4 dòng vs ~30 dòng VBA)")
    print("=" * 70)
    streaks = task_4_longest_streak(df)
    print(f"Longest winning streak:  {streaks['longest_win_streak']} ngày")
    print(f"Longest losing streak:   {streaks['longest_loss_streak']} ngày")

    print("\n" + "=" * 70)
    print(" TASK 5: Top 5 tháng tốt nhất (2 dòng vs nhiều thao tác Excel)")
    print("=" * 70)
    top_months = task_5_top_months(df, top_n=5)
    print("Top 5 best months by total return:")
    for date, ret in top_months.items():
        print(f"  {date.strftime('%Y-%m'):10s}  {ret:+.2%}")

    print("\n" + "=" * 70)
    print(" TASK 6 BONUS: Cross-sectional ranking 6 instruments (~5 dòng)")
    print("=" * 70)
    print("Top 3 instruments by 6-month Sharpe:")
    top_instruments = task_6_cross_sectional_rank(top_n=3)
    for name, sharpe in top_instruments.items():
        print(f"  {name:6s}  Sharpe = {sharpe:+.2f}")

    print(
        "\n6 tasks. Tổng cộng ~15 dòng pandas thuần. Nếu viết bằng VBA: "
        "~150-200 dòng và chậm gấp nhiều lần. Đây là lý do không quay lại Excel."
    )


if __name__ == "__main__":
    main()
