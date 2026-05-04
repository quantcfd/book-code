"""
QuantCFD - Chapter 4 - Spread Session Analyzer
================================================
Section 4.3 + Bài tập 2: Phân tích spread theo giờ UTC.

Input: tick data với cột bid, ask (từ MT5 hoặc Dukascopy đã convert).
Output: bảng + heatmap của median spread theo giờ × ngày trong tuần.

Kết quả này là FILTER cho mọi chiến lược intraday: chỉ trade khi
spread thấp hơn ngưỡng — vì sao? Xem Chương 4 mở đầu (Câu chuyện Linh).

Chạy:
    python chapter-04/spread_session_analyzer.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def spread_by_hour(
    ticks_df: pd.DataFrame,
    pip_size: float = 1e-4,
    bid_col: str = "bid",
    ask_col: str = "ask",
) -> pd.DataFrame:
    """
    Median + statistics spread theo từng giờ UTC.

    Args:
        ticks_df: DataFrame với index=datetime UTC, có cột bid, ask.
        pip_size: 1e-4 cho EUR/USD-like, 1e-2 cho XAU/USD, 1e-2 cho JPY pairs.
        bid_col, ask_col: tên cột bid/ask.

    Returns:
        DataFrame với index=hour (0-23), columns=median/mean/p95/max.
    """
    df = ticks_df.copy()
    df["spread_pips"] = (df[ask_col] - df[bid_col]) / pip_size
    df["hour_utc"] = df.index.hour

    return df.groupby("hour_utc")["spread_pips"].agg(
        median="median",
        mean="mean",
        p95=lambda s: s.quantile(0.95),
        max="max",
        count="count",
    )


def spread_heatmap(
    ticks_df: pd.DataFrame,
    pip_size: float = 1e-4,
    save_path: str = "spread_heatmap.png",
) -> pd.DataFrame:
    """
    Vẽ heatmap median spread theo (day_of_week, hour_utc).

    Returns:
        Matrix 7x24 of median spreads.
    """
    df = ticks_df.copy()
    df["spread_pips"] = (df["ask"] - df["bid"]) / pip_size
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek  # 0=Mon, 6=Sun

    matrix = df.groupby(["dow", "hour"])["spread_pips"].median().unstack()

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(matrix.values, aspect="auto", cmap="RdYlGn_r", interpolation="nearest")

    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Day of week")
    ax.set_title("Median Spread Heatmap (pips)")

    # Annotate cells với value
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=7, color="black")

    plt.colorbar(im, ax=ax, label="Spread (pips)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved heatmap: {save_path}")

    return matrix


def identify_best_worst_hours(
    spread_table: pd.DataFrame, n: int = 3
) -> tuple:
    """Tìm n giờ có median spread thấp nhất/cao nhất."""
    sorted_by_median = spread_table["median"].sort_values()
    best = sorted_by_median.head(n)
    worst = sorted_by_median.tail(n)[::-1]
    return best, worst


def main() -> None:
    print(
        "Demo dùng synthetic data (vì chạy trên Linux không có MT5).\n"
        "Trên Windows + MT5: replace block dưới bằng:\n"
        "    from mt5_fetcher import init_mt5, fetch_mt5_ticks\n"
        "    init_mt5()\n"
        "    ticks = fetch_mt5_ticks('EURUSD', count=500000)\n"
    )

    # Synthetic ticks: simulate spread cao về đêm (Asia), thấp giờ London
    np.random.seed(42)
    n = 200_000
    timestamps = pd.date_range(
        start="2024-12-01", periods=n, freq="2s", tz="UTC"
    )
    hours = timestamps.hour

    # Base spread thấp giờ London/NY, cao giờ Asia
    base_spread = np.where(
        (hours >= 7) & (hours <= 16), 0.4,  # London
        np.where(
            (hours >= 13) & (hours <= 21), 0.5,  # NY
            np.where(
                (hours >= 22) | (hours <= 6), 2.0,  # Asia + rollover
                0.6,
            )
        )
    )
    # Spike at rollover hour 21-22
    base_spread = np.where((hours >= 21) & (hours <= 22), 4.5, base_spread)
    # Random noise
    spread_pips = np.maximum(0.1, base_spread + np.random.normal(0, 0.15, n))

    bid = 1.05 + np.random.normal(0, 0.001, n)
    ask = bid + spread_pips * 1e-4

    ticks = pd.DataFrame({"bid": bid, "ask": ask}, index=timestamps)
    ticks.index.name = "time"

    print(f"\nSynthetic ticks: {len(ticks):,} samples\n")

    # 1. Spread by hour
    table = spread_by_hour(ticks)
    print("Median spread by hour UTC (top 10 rows):")
    print(table.head(10).round(2))

    # 2. Best/worst hours
    best, worst = identify_best_worst_hours(table, n=3)
    print(f"\n3 giờ spread THẤP nhất:")
    for hour, val in best.items():
        print(f"  {hour:02d}:00 UTC  →  {val:.2f} pips")
    print(f"\n3 giờ spread CAO nhất:")
    for hour, val in worst.items():
        print(f"  {hour:02d}:00 UTC  →  {val:.2f} pips")

    # 3. Heatmap
    print("\nGenerating heatmap...")
    spread_heatmap(ticks)

    print(
        "\nKết luận: kết quả này → filter cho mọi chiến lược intraday. "
        "Ví dụ: 'chỉ trade EURUSD khi spread < 0.8 pip' loại bỏ giờ Asia + rollover."
    )


if __name__ == "__main__":
    main()
