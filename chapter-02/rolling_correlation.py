"""
QuantCFD - Chapter 2 - Rolling Correlation
===========================================
Section 2.4: Tương quan thay đổi theo thời gian.

Demo: cùng cặp asset, correlation có thể đảo chiều hoàn toàn trong vòng
6 tháng. Đây là lý do "diversification biến mất khi bạn cần nó nhất".

Chạy:
    python chapter-02/rolling_correlation.py
"""
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def compute_rolling_correlations(
    returns: pd.DataFrame,
    pairs: list,
    window: int = 90,
) -> pd.DataFrame:
    """
    Tính rolling correlation cho danh sách pairs.

    Args:
        returns: DataFrame, mỗi cột là returns của 1 asset.
        pairs: list of tuples (asset_a, asset_b).
        window: số ngày rolling.

    Returns:
        DataFrame với cột là tên pair, value là rolling corr.
    """
    rolling_corr = pd.DataFrame(index=returns.index)
    for a, b in pairs:
        col_name = f"{a} vs {b}"
        rolling_corr[col_name] = returns[a].rolling(window).corr(returns[b])
    return rolling_corr


def print_corr_summary(rolling_corr: pd.DataFrame, window: int) -> None:
    """In bảng min, mean, max của mỗi pair."""
    print(f"\nRolling correlation ({window} ngày):\n")
    print(
        f"{'Pair':18s}  {'Min':>8s}  {'Mean':>8s}  {'Max':>8s}  {'Range':>8s}"
    )
    print("-" * 58)
    for col in rolling_corr.columns:
        s = rolling_corr[col].dropna()
        if len(s) == 0:
            continue
        print(
            f"{col:18s}  {s.min():+8.3f}  {s.mean():+8.3f}  "
            f"{s.max():+8.3f}  {s.max() - s.min():>8.3f}"
        )


def plot_rolling_correlations(
    rolling_corr: pd.DataFrame,
    window: int,
    save_path: str = None,
) -> None:
    """Vẽ rolling correlations."""
    fig, ax = plt.subplots(figsize=(12, 6))
    rolling_corr.plot(ax=ax, lw=1.2)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.axhline(0.7, color="red", ls=":", alpha=0.4, label="|ρ|=0.7")
    ax.axhline(-0.7, color="red", ls=":", alpha=0.4)
    ax.set_title(f"Rolling {window}-day correlation — chú ý sự biến động")
    ax.set_ylabel("Correlation")
    ax.set_ylim(-1, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"\n  Saved plot to: {save_path}")
    plt.show()


def main() -> None:
    tickers = {
        "BTC": "BTC-USD",
        "NDX": "^NDX",
        "GOLD": "GC=F",
        "DXY": "DX-Y.NYB",
    }

    data = {}
    for name, ticker in tickers.items():
        df = yf.download(
            ticker,
            start="2020-01-01",
            end="2024-12-31",
            progress=False,
            auto_adjust=True,
        )
        data[name] = df["Close"].pct_change()

    returns = pd.DataFrame(data).dropna()

    pairs = [
        ("BTC", "NDX"),
        ("GOLD", "DXY"),
        ("BTC", "GOLD"),
    ]

    window = 90
    rolling_corr = compute_rolling_correlations(returns, pairs, window=window)

    print_corr_summary(rolling_corr, window=window)
    plot_rolling_correlations(
        rolling_corr, window=window, save_path="rolling_correlation.png"
    )

    print(
        "\nKết luận: BTC-NDX correlation từng ở khoảng -0.1 (đầu COVID) "
        "lên +0.8 (2022-2023). Cùng một cặp asset, correlation đảo chiều "
        "hoàn toàn trong vòng vài tháng. Đừng bao giờ tin con số tĩnh."
    )


if __name__ == "__main__":
    main()
