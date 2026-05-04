"""
QuantCFD - Chapter 3 - 5 Essential Matplotlib Charts
=====================================================
Section 3.4: 5 loại biểu đồ trader cần biết.

Tạo 5 chart lưu thành file PNG, mỗi cái minh hoạ một use case khác nhau.

Chạy:
    python chapter-03/matplotlib_5_charts.py
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Optional: seaborn cho heatmap đẹp hơn
try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def chart_1_line_equity(df: pd.DataFrame, save_path: str = "chart1_equity.png") -> None:
    """Line chart — equity curve, trader's #1 most-used chart."""
    fig, ax = plt.subplots(figsize=(12, 5))
    df["Equity"].plot(ax=ax, color="steelblue", lw=1.5, label="Strategy")
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax.set_title("Equity Curve — XAUUSD MA(20,50) Strategy")
    ax.set_ylabel("Equity (start = 1.0)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Chart 1 (line): {save_path}")


def chart_2_histogram(df: pd.DataFrame, save_path: str = "chart2_histogram.png") -> None:
    """Histogram — distribution of returns."""
    fig, ax = plt.subplots(figsize=(10, 5))
    df["Return"].dropna().hist(bins=80, ax=ax, color="steelblue", alpha=0.7)
    ax.axvline(
        df["Return"].mean(), color="red", ls="--", label=f"Mean ({df['Return'].mean():.4%})"
    )
    ax.set_title("Distribution of Daily Returns")
    ax.set_xlabel("Daily return")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Chart 2 (histogram): {save_path}")


def chart_3_scatter(save_path: str = "chart3_scatter.png") -> None:
    """Scatter — correlation 2 instruments."""
    btc = yf.download(
        "BTC-USD",
        start="2022-01-01",
        end="2024-12-31",
        progress=False,
        auto_adjust=True,
    )["Close"].pct_change()
    eth = yf.download(
        "ETH-USD",
        start="2022-01-01",
        end="2024-12-31",
        progress=False,
        auto_adjust=True,
    )["Close"].pct_change()

    aligned = pd.concat([btc, eth], axis=1).dropna()
    aligned.columns = ["BTC", "ETH"]
    corr = aligned["BTC"].corr(aligned["ETH"])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(aligned["BTC"], aligned["ETH"], alpha=0.4, s=12)
    ax.set_xlabel("BTC daily return")
    ax.set_ylabel("ETH daily return")
    ax.set_title(f"BTC vs ETH — corr = {corr:.3f}")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Chart 3 (scatter): {save_path}")


def chart_4_heatmap(save_path: str = "chart4_heatmap.png") -> None:
    """Heatmap — correlation matrix nhiều instruments."""
    tickers = ["BTC-USD", "ETH-USD", "GC=F", "ES=F", "DX-Y.NYB"]
    names = ["BTC", "ETH", "GOLD", "US500", "DXY"]

    rets = pd.DataFrame()
    for t, n in zip(tickers, names):
        rets[n] = yf.download(
            t,
            start="2022-01-01",
            end="2024-12-31",
            progress=False,
            auto_adjust=True,
        )["Close"].pct_change()

    corr = rets.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    if HAS_SEABORN:
        sns.heatmap(
            corr,
            annot=True,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax,
            fmt=".2f",
        )
    else:
        # Fallback dùng matplotlib thuần
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(
                    j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center"
                )
        plt.colorbar(im, ax=ax)

    ax.set_title("Correlation Matrix — 5 CFD Instruments (2022-2024)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Chart 4 (heatmap): {save_path}")


def chart_5_subplot_grid(
    df: pd.DataFrame, save_path: str = "chart5_subplots.png"
) -> None:
    """Subplot grid — multi-panel comparison cho strategy report."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Top-left: price
    df["Close"].plot(ax=axes[0, 0], title="Price", color="steelblue")
    axes[0, 0].grid(alpha=0.3)

    # Top-right: returns histogram
    df["Return"].dropna().hist(bins=60, ax=axes[0, 1], color="steelblue", alpha=0.7)
    axes[0, 1].set_title("Returns distribution")
    axes[0, 1].grid(alpha=0.3)

    # Bottom-left: equity curve
    df["Equity"].plot(ax=axes[1, 0], title="Strategy equity curve", color="green")
    axes[1, 0].axhline(1.0, color="gray", ls="--", alpha=0.5)
    axes[1, 0].grid(alpha=0.3)

    # Bottom-right: rolling vol
    df["Vol20"].plot(
        ax=axes[1, 1], title="20-day rolling annualized vol", color="darkorange"
    )
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Chart 5 (subplots): {save_path}")


def main() -> None:
    # Load data Gold để demo
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
    df["Vol20"] = df["Return"].rolling(20).std() * np.sqrt(252)
    df["Signal"] = np.where(df["MA20"] > df["MA50"], 1, -1)
    df["Position"] = df["Signal"].shift(1)
    df["StratRet"] = df["Position"] * df["Return"]
    df["Equity"] = (1 + df["StratRet"]).cumprod()

    print("\nGenerating 5 essential trader charts:\n")
    chart_1_line_equity(df)
    chart_2_histogram(df)
    chart_3_scatter()
    chart_4_heatmap()
    chart_5_subplot_grid(df)
    print("\nAll 5 charts saved as PNG in current directory.")
    print(
        "\nHọc thuộc 5 loại này — đủ cho 95% nhu cầu visualization của quant retail. "
        "Đừng học hết matplotlib, vô ích."
    )


if __name__ == "__main__":
    main()
