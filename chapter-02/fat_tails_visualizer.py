"""
QuantCFD - Chapter 2 - Fat Tails Visualizer
============================================
Section 2.2: Phân phối lợi nhuận thực tế — Fat tails.

Chứng minh thực nghiệm rằng returns thị trường KHÔNG phân phối Normal:
  - Kurtosis ~10-15 (Normal = 0)
  - Số ±5σ events nhiều gấp 80-150 lần Normal predicts
  - Skewness âm/dương khác nhau theo asset class

So sánh trực quan bằng 2 plot: linear scale vs log scale.

Chạy:
    python chapter-02/fat_tails_visualizer.py
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats


def analyze_fat_tails(
    returns: pd.Series,
    name: str = "Asset",
) -> dict:
    """
    Phân tích fat tails của một series returns.

    Returns:
        dict gồm mean, std, skewness, kurtosis, và bảng counts ±kσ events.
    """
    r = returns.dropna()
    mu, sigma = r.mean(), r.std()
    skew = stats.skew(r)
    kurt = stats.kurtosis(r)  # excess kurtosis (Normal = 0)

    print(f"\n{name} daily returns:")
    print(f"  N observations: {len(r):,}")
    print(f"  Mean (μ):      {mu:.4%}")
    print(f"  Std  (σ):      {sigma:.4%}")
    print(f"  Skewness:      {skew:+.3f}   (Normal = 0)")
    print(f"  Excess kurt:   {kurt:+.3f}   (Normal = 0; >3 = fat tails)")

    print(f"\n  ±k σ events:")
    print(f"  {'k':<3s} {'Actual':>8s} {'Normal':>10s} {'Ratio':>8s}")
    print(f"  {'-'*32}")
    sigma_events = {}
    for k in [3, 4, 5, 6]:
        actual = int((np.abs(r - mu) > k * sigma).sum())
        expected_normal = len(r) * 2 * (1 - stats.norm.cdf(k))
        ratio = actual / max(expected_normal, 0.01)
        sigma_events[k] = (actual, expected_normal, ratio)
        print(f"  {k:<3d} {actual:>8d} {expected_normal:>10.2f} {ratio:>7.1f}x")

    return {
        "mean": mu,
        "std": sigma,
        "skewness": skew,
        "excess_kurtosis": kurt,
        "sigma_events": sigma_events,
    }


def plot_distribution(returns: pd.Series, name: str, save_path: str = None) -> None:
    """Vẽ histogram thực tế vs Normal fitted, ở 2 scale (linear + log)."""
    r = returns.dropna()
    mu, sigma = r.mean(), r.std()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.linspace(r.min(), r.max(), 200)

    # Linear scale
    axes[0].hist(
        r, bins=80, density=True, alpha=0.6, color="steelblue", label=f"{name} actual"
    )
    axes[0].plot(
        x, stats.norm.pdf(x, mu, sigma), "r-", lw=2, label="Normal fitted"
    )
    axes[0].set_title(f"{name} — Linear scale (fat tails khó nhìn)")
    axes[0].set_xlabel("Daily return")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Log scale (fat tails hiện rõ)
    axes[1].hist(
        r, bins=80, density=True, alpha=0.6, color="steelblue", label=f"{name} actual"
    )
    axes[1].plot(
        x, stats.norm.pdf(x, mu, sigma), "r-", lw=2, label="Normal fitted"
    )
    axes[1].set_yscale("log")
    axes[1].set_title(f"{name} — Log scale (fat tails hiện rõ ở 2 đuôi)")
    axes[1].set_xlabel("Daily return")
    axes[1].set_ylabel("Density (log)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"\n  Saved plot to: {save_path}")
    plt.show()


def main() -> None:
    # 8 năm BTC data
    btc = yf.download(
        "BTC-USD",
        start="2017-01-01",
        end="2024-12-31",
        progress=False,
        auto_adjust=True,
    )
    returns = btc["Close"].pct_change().dropna()

    analyze_fat_tails(returns, name="BTC")
    plot_distribution(returns, name="BTC", save_path="btc_fat_tails.png")

    print(
        "\nKết luận: BTC daily returns hoàn toàn KHÔNG là Normal. "
        "±5σ events xảy ra nhiều gấp ~100x so với Normal predicts. "
        "Mọi chiến lược dựa trên giả định Normal sẽ cháy ở tail event."
    )


if __name__ == "__main__":
    main()
