"""
QuantCFD - Chapter 2 - Sharpe Statistical Significance
========================================================
Section 2.5: P-value và "bao nhiêu trades thì đủ".

Sharpe ratio in-sample là một estimate có sai số. Sample size nhỏ →
sai số lớn → không tin được Sharpe phản ánh edge thực.

Function sharpe_significance() trả về t-statistic, p-value, và minimum
Sharpe cần có để statistically significant tại sample size hiện tại.

Tham khảo: Andrew Lo, "The Statistics of Sharpe Ratios" (2002).

Chạy:
    python chapter-02/sharpe_significance.py
"""
import numpy as np
import pandas as pd
from scipy import stats


def sharpe_significance(
    daily_returns: pd.Series,
    periods_per_year: int = 252,
    confidence: float = 0.95,
) -> dict:
    """
    Tính Sharpe ratio và đánh giá statistical significance.

    Args:
        daily_returns: Series % thay đổi hàng ngày.
        periods_per_year: 252 (forex/indices/stocks), 365 (crypto 24/7).
        confidence: mức tin cậy mong muốn (default 0.95).

    Returns:
        dict gồm:
            n_observations, years, sharpe_annual, t_statistic,
            p_value, is_significant_95pct, min_sharpe_for_sig
    """
    r = daily_returns.dropna()
    n = len(r)
    if n < 30:
        return {"error": "Cần tối thiểu 30 observations để tính significance"}

    sharpe_daily = r.mean() / r.std()
    sharpe_annual = sharpe_daily * np.sqrt(periods_per_year)

    # T-statistic của Sharpe (xấp xỉ Lo 2002)
    # t-stat ≈ Sharpe_annual × √years
    years = n / periods_per_year
    t_stat = sharpe_annual * np.sqrt(years)

    # P-value 2 phía
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))

    # Min Sharpe để significance ở mức yêu cầu
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    min_sharpe_significant = t_crit / np.sqrt(years)

    return {
        "n_observations": n,
        "years": round(years, 2),
        "sharpe_annual": round(sharpe_annual, 3),
        "t_statistic": round(t_stat, 3),
        "p_value": round(p_value, 4),
        "is_significant_95pct": p_value < 0.05,
        "min_sharpe_for_sig": round(min_sharpe_significant, 3),
    }


def years_needed_for_sharpe(target_sharpe: float, confidence: float = 0.95) -> float:
    """
    Tính số năm cần thiết để Sharpe = target_sharpe đạt significance.

    Công thức ngược: t-stat = Sharpe × √years > t_crit
                    → years > (t_crit / Sharpe)²
    """
    t_crit = stats.norm.ppf((1 + confidence) / 2)  # ~1.96 cho 95%
    return (t_crit / target_sharpe) ** 2


def print_sharpe_table() -> None:
    """In bảng tham khảo: Sharpe vs năm cần thiết."""
    print(
        "\nBảng tham khảo: Sharpe in-sample vs SỐ NĂM cần để significant 95%"
    )
    print("=" * 60)
    print(f"  {'Sharpe':>8s}  {'Năm cần thiết':>15s}")
    print("-" * 60)
    for sharpe in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        years = years_needed_for_sharpe(sharpe)
        years_str = (
            f"{years:.1f} năm"
            if years >= 1
            else f"{years * 12:.1f} tháng"
        )
        print(f"  {sharpe:>8.1f}  {years_str:>15s}")


def print_report(label: str, report: dict) -> None:
    print(f"\n{label}:")
    for k, v in report.items():
        if isinstance(v, bool):
            check = "✓" if v else "✗"
            print(f"  {k:25s} = {v}  {check}")
        else:
            print(f"  {k:25s} = {v}")


def main() -> None:
    # Demo: 2 chiến lược cùng "true Sharpe" = 1.0 nhưng khác sample size
    np.random.seed(42)

    # Mỗi ngày return ~ N(0.0006, 0.01)
    # → Sharpe daily = 0.06, Sharpe annual = 0.06 × √252 ≈ 0.95
    short_strat = pd.Series(np.random.normal(0.0006, 0.01, 252))
    long_strat = pd.Series(np.random.normal(0.0006, 0.01, 1260))

    print("=" * 60)
    print("Demo: 2 chiến lược cùng 'true Sharpe' ~1.0, khác sample size")
    print("=" * 60)
    print_report("Chiến lược 1 NĂM (252 days)", sharpe_significance(short_strat))
    print_report("Chiến lược 5 NĂM (1260 days)", sharpe_significance(long_strat))

    print_sharpe_table()

    print(
        "\nKết luận: Sharpe 1.0 cần ~4 năm dữ liệu để statistically "
        "significant. Sharpe 0.5 cần ~16 năm. Đó là lý do retail cần ưu tiên "
        "chiến lược frequency cao — sample size lớn → thống kê có ý nghĩa nhanh."
    )


if __name__ == "__main__":
    main()
