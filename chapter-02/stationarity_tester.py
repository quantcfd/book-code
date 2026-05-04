"""
QuantCFD - Chapter 2 - Stationarity Tester
===========================================
Section 2.3: Stationarity — vì sao chiến lược "ngừng chạy".

ADF test để check stationarity:
    H0: chuỗi có unit root (= không stationary)
    p-value < 0.05 → reject H0 → chuỗi stationary

Demo:
    1. PRICE thường KHÔNG stationary
    2. RETURNS thường stationary
    3. Spread của 2 asset cointegrated CÓ THỂ stationary (nền tảng pair trading)
    4. Rolling ADF: stationarity không cố định theo thời gian

Chạy:
    python chapter-02/stationarity_tester.py
"""
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller


def adf_report(series: pd.Series, name: str) -> dict:
    """
    Chạy ADF test và in kết quả dễ hiểu.

    Returns:
        dict với adf_stat, p_value, is_stationary.
    """
    series = series.dropna()
    if len(series) < 50:
        print(f"  {name:30s}  Không đủ data (cần ≥50 obs)")
        return {}

    result = adfuller(series, autolag="AIC")
    stat, pval = result[0], result[1]
    is_stationary = pval < 0.05
    flag = "✓ STATIONARY    " if is_stationary else "✗ NON-STATIONARY"

    print(
        f"  {name:30s}  ADF stat={stat:+7.3f}  "
        f"p-value={pval:.4f}  {flag}"
    )

    return {
        "name": name,
        "adf_stat": stat,
        "p_value": pval,
        "is_stationary": is_stationary,
    }


def rolling_adf_check(series: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling ADF test — kiểm tra stationarity có ổn định theo thời gian không.

    Returns:
        pd.Series rolling p-values.
    """
    s = series.dropna()

    def safe_adf(x):
        x = x.dropna()
        if len(x) < 50:
            return np.nan
        try:
            return adfuller(x)[1]
        except Exception:
            return np.nan

    return s.rolling(window).apply(safe_adf, raw=False)


def main() -> None:
    start, end = "2020-01-01", "2024-12-31"

    btc = yf.download(
        "BTC-USD", start=start, end=end, progress=False, auto_adjust=True
    )["Close"]
    eth = yf.download(
        "ETH-USD", start=start, end=end, progress=False, auto_adjust=True
    )["Close"]
    gold = yf.download(
        "GC=F", start=start, end=end, progress=False, auto_adjust=True
    )["Close"]

    # Test 1: PRICE — kỳ vọng non-stationary
    print("\n" + "=" * 70)
    print("TEST 1 — PRICE (kỳ vọng: NON-STATIONARY)")
    print("=" * 70)
    adf_report(btc, "BTC price")
    adf_report(eth, "ETH price")
    adf_report(gold, "Gold price")

    # Test 2: RETURNS — kỳ vọng stationary
    print("\n" + "=" * 70)
    print("TEST 2 — RETURNS (kỳ vọng: STATIONARY)")
    print("=" * 70)
    adf_report(btc.pct_change(), "BTC returns")
    adf_report(eth.pct_change(), "ETH returns")
    adf_report(gold.pct_change(), "Gold returns")

    # Test 3: ETH/BTC ratio — cointegration check
    print("\n" + "=" * 70)
    print("TEST 3 — RATIO ETH/BTC (cointegration check cho pair trade)")
    print("=" * 70)
    # Align indices first
    aligned = pd.concat([eth, btc], axis=1).dropna()
    aligned.columns = ["ETH", "BTC"]
    ratio = aligned["ETH"] / aligned["BTC"]
    adf_report(ratio, "ETH/BTC ratio")

    # Test 4: Rolling ADF — stationarity stability
    print("\n" + "=" * 70)
    print("TEST 4 — ROLLING ADF của BTC returns (window 252 ngày)")
    print("=" * 70)
    btc_ret = btc.pct_change()
    rolling_pval = rolling_adf_check(btc_ret, window=252).dropna()
    print(f"  Min p-value:       {rolling_pval.min():.4f}")
    print(f"  Max p-value:       {rolling_pval.max():.4f}")
    print(f"  Mean p-value:      {rolling_pval.mean():.4f}")
    print(
        f"  % windows stationary (<0.05): {(rolling_pval < 0.05).mean():.1%}"
    )

    print(
        "\nKết luận: Stationarity không phải tính chất TĨNH. Nó thay đổi theo "
        "regime. Mọi pair trading strategy cần rolling ADF check + auto "
        "kill-switch khi p-value vượt threshold."
    )


if __name__ == "__main__":
    main()
