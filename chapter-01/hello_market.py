"""
QuantCFD - Chapter 1 - Hello Market
====================================
Bài tập 3 / Chương 1: Setup môi trường + first script

Mục đích:
    Lấy dữ liệu BTC, XAUUSD (Gold), US500 (SPX futures) và in thống kê
    cơ bản. Kiểm tra môi trường Python + pandas + yfinance đã hoạt động.

Output kỳ vọng:
    3 dòng số. Mỗi dòng là một thị trường với annual return, volatility,
    Sharpe, max drawdown.

Chạy:
    python chapter-01/hello_market.py
"""
import numpy as np
import pandas as pd
import yfinance as yf


def stats(name: str, returns: pd.Series, periods_per_year: int = 252) -> None:
    """
    In thống kê cơ bản của một series returns.

    Args:
        name: Tên hiển thị (vd: 'BTC', 'XAUUSD').
        returns: Series % thay đổi hàng ngày.
        periods_per_year: 252 cho stocks/forex/indices, 365 cho crypto 24/7.
    """
    r = returns.dropna()
    if len(r) == 0:
        print(f"{name:8s} | No data")
        return

    ann_return = r.mean() * periods_per_year
    ann_vol = r.std() * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    cum = (1 + r).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()

    print(
        f"{name:8s} | Annual Return: {ann_return:7.2%} | "
        f"Annual Vol: {ann_vol:6.2%} | Sharpe: {sharpe:5.2f} | "
        f"Max DD: {max_dd:7.2%}"
    )


def main() -> None:
    start, end = "2023-01-01", "2024-12-31"

    # ---- Crypto: BTC ----
    btc = yf.download(
        "BTC-USD", start=start, end=end, progress=False, auto_adjust=True
    )
    btc["Returns"] = btc["Close"].pct_change()

    # ---- Commodity: Gold (proxy cho XAUUSD CFD) ----
    gold = yf.download(
        "GC=F", start=start, end=end, progress=False, auto_adjust=True
    )
    gold["Returns"] = gold["Close"].pct_change()

    # ---- Indices: S&P 500 futures (proxy cho US500 CFD) ----
    spx = yf.download(
        "ES=F", start=start, end=end, progress=False, auto_adjust=True
    )
    spx["Returns"] = spx["Close"].pct_change()

    # In header
    print(
        f"\n{'Asset':8s} | {'Annual Return':14s} | {'Annual Vol':12s} | "
        f"{'Sharpe':6s} | {'Max DD':10s}"
    )
    print("-" * 78)

    # In từng dòng (chú ý: BTC dùng 365 vì trade 24/7, hai cái còn lại 252)
    stats("BTC", btc["Returns"], periods_per_year=365)
    stats("XAUUSD", gold["Returns"], periods_per_year=252)
    stats("US500", spx["Returns"], periods_per_year=252)

    print(
        "\nIntuition đầu tiên: 3 thị trường có characteristics khác nhau. "
        "Một chiến lược chạy tốt trên BTC chưa chắc chạy tốt trên XAUUSD."
    )


if __name__ == "__main__":
    main()
