"""
QuantCFD - Chapter 4 - Binance Funding Rate Analyzer
======================================================
Section 4.2: Funding rate cho perpetual futures.

Funding 0.01% mỗi 8h = 10.95%/năm. Hold long perpetual khi funding dương
mà không tính cost = ignore một khoản chi phí lớn.

Yêu cầu:
    pip install ccxt

Chạy:
    python chapter-04/binance_funding.py
"""
import ccxt
import pandas as pd


def fetch_funding_rate_history(
    symbol: str = "BTC/USDT:USDT",
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch funding rate history cho perpetual contract.

    Args:
        symbol: ccxt format cho perpetual: "BTC/USDT:USDT".
        limit: số records (Binance max 1000).

    Returns:
        DataFrame với cột: datetime, funding_rate, mark_price.
    """
    exchange = ccxt.binance({"options": {"defaultType": "swap"}})
    funding = exchange.fetch_funding_rate_history(symbol, limit=limit)

    df = pd.DataFrame(funding)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df.set_index("datetime", inplace=True)
    return df


def analyze_funding(df: pd.DataFrame) -> dict:
    """Tính các thống kê funding rate quan trọng."""
    fr = df["fundingRate"].dropna()
    n = len(fr)

    if n == 0:
        return {}

    # 3 funding events/day → annualization factor
    annual_factor = 365 * 3

    return {
        "n_observations": n,
        "mean_per_event": fr.mean(),
        "median_per_event": fr.median(),
        "max_per_event": fr.max(),
        "min_per_event": fr.min(),
        "pct_positive": (fr > 0).mean(),
        "annualized_long_cost": fr.mean() * annual_factor,  # long pays when positive
        "annualized_short_yield": -fr.mean() * annual_factor,  # short receives
    }


def main() -> None:
    print("Fetching BTC perpetual funding history...")
    df = fetch_funding_rate_history("BTC/USDT:USDT", limit=1000)
    print(f"Loaded {len(df)} funding events")

    if len(df) == 0:
        print("No data returned. Check API or try again.")
        return

    stats = analyze_funding(df)
    print(f"\n{'='*55}")
    print("BTC PERPETUAL FUNDING RATE ANALYSIS")
    print("=" * 55)
    print(f"  Observations:           {stats['n_observations']:,}")
    print(f"  % positive funding:     {stats['pct_positive']:.1%}")
    print(f"  Mean funding/event:     {stats['mean_per_event']:+.5%}")
    print(f"  Median funding/event:   {stats['median_per_event']:+.5%}")
    print(f"  Max funding/event:      {stats['max_per_event']:+.5%}")
    print(f"  Min funding/event:      {stats['min_per_event']:+.5%}")
    print(f"\n  Annualized long cost:   {stats['annualized_long_cost']:+.2%}")
    print(f"  Annualized short yield: {stats['annualized_short_yield']:+.2%}")

    print(
        "\nIntuition: nếu hold long BTC perpetual quanh năm khi funding dương "
        "trung bình, anh em mất ~10% NĂM chỉ riêng cho funding. Đó là lý do "
        "carry trade (long spot + short perp) là một edge thực có cơ chế kinh tế."
    )


if __name__ == "__main__":
    main()
