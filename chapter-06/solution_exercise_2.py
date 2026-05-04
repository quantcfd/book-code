"""
Bài 2 — Tear sheet generator demo trên 3 strategies

Generate 3 PDFs cho XAUUSD MA crossover, EURUSD breakout, BTC trend.
So sánh side-by-side và quyết định nên live trade strategy nào.
"""
import numpy as np
import pandas as pd
from tear_sheet import create_tear_sheet
from metrics import sharpe_ratio, max_drawdown, calmar_ratio


def make_synthetic_strategy(name: str, mu: float, sigma: float, n: int, seed: int):
    np.random.seed(seed)
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    returns = pd.Series(np.random.normal(mu, sigma, n), index=dates)

    # Build trades từ returns (simplified)
    n_trades = n // 5   # ~1 trade / 5 days
    trade_pnls = np.random.normal(mu * 5 * 100, sigma * np.sqrt(5) * 100, n_trades)
    trades = pd.DataFrame({
        'pnl': trade_pnls,
        'duration_days': np.random.randint(1, 10, n_trades),
    })
    return returns, trades


def main():
    n = 1500
    strategies = [
        ('XAUUSD MA crossover', 0.0006, 0.011, 42),
        ('EURUSD Donchian breakout', 0.0004, 0.008, 7),
        ('BTC perpetual trend', 0.0010, 0.025, 99),
    ]

    benchmark = pd.Series(
        np.random.normal(0.0003, 0.010, n),
        index=pd.date_range('2020-01-01', periods=n, freq='D'),
    )

    summary = []
    for name, mu, sigma, seed in strategies:
        returns, trades = make_synthetic_strategy(name, mu, sigma, n, seed)
        out_path = f'/tmp/tear_sheet_{name.replace(" ", "_").lower()}.pdf'
        create_tear_sheet(
            returns=returns,
            trades=trades,
            benchmark_returns=benchmark,
            strategy_name=name,
            instrument=name.split()[0],
            output_path=out_path,
        )

        sharpe = sharpe_ratio(returns)
        dd = max_drawdown(returns)
        calmar = calmar_ratio(returns)
        summary.append({
            'strategy': name,
            'sharpe':   sharpe,
            'max_dd':   dd['max_drawdown'] * 100,
            'calmar':   calmar,
            'pdf':      out_path,
        })

    print("\n=== SO SÁNH 3 STRATEGIES ===")
    df = pd.DataFrame(summary)
    print(df.to_string(index=False))

    print("\n=== VERDICT ===")
    best = df.sort_values('sharpe', ascending=False).iloc[0]
    print(f"Highest Sharpe: {best['strategy']} ({best['sharpe']:.2f})")
    print(f"Best Calmar:    {df.sort_values('calmar', ascending=False).iloc[0]['strategy']}")
    print(f"Lowest DD:      {df.sort_values('max_dd', ascending=False).iloc[0]['strategy']}")
    print("\nKhuyến nghị: combine cả 3 trong portfolio (xem Bài 3)")


if __name__ == '__main__':
    main()
