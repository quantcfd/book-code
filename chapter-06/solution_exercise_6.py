"""
Bài 6 (BONUS) — Strategy ranking system

Z-score 8 metrics cho mỗi strategy, weight, sum → final score.
"""
import numpy as np
import pandas as pd
from metrics import (
    sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown,
    profit_factor, tail_ratio, returns_summary,
)


def rank_strategies(strategies_returns: dict, weights: dict = None) -> pd.DataFrame:
    """Rank strategies by composite z-score."""
    if weights is None:
        weights = {
            'sharpe':        0.25,
            'sortino':       0.15,
            'calmar':        0.20,
            'max_dd_neg':    0.15,
            'pct_pos_months': 0.10,
            'profit_factor': 0.10,
            'tail_ratio':    0.05,
        }

    metrics_data = []
    for name, returns in strategies_returns.items():
        rs = returns_summary(returns)
        dd = max_drawdown(returns)
        # Make synthetic trades for profit factor
        synthetic_pnl = returns * 100  # treat each day as "trade"
        metrics_data.append({
            'strategy':        name,
            'sharpe':          sharpe_ratio(returns),
            'sortino':         sortino_ratio(returns),
            'calmar':          calmar_ratio(returns),
            'max_dd_neg':      -dd['max_drawdown'],   # higher = better
            'pct_pos_months':  rs.get('pct_positive_months', 0.5) or 0.5,
            'profit_factor':   profit_factor(synthetic_pnl),
            'tail_ratio':      tail_ratio(returns),
        })

    df = pd.DataFrame(metrics_data).set_index('strategy')

    # Z-score
    z = df.copy()
    for col in df.columns:
        std = df[col].std()
        if std > 0:
            z[col] = (df[col] - df[col].mean()) / std
        else:
            z[col] = 0

    # Weighted sum
    score = pd.Series(0.0, index=z.index)
    for col, w in weights.items():
        if col in z.columns:
            score += z[col] * w

    df['score'] = score
    df['rank'] = score.rank(ascending=False, method='min').astype(int)
    return df.sort_values('score', ascending=False)


def main():
    np.random.seed(42)
    n = 1500
    dates = pd.date_range('2020-01-01', periods=n, freq='D')

    strategies = {
        'XAUUSD MA':       pd.Series(np.random.normal(0.0008, 0.011, n), index=dates),
        'EURUSD breakout': pd.Series(np.random.normal(0.0005, 0.009, n), index=dates),
        'US500 RSI':       pd.Series(np.random.normal(0.0004, 0.013, n), index=dates),
        'BTC trend':       pd.Series(np.random.normal(0.0010, 0.027, n), index=dates),
        'GBPUSD scalp':    pd.Series(np.random.normal(0.0002, 0.008, n), index=dates),
    }

    print("=" * 80)
    print("BÀI 6 (BONUS) — STRATEGY RANKING")
    print("=" * 80)
    df = rank_strategies(strategies)
    print(df.round(3).to_string())

    print(f"\nWinner: {df.index[0]} (score {df.iloc[0]['score']:+.2f})")


if __name__ == '__main__':
    main()
