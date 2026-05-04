"""
QuantCFD Chương 5 — Multi-strategy portfolio engine

Run multiple strategies simultaneously trên cùng instrument hoặc khác instruments,
combine equity curves theo weights.

Benefits của multi-strategy:
    - Diversification giảm correlation → portfolio Sharpe > sum(individual Sharpes)/N
    - Smooth equity curve, ít drawdown rough patches
    - Nếu 1 strategy fail regime, others compensate

Trade-off:
    - Capital allocation: weight 1.0 mỗi strategy = 100% × N → leverage thực
    - Implementation phức tạp: position netting, margin, margin call
    - Backtest realistic phải capture interaction (correlation, shared cost)
"""
import numpy as np
import pandas as pd


def run_multi_strategy_backtest(
    df: pd.DataFrame,
    strategies: dict,    # {name: signal_function}
    weights: dict,       # {name: weight}
    cost_function=None,
) -> dict:
    """
    Combine signals từ multiple strategies với weights.

    Position thực = sum(weight_i × signal_i)
    Position có thể > 1 (long combined) hoặc < -1 (short combined) — phụ thuộc weights.
    """
    if abs(sum(weights.values()) - 1.0) > 0.01:
        print(f"WARN: weights tổng = {sum(weights.values())}, không = 1.0")

    # Generate signals từ mỗi strategy
    signals_dict = {name: fn(df) for name, fn in strategies.items()}

    # Combined position
    combined_pos = pd.Series(0.0, index=df.index)
    for name, sig in signals_dict.items():
        combined_pos += weights[name] * sig

    # Backtest combined
    pos_lagged = combined_pos.shift(1).fillna(0)
    returns = df['close'].pct_change().fillna(0)
    raw_pnl = pos_lagged * returns

    # Costs (simplified)
    pos_change = pos_lagged.diff().abs().fillna(0)
    if cost_function is not None:
        cost = cost_function(pos_change)
    else:
        cost = pos_change * 0.0002  # 2 bps default

    net_pnl = raw_pnl - cost
    equity = (1 + net_pnl).cumprod()

    # Per-strategy contribution
    contributions = {}
    for name, sig in signals_dict.items():
        sig_lagged = sig.shift(1).fillna(0)
        strat_returns = sig_lagged * returns * weights[name]
        contributions[name] = {
            'total_contribution': strat_returns.sum(),
            'sharpe': strat_returns.mean() / strat_returns.std() * np.sqrt(252)
                      if strat_returns.std() > 0 else 0,
        }

    portfolio_sharpe = net_pnl.mean() / net_pnl.std() * np.sqrt(252) if net_pnl.std() > 0 else 0

    # Correlation matrix giữa strategies
    strat_returns_df = pd.DataFrame({
        name: signals_dict[name].shift(1).fillna(0) * returns
        for name in strategies
    })
    correlation_matrix = strat_returns_df.corr()

    return {
        'equity': equity,
        'total_return': equity.iloc[-1] - 1,
        'portfolio_sharpe': portfolio_sharpe,
        'max_dd': (equity / equity.cummax() - 1).min(),
        'contributions': contributions,
        'correlation_matrix': correlation_matrix,
    }


# ============================================================
# Example strategies
# ============================================================
def trend_strategy(df, fast=20, slow=50):
    ma_fast = df['close'].rolling(fast).mean()
    ma_slow = df['close'].rolling(slow).mean()
    return ((ma_fast > ma_slow).astype(int) - (ma_fast < ma_slow).astype(int)).fillna(0)


def breakout_strategy(df, n=20):
    high_n = df['high'].rolling(n).max().shift(1)
    low_n = df['low'].rolling(n).min().shift(1)
    pos = pd.Series(0, index=df.index)
    pos[df['close'] > high_n] = 1
    pos[df['close'] < low_n] = -1
    return pos


def mean_reversion_strategy(df, n=2, oversold=10, overbought=90):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(n).mean()
    loss = -delta.where(delta < 0, 0).rolling(n).mean()
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)

    pos = pd.Series(0, index=df.index)
    pos[rsi < oversold] = 1
    pos[rsi > overbought] = -1
    return pos.fillna(0)


def demo():
    """Run multi-strategy backtest trên synthetic data."""
    np.random.seed(0)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    close = 100 + np.cumsum(np.random.normal(0, 1, 1000))
    df = pd.DataFrame({
        'close': close,
        'high': close + np.random.uniform(0.1, 1, 1000),
        'low': close - np.random.uniform(0.1, 1, 1000),
    }, index=dates)

    strategies = {
        'trend':      lambda d: trend_strategy(d, fast=20, slow=50),
        'breakout':   lambda d: breakout_strategy(d, n=20),
        'mean_rev':   lambda d: mean_reversion_strategy(d, n=2),
    }
    weights = {'trend': 0.4, 'breakout': 0.3, 'mean_rev': 0.3}

    result = run_multi_strategy_backtest(df, strategies, weights)

    print("Multi-strategy backtest")
    print(f"Total return:      {result['total_return']*100:+.2f}%")
    print(f"Portfolio Sharpe:  {result['portfolio_sharpe']:.2f}")
    print(f"Max DD:            {result['max_dd']*100:.2f}%")

    print("\nPer-strategy contribution:")
    for name, c in result['contributions'].items():
        print(f"  {name:12s}  total={c['total_contribution']*100:+.2f}%  sharpe={c['sharpe']:.2f}")

    print("\nCorrelation matrix:")
    print(result['correlation_matrix'].round(2))

    print("\nDiversification check:")
    avg_pairwise_corr = (result['correlation_matrix'].values[
        np.triu_indices_from(result['correlation_matrix'], k=1)
    ]).mean()
    print(f"  Avg pairwise corr: {avg_pairwise_corr:.2f}  (lower = better diversification)")


if __name__ == '__main__':
    demo()
