"""
QuantCFD Chương 5 — Demo 3: BTC perpetual trend following (H1)

Strategy: Long khi MA20 > MA50, short khi MA20 < MA50.
Realistic costs:
    - Spread: 0.5 bps + slippage 1 bps
    - Commission (Binance taker): 0.04% notional
    - Funding rate: 8h cycle, avg ±0.01%/8h. Long pay khi positive, short receive.

Funding cost compound theo direction:
    - Bull market: positive funding ~0.01-0.05%/8h → long pay 0.03-0.15%/day
    - Khi extreme: positive funding 0.1%/8h → long pay 0.3%/day → ăn vào edge

Demo này show: tại sao crypto perpetual trend không "free money" nếu strategy long-bias.
"""
import argparse
import numpy as np
import pandas as pd


def ma_cross_signals(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """+1 nếu MA fast > MA slow (long-only trend, không short)."""
    ma_fast = df['close'].rolling(fast).mean()
    ma_slow = df['close'].rolling(slow).mean()
    return ((ma_fast > ma_slow).astype(int)).fillna(0)


def funding_pnl_per_bar(
    position: pd.Series,
    funding_rate_8h: pd.Series,
    bars_per_8h: int = 8,    # H1 bars
) -> pd.Series:
    """
    Funding charge spread evenly across 8-hour window.
    position[t] = 1 → pay funding_rate_8h / bars_per_8h mỗi bar.

    Tradeoff: thực tế funding chỉ tính tại 00:00 / 08:00 / 16:00 UTC, nhưng
    spread cost giúp accounting đơn giản và không ảnh hưởng total cost.
    """
    return -position * funding_rate_8h / bars_per_8h


def run_backtest_btc_perp(
    df: pd.DataFrame,
    signals: pd.Series,
    funding_rate_8h: pd.Series,
    spread_bps: float = 0.5,
    commission_pct: float = 0.0004,
    slippage_bps: float = 1.0,
) -> dict:
    """Vectorized engine với funding cost integrated."""
    pos = signals.shift(1).fillna(0)
    returns = df['close'].pct_change().fillna(0)
    raw_pnl = pos * returns

    # Transaction costs
    pos_change = pos.diff().abs().fillna(0)
    spread_cost = (spread_bps + slippage_bps) / 10_000 * pos_change
    commission = commission_pct * pos_change
    funding_cost = funding_pnl_per_bar(pos, funding_rate_8h, bars_per_8h=8)

    net_pnl = raw_pnl - spread_cost - commission + funding_cost
    equity = (1 + net_pnl).cumprod()

    sharpe = net_pnl.mean() / net_pnl.std() * np.sqrt(24 * 365) if net_pnl.std() > 0 else 0
    max_dd = (equity / equity.cummax() - 1).min()
    total_funding = funding_cost.sum()

    return {
        'equity': equity,
        'total_return': equity.iloc[-1] - 1,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'total_funding_paid': -total_funding,
        'total_spread_cost': spread_cost.sum(),
        'total_commission': commission.sum(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='BTC H1 with close + funding_rate columns')
    args = parser.parse_args()

    df = pd.read_csv(args.csv, parse_dates=['datetime'], index_col='datetime')
    if 'funding_rate_8h' not in df.columns:
        # Fallback: simulate avg funding 0.01%/8h
        df['funding_rate_8h'] = 0.0001
        print('WARN: dùng synthetic funding 0.01%/8h. Lấy data thực từ Binance API.')

    signals = ma_cross_signals(df, fast=20, slow=50)
    result = run_backtest_btc_perp(df, signals, df['funding_rate_8h'])

    print(f"Final equity multiple: {result['equity'].iloc[-1]:.3f}")
    print(f"Total return:    {result['total_return']*100:+.2f}%")
    print(f"Sharpe:          {result['sharpe']:.2f}")
    print(f"Max DD:          {result['max_dd']*100:.2f}%")
    print(f"\nCost breakdown:")
    print(f"  Spread+slip:   {result['total_spread_cost']*100:.2f}%")
    print(f"  Commission:    {result['total_commission']*100:.2f}%")
    print(f"  Funding paid:  {result['total_funding_paid']*100:.2f}%  ← thường lớn hơn cả 2 trên")


if __name__ == '__main__':
    main()
