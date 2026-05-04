"""
QuantCFD Chương 5 — Demo 2: EURUSD Donchian breakout (H4)

Strategy:
    - Long khi close > rolling high N bars qua
    - Short khi close < rolling low N bars qua
    - Exit khi cross qua midline N/2 bars

Realistic costs:
    - EURUSD spread: 0.8 pip avg, 1.5 pip session-edge
    - Commission: $7/lot round-trip (ECN broker)
    - Slippage: 0.3 pip ATR-scaled

Run:
    python demo_eurusd_breakout.py --csv data/EURUSD_H4_2020_2024.csv
"""
import argparse
import numpy as np
import pandas as pd

from cost_models import session_aware_spread_eurusd
from backtest_engine import run_backtest


def donchian_breakout_signals(df: pd.DataFrame, n_breakout: int = 20, n_exit: int = 10) -> pd.Series:
    """+1 = long, -1 = short, 0 = flat. Exit khi cross midline n_exit-bar."""
    high_n = df['high'].rolling(n_breakout).max().shift(1)
    low_n = df['low'].rolling(n_breakout).min().shift(1)
    mid_exit = (df['high'].rolling(n_exit).max().shift(1) +
                df['low'].rolling(n_exit).min().shift(1)) / 2

    position = pd.Series(0, index=df.index, dtype=int)
    state = 0
    for i in range(len(df)):
        close = df['close'].iloc[i]
        if state == 0:
            if close > high_n.iloc[i]:
                state = 1
            elif close < low_n.iloc[i]:
                state = -1
        elif state == 1 and close < mid_exit.iloc[i]:
            state = 0
        elif state == -1 and close > mid_exit.iloc[i]:
            state = 0
        position.iloc[i] = state

    return position


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--n_breakout', type=int, default=20)
    parser.add_argument('--n_exit', type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, parse_dates=['datetime'], index_col='datetime')
    print(f"Data: {df.index[0]} → {df.index[-1]}, {len(df)} bars")

    signals = donchian_breakout_signals(df, args.n_breakout, args.n_exit)
    print(f"Long bars:  {(signals > 0).sum()}  ({(signals > 0).mean()*100:.1f}%)")
    print(f"Short bars: {(signals < 0).sum()}  ({(signals < 0).mean()*100:.1f}%)")

    spread = session_aware_spread_eurusd(df)
    result = run_backtest(
        df=df,
        signals=signals,
        spread=spread,
        commission_per_lot=7.0,
        contract_size=100_000,
        position_size=0.1,
        initial_capital=10_000,
    )

    print(f"\nFinal equity:  ${result['equity'].iloc[-1]:,.2f}")
    print(f"Total return:  {result['total_return']*100:+.2f}%")
    print(f"Sharpe (4h annualized 2190): {result['sharpe_4h']:.2f}")
    print(f"Max DD:        {result['max_dd']*100:.2f}%")
    print(f"Win rate:      {result['win_rate']*100:.1f}%")
    print(f"# trades:      {result['num_trades']}")


if __name__ == '__main__':
    main()
