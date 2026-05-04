"""
QuantCFD Chương 5 — Demo 4: US500 (S&P500 CFD) RSI(2) mean reversion (Daily)

Larry Connors' RSI(2) — chiến lược mean reversion nổi tiếng.

Rules:
    - Long khi: close > MA200 (uptrend filter) AND RSI(2) < 10 (oversold)
    - Exit khi: close > MA5 (mean reversion completed)
    - Short: ngược lại (close < MA200, RSI(2) > 90, exit khi close < MA5)

US500 CFD costs:
    - Spread: 0.5 pts (overnight 1.0 pts)
    - Commission: 0 (markup vào spread)
    - Overnight financing: SOFR + 2.5% = ~7%/year (đáng kể với swing)

Run:
    python demo_us500_rsi.py --csv data/US500_D1_2015_2024.csv
"""
import argparse
import numpy as np
import pandas as pd


def rsi(prices: pd.Series, n: int = 2) -> pd.Series:
    """Wilder RSI."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(n).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(n).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def rsi2_signals(df: pd.DataFrame) -> pd.Series:
    """+1 = long, -1 = short, 0 = flat. Sticky position."""
    ma200 = df['close'].rolling(200).mean()
    ma5 = df['close'].rolling(5).mean()
    rsi2 = rsi(df['close'], 2)

    position = pd.Series(0, index=df.index, dtype=int)
    state = 0
    for i in range(len(df)):
        if pd.isna(ma200.iloc[i]) or pd.isna(rsi2.iloc[i]):
            continue
        close = df['close'].iloc[i]
        if state == 0:
            if close > ma200.iloc[i] and rsi2.iloc[i] < 10:
                state = 1
            elif close < ma200.iloc[i] and rsi2.iloc[i] > 90:
                state = -1
        elif state == 1 and close > ma5.iloc[i]:
            state = 0
        elif state == -1 and close < ma5.iloc[i]:
            state = 0
        position.iloc[i] = state

    return position


def run_backtest_us500(
    df: pd.DataFrame,
    signals: pd.Series,
    spread_pts: float = 1.0,
    swap_annual_pct: float = 0.07,
    point_value: float = 1.0,
    initial_capital: float = 10_000,
) -> dict:
    """Daily bars, swap charged khi position carry overnight (mỗi bar = 1 đêm)."""
    pos = signals.shift(1).fillna(0)
    pct = df['close'].pct_change().fillna(0)
    raw_pnl = pos * pct

    pos_change = pos.diff().abs().fillna(0)
    avg_close = df['close'].mean()
    spread_cost = (spread_pts / avg_close) * pos_change

    swap_per_day = swap_annual_pct / 365
    swap_cost = pos.abs() * swap_per_day  # Pay swap cả long lẫn short trong CFD

    net_pnl = raw_pnl - spread_cost - swap_cost
    equity = initial_capital * (1 + net_pnl).cumprod()

    sharpe = net_pnl.mean() / net_pnl.std() * np.sqrt(252) if net_pnl.std() > 0 else 0
    max_dd = (equity / equity.cummax() - 1).min()

    pos_diff = pos.diff().fillna(0)
    num_trades = (pos_diff != 0).sum() // 2

    # Trade-level analysis
    trades_pnl = []
    cur_pnl = 0
    for i, p in enumerate(pos):
        if i == 0:
            continue
        if pos.iloc[i-1] == 0 and p != 0:
            cur_pnl = 0
        cur_pnl += net_pnl.iloc[i]
        if pos.iloc[i-1] != 0 and p == 0:
            trades_pnl.append(cur_pnl)
    trades_pnl = pd.Series(trades_pnl)
    win_rate = (trades_pnl > 0).mean() if len(trades_pnl) > 0 else 0
    avg_win = trades_pnl[trades_pnl > 0].mean() if (trades_pnl > 0).any() else 0
    avg_loss = trades_pnl[trades_pnl < 0].mean() if (trades_pnl < 0).any() else 0

    return {
        'equity': equity,
        'total_return': equity.iloc[-1] / initial_capital - 1,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'num_trades': num_trades,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'payoff_ratio': abs(avg_win / avg_loss) if avg_loss < 0 else float('inf'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, parse_dates=['datetime'], index_col='datetime')
    print(f"Data: {df.index[0].date()} → {df.index[-1].date()}, {len(df)} days")

    signals = rsi2_signals(df)
    print(f"Long days:  {(signals > 0).sum()}")
    print(f"Short days: {(signals < 0).sum()}")
    print(f"Flat days:  {(signals == 0).sum()}")

    result = run_backtest_us500(df, signals)

    print(f"\nFinal equity:   ${result['equity'].iloc[-1]:,.2f}")
    print(f"Total return:   {result['total_return']*100:+.2f}%")
    print(f"Annual Sharpe:  {result['sharpe']:.2f}")
    print(f"Max DD:         {result['max_dd']*100:.2f}%")
    print(f"# Trades:       {result['num_trades']}")
    print(f"Win rate:       {result['win_rate']*100:.1f}%")
    print(f"Payoff ratio:   {result['payoff_ratio']:.2f}")


if __name__ == '__main__':
    main()
