"""
QuantCFD — Chương 7
Triple MA + MACD Strategy

Reference: Ch7.5 — Strategy 3.
3 MAs alignment + MACD confirmation.
Reduces whipsaw vs simple MA crossover.
"""

import numpy as np
import pandas as pd


def triple_ma_signals(df: pd.DataFrame,
                      fast: int = 10, medium: int = 30, slow: int = 100,
                      ma_type: str = "ema") -> pd.DataFrame:
    """
    Generate triple MA alignment signals.
    
    Long: MA_fast > MA_medium > MA_slow
    Short: MA_fast < MA_medium < MA_slow
    Flat: alignment unclear
    
    SHIFTED by 1 bar.
    """
    out = df.copy()
    
    if ma_type == "sma":
        out["ma_fast"] = out["close"].rolling(fast).mean()
        out["ma_medium"] = out["close"].rolling(medium).mean()
        out["ma_slow"] = out["close"].rolling(slow).mean()
    elif ma_type == "ema":
        out["ma_fast"] = out["close"].ewm(span=fast, adjust=False).mean()
        out["ma_medium"] = out["close"].ewm(span=medium, adjust=False).mean()
        out["ma_slow"] = out["close"].ewm(span=slow, adjust=False).mean()
    else:
        raise ValueError(f"Unknown ma_type: {ma_type}")
    
    long_align = (out["ma_fast"] > out["ma_medium"]) & (out["ma_medium"] > out["ma_slow"])
    short_align = (out["ma_fast"] < out["ma_medium"]) & (out["ma_medium"] < out["ma_slow"])
    
    raw_signal = pd.Series(0, index=out.index)
    raw_signal[long_align] = 1
    raw_signal[short_align] = -1
    
    out["signal"] = raw_signal.shift(1)
    return out


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal_period: int = 9):
    """
    Compute MACD indicator.
    
    Returns df with: macd_line, macd_signal, macd_histogram (shifted by 1).
    """
    out = df.copy()
    ema_fast = out["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["close"].ewm(span=slow, adjust=False).mean()
    out["macd_line"] = (ema_fast - ema_slow).shift(1)
    out["macd_signal"] = out["macd_line"].ewm(span=signal_period, adjust=False).mean()
    out["macd_histogram"] = out["macd_line"] - out["macd_signal"]
    return out


def triple_ma_with_macd(df: pd.DataFrame,
                        fast: int = 10, medium: int = 30, slow: int = 100,
                        macd_fast: int = 12, macd_slow: int = 26,
                        macd_signal: int = 9) -> pd.DataFrame:
    """
    Triple MA alignment + MACD confirmation.
    
    Long: triple MA bullish aligned AND MACD histogram > 0
    Short: triple MA bearish aligned AND MACD histogram < 0
    """
    out = triple_ma_signals(df, fast, medium, slow, ma_type="ema")
    out = macd(out, macd_fast, macd_slow, macd_signal)
    
    triple_signal = out["signal"]
    macd_bullish = (out["macd_histogram"] > 0).astype(int)
    macd_bearish = (out["macd_histogram"] < 0).astype(int)
    
    confirmed_long = (triple_signal == 1) & (macd_bullish == 1)
    confirmed_short = (triple_signal == -1) & (macd_bearish == 1)
    
    final_signal = pd.Series(0, index=out.index)
    final_signal[confirmed_long] = 1
    final_signal[confirmed_short] = -1
    
    out["confirmed_signal"] = final_signal
    return out


def backtest_triple_ma(df: pd.DataFrame, use_macd: bool = True,
                       cost_per_trade: float = 0.0005) -> dict:
    """Backtest triple MA strategy with optional MACD filter."""
    if use_macd:
        sig_df = triple_ma_with_macd(df)
        signal_col = "confirmed_signal"
    else:
        sig_df = triple_ma_signals(df)
        signal_col = "signal"
    
    sig_df["asset_return"] = sig_df["close"].pct_change()
    sig_df["strat_return"] = sig_df[signal_col] * sig_df["asset_return"]
    
    sig_df["pos_change"] = sig_df[signal_col].diff().abs().fillna(0)
    sig_df["strat_return_net"] = (
        sig_df["strat_return"] - sig_df["pos_change"] * cost_per_trade
    )
    
    df_clean = sig_df.dropna()
    if len(df_clean) < 30:
        return {"sharpe": np.nan}
    
    bars_per_year = 252  # daily assumed
    sharpe = (df_clean["strat_return_net"].mean() / df_clean["strat_return_net"].std()
              * np.sqrt(bars_per_year)) if df_clean["strat_return_net"].std() > 0 else 0
    cagr = (1 + df_clean["strat_return_net"].mean()) ** bars_per_year - 1
    equity = (1 + df_clean["strat_return_net"]).cumprod()
    max_dd = (equity / equity.cummax() - 1).min()
    total_trades = int(df_clean["pos_change"].sum() / 2)
    
    return {
        "sharpe": sharpe,
        "cagr": cagr,
        "max_dd": max_dd,
        "total_trades": total_trades,
        "equity_curve": equity,
    }


if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range("2018-01-01", "2024-12-31", freq="D")
    returns = np.random.randn(len(dates)) * 0.012 + 0.0003
    prices = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({"close": prices}, index=dates)
    
    print("=" * 60)
    print("Triple MA + MACD Backtest")
    print("=" * 60)
    
    print("\nWithout MACD filter:")
    r1 = backtest_triple_ma(df, use_macd=False)
    print(f"  Sharpe: {r1['sharpe']:.3f} | CAGR: {r1['cagr']*100:.2f}% | "
          f"DD: {r1['max_dd']*100:.2f}% | Trades: {r1['total_trades']}")
    
    print("\nWith MACD confirmation:")
    r2 = backtest_triple_ma(df, use_macd=True)
    print(f"  Sharpe: {r2['sharpe']:.3f} | CAGR: {r2['cagr']*100:.2f}% | "
          f"DD: {r2['max_dd']*100:.2f}% | Trades: {r2['total_trades']}")
