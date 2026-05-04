"""
QuantCFD Chương 6 — metrics.py
Đầy đủ 40+ metrics đề cập trong chương 6.

Usage:
    from metrics import returns_summary, sharpe_ratio, ...

    rs = returns_summary(daily_returns, periods_per_year=252)
    sharpe = sharpe_ratio(daily_returns)
    dd_info = max_drawdown(daily_returns)
"""
import numpy as np
import pandas as pd
from typing import Optional


# ============================================================
# RETURNS METRICS (Section 6.2)
# ============================================================
def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    """CAGR từ equity curve."""
    if len(equity) < 2:
        return 0.0
    n_years = len(equity) / periods_per_year
    if n_years <= 0:
        return 0.0
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1


def cagr_from_returns(returns: pd.Series, periods_per_year: int = 252) -> float:
    n_years = len(returns) / periods_per_year
    total_return = (1 + returns).prod() - 1
    return (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0


def geometric_mean(returns: pd.Series, periods_per_year: int = 252) -> float:
    n = len(returns)
    return (1 + returns).prod() ** (periods_per_year / n) - 1 if n > 0 else 0


def arithmetic_mean(returns: pd.Series, periods_per_year: int = 252) -> float:
    return returns.mean() * periods_per_year


def returns_summary(returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Summary đầy đủ returns metrics."""
    r = returns.dropna()
    if len(r) == 0:
        return {}
    n = len(r)
    n_years = n / periods_per_year

    total = (1 + r).prod() - 1
    cagr_val = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else 0
    arith_mean = r.mean() * periods_per_year
    geo_mean = (1 + r).prod() ** (periods_per_year / n) - 1 if n > 0 else 0

    monthly = (1 + r).resample('ME').prod() - 1 if isinstance(r.index, pd.DatetimeIndex) else pd.Series([])
    yearly = (1 + r).resample('YE').prod() - 1 if isinstance(r.index, pd.DatetimeIndex) else pd.Series([])

    return {
        'total_return':        total,
        'cagr':                cagr_val,
        'arith_mean_annual':   arith_mean,
        'geo_mean_annual':     geo_mean,
        'best_day':            r.max(),
        'worst_day':           r.min(),
        'best_month':          monthly.max() if len(monthly) > 0 else np.nan,
        'worst_month':         monthly.min() if len(monthly) > 0 else np.nan,
        'best_year':           yearly.max() if len(yearly) > 0 else np.nan,
        'worst_year':          yearly.min() if len(yearly) > 0 else np.nan,
        'pct_positive_months': (monthly > 0).mean() if len(monthly) > 0 else np.nan,
        'pct_positive_years':  (yearly > 0).mean() if len(yearly) > 0 else np.nan,
        'n_periods':           n,
        'n_years':             n_years,
    }


# ============================================================
# RISK METRICS (Section 6.3)
# ============================================================
def annual_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    return returns.std() * np.sqrt(periods_per_year)


def downside_vol(
    returns: pd.Series,
    target: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    downside = returns[returns < target] - target
    if len(downside) == 0:
        return 0.0
    return np.sqrt((downside ** 2).mean()) * np.sqrt(periods_per_year)


def max_drawdown(returns: pd.Series) -> dict:
    """Max drawdown + peak/trough/recovery dates + days underwater."""
    equity = (1 + returns.dropna()).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max

    if len(drawdown) == 0 or drawdown.min() == 0:
        return {
            'max_drawdown':    0,
            'peak_date':       None,
            'trough_date':     None,
            'recovery_date':   None,
            'days_underwater': 0,
            'recovered':       True,
            'drawdown_series': drawdown,
        }

    max_dd = drawdown.min()
    trough_date = drawdown.idxmin()
    peak_date = equity.loc[:trough_date].idxmax()

    after_trough = equity.loc[trough_date:]
    peak_value = equity.loc[peak_date]
    recovered = after_trough[after_trough >= peak_value]
    recovery_date = recovered.index[0] if len(recovered) > 0 else None

    if recovery_date and isinstance(equity.index, pd.DatetimeIndex):
        days_underwater = (recovery_date - peak_date).days
        recovered_flag = True
    elif isinstance(equity.index, pd.DatetimeIndex):
        days_underwater = (drawdown.index[-1] - peak_date).days
        recovered_flag = False
    else:
        days_underwater = len(drawdown.loc[peak_date:])
        recovered_flag = recovery_date is not None

    return {
        'max_drawdown':    max_dd,
        'peak_date':       peak_date,
        'trough_date':     trough_date,
        'recovery_date':   recovery_date,
        'days_underwater': days_underwater,
        'recovered':       recovered_flag,
        'drawdown_series': drawdown,
    }


def average_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns.dropna()).cumprod()
    drawdown = (equity / equity.cummax() - 1)
    in_drawdown = drawdown[drawdown < 0]
    return in_drawdown.mean() if len(in_drawdown) > 0 else 0.0


def ulcer_index(returns: pd.Series) -> float:
    """Ulcer = sqrt(mean(drawdown%^2)). Peter Martin 1989."""
    equity = (1 + returns.dropna()).cumprod()
    drawdown_pct = (equity / equity.cummax() - 1) * 100
    return np.sqrt((drawdown_pct ** 2).mean())


def pain_index(returns: pd.Series) -> float:
    equity = (1 + returns.dropna()).cumprod()
    drawdown = (equity / equity.cummax() - 1)
    return abs(drawdown.mean())


def lake_ratio(returns: pd.Series) -> float:
    equity = (1 + returns.dropna()).cumprod()
    running_max = equity.cummax()
    lake = (running_max - equity).sum()
    earth = equity.sum()
    return lake / earth if earth > 0 else 0


# ============================================================
# RISK-ADJUSTED RATIOS (Section 6.4)
# ============================================================
def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    r = returns.dropna()
    if len(r) == 0 or r.std() == 0:
        return 0.0
    excess = r.mean() - risk_free_rate / periods_per_year
    return excess / r.std() * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    target: float = 0.0,
) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    excess = r.mean() - risk_free_rate / periods_per_year
    downside = r[r < target] - target
    if len(downside) == 0:
        return float('inf')
    downside_dev = np.sqrt((downside ** 2).mean())
    if downside_dev == 0:
        return 0
    return excess / downside_dev * np.sqrt(periods_per_year)


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    n_years = len(r) / periods_per_year
    cagr_val = (1 + r).prod() ** (1 / n_years) - 1 if n_years > 0 else 0
    equity = (1 + r).cumprod()
    max_dd = (equity / equity.cummax() - 1).min()
    return cagr_val / abs(max_dd) if max_dd != 0 else 0


def mar_ratio(
    returns: pd.Series,
    lookback_years: float = 3.0,
    periods_per_year: int = 252,
) -> float:
    r = returns.dropna()
    n_lookback = int(lookback_years * periods_per_year)
    r_recent = r.tail(n_lookback)
    n_years = len(r_recent) / periods_per_year
    cagr_val = (1 + r_recent).prod() ** (1 / n_years) - 1 if n_years > 0 else 0
    equity = (1 + r_recent).cumprod()
    max_dd = (equity / equity.cummax() - 1).min()
    return cagr_val / abs(max_dd) if max_dd != 0 else 0


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    r = returns.dropna() - threshold
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return gains / losses if losses > 0 else float('inf')


def burke_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    r = returns.dropna()
    annual_ret = r.mean() * periods_per_year
    equity = (1 + r).cumprod()
    drawdown = (equity / equity.cummax() - 1)
    drawdowns_sq = (drawdown[drawdown < 0] ** 2).sum()
    if drawdowns_sq == 0:
        return float('inf')
    return (annual_ret - risk_free_rate) / np.sqrt(drawdowns_sq)


def sterling_ratio(
    returns: pd.Series,
    excess_dd_threshold: float = 0.10,
    periods_per_year: int = 252,
) -> float:
    r = returns.dropna()
    n_years = len(r) / periods_per_year
    cagr_val = (1 + r).prod() ** (1 / n_years) - 1 if n_years > 0 else 0
    equity = (1 + r).cumprod()
    max_dd = abs((equity / equity.cummax() - 1).min())
    return cagr_val / (max_dd + excess_dd_threshold)


def martin_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    annual_ret = returns.mean() * periods_per_year
    ulcer = ulcer_index(returns)
    if ulcer == 0:
        return float('inf')
    return (annual_ret - risk_free_rate) * 100 / ulcer


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ['s', 'b']
    excess = aligned['s'] - aligned['b']
    tracking_error = excess.std() * np.sqrt(periods_per_year)
    if tracking_error == 0:
        return 0
    return excess.mean() * periods_per_year / tracking_error


def treynor_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ['s', 'b']
    cov = aligned[['s', 'b']].cov()
    beta = cov.loc['s', 'b'] / cov.loc['b', 'b']
    annual_excess = aligned['s'].mean() * periods_per_year - risk_free_rate
    return annual_excess / beta if beta != 0 else 0


def m_squared(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    s_sharpe = sharpe_ratio(strategy_returns, risk_free_rate, periods_per_year)
    b_vol = benchmark_returns.std() * np.sqrt(periods_per_year)
    return risk_free_rate + s_sharpe * b_vol


# ============================================================
# TRADE-LEVEL METRICS (Section 6.5)
# ============================================================
def win_rate(trades_pnl: pd.Series) -> float:
    return (trades_pnl > 0).mean() if len(trades_pnl) > 0 else 0


def payoff_ratio(trades_pnl: pd.Series) -> float:
    wins = trades_pnl[trades_pnl > 0]
    losses = trades_pnl[trades_pnl < 0]
    if len(wins) == 0 or len(losses) == 0:
        return float('inf') if len(wins) > 0 else 0
    return wins.mean() / abs(losses.mean())


def profit_factor(trades_pnl: pd.Series) -> float:
    wins = trades_pnl[trades_pnl > 0].sum()
    losses = -trades_pnl[trades_pnl < 0].sum()
    return wins / losses if losses > 0 else float('inf')


def expectancy_per_trade(trades_pnl: pd.Series) -> float:
    n = len(trades_pnl)
    if n == 0:
        return 0.0
    wins = trades_pnl[trades_pnl > 0]
    losses = trades_pnl[trades_pnl < 0]
    p_win = len(wins) / n
    p_loss = len(losses) / n
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    return p_win * avg_win - p_loss * avg_loss


def kelly_fraction(trades_pnl: pd.Series) -> float:
    n = len(trades_pnl)
    if n == 0:
        return 0.0
    wins = trades_pnl[trades_pnl > 0]
    losses = trades_pnl[trades_pnl < 0]
    if len(wins) == 0 or len(losses) == 0:
        return 0
    p_win = len(wins) / n
    p_loss = 1 - p_win
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    payoff = avg_win / avg_loss
    return p_win - p_loss / payoff


def max_consecutive_losses(trades_pnl: pd.Series) -> int:
    is_loss = (trades_pnl < 0).astype(int)
    if is_loss.sum() == 0:
        return 0
    streaks = is_loss.groupby((is_loss != is_loss.shift()).cumsum()).cumsum()
    return int(streaks.max())


def max_consecutive_wins(trades_pnl: pd.Series) -> int:
    is_win = (trades_pnl > 0).astype(int)
    if is_win.sum() == 0:
        return 0
    streaks = is_win.groupby((is_win != is_win.shift()).cumsum()).cumsum()
    return int(streaks.max())


# ============================================================
# TAIL RISK (Section 6.6)
# ============================================================
def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    return returns.dropna().quantile(1 - confidence)


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    from scipy import stats
    mu = returns.mean()
    sigma = returns.std()
    return mu + sigma * stats.norm.ppf(1 - confidence)


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    var = historical_var(returns, confidence)
    tail_returns = returns[returns <= var]
    return tail_returns.mean() if len(tail_returns) > 0 else var


def distribution_moments(returns: pd.Series) -> dict:
    from scipy.stats import skew, kurtosis
    r = returns.dropna()
    return {
        'mean':     r.mean(),
        'std':      r.std(),
        'skew':     skew(r),
        'kurtosis': kurtosis(r, fisher=True),
        'min':      r.min(),
        'max':      r.max(),
        'median':   r.median(),
    }


def tail_ratio(returns: pd.Series) -> float:
    r = returns.dropna()
    p95 = r.quantile(0.95)
    p05 = r.quantile(0.05)
    return abs(p95) / abs(p05) if p05 != 0 else float('inf')


# ============================================================
# STABILITY (Section 6.7)
# ============================================================
def yearly_breakdown(returns: pd.Series) -> pd.DataFrame:
    """Sharpe, return, DD per year."""
    if not isinstance(returns.index, pd.DatetimeIndex):
        return pd.DataFrame()
    yearly_data = []
    for year, group in returns.groupby(returns.index.year):
        if len(group) < 50:
            continue
        annual_ret = (1 + group).prod() - 1
        annual_vol = group.std() * np.sqrt(252)
        sharpe = group.mean() / group.std() * np.sqrt(252) if group.std() > 0 else 0
        equity = (1 + group).cumprod()
        max_dd = (equity / equity.cummax() - 1).min()
        yearly_data.append({
            'year':   year,
            'return': annual_ret * 100,
            'vol':    annual_vol * 100,
            'sharpe': sharpe,
            'max_dd': max_dd * 100,
            'n_days': len(group),
            'best_day':  group.max() * 100,
            'worst_day': group.min() * 100,
        })
    df = pd.DataFrame(yearly_data)
    return df.set_index('year') if 'year' in df.columns else df


def rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    periods_per_year: int = 252,
) -> pd.Series:
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    return (rolling_mean / rolling_std) * np.sqrt(periods_per_year)


def edge_significance(returns: pd.Series, periods_per_year: int = 252) -> dict:
    from scipy import stats
    r = returns.dropna()
    n = len(r)
    if n < 2 or r.std() == 0:
        return {'sharpe': 0, 't_stat': 0, 'p_value': 1, 'significant': False, 'n': n}
    sharpe_annualized = r.mean() / r.std() * np.sqrt(periods_per_year)
    sharpe_per_period = r.mean() / r.std()
    t_stat = sharpe_per_period * np.sqrt(n)
    p_value = 1 - stats.t.cdf(t_stat, n - 1)
    return {
        'sharpe':          sharpe_annualized,
        't_stat':          t_stat,
        'p_value':         p_value,
        'significant':     p_value < 0.05,
        'n_observations':  n,
        'n_years':         n / periods_per_year,
    }


def stability_index(returns: pd.Series, period_size: int = 60) -> dict:
    n_chunks = len(returns) // period_size
    chunk_sharpes = []
    for i in range(n_chunks):
        chunk = returns.iloc[i*period_size:(i+1)*period_size]
        if chunk.std() > 0:
            s = chunk.mean() / chunk.std() * np.sqrt(252)
            chunk_sharpes.append(s)
    chunk_sharpes = pd.Series(chunk_sharpes)
    if len(chunk_sharpes) == 0:
        return {}
    return {
        'n_chunks':            len(chunk_sharpes),
        'pct_positive':        (chunk_sharpes > 0).mean() * 100,
        'pct_above_1':         (chunk_sharpes > 1).mean() * 100,
        'median_chunk_sharpe': chunk_sharpes.median(),
        'worst_chunk_sharpe':  chunk_sharpes.min(),
    }


# ============================================================
# BENCHMARK (Section 6.8)
# ============================================================
def alpha_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    from scipy.stats import linregress
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ['s', 'b']
    rf_per_period = risk_free_rate / periods_per_year
    excess_s = aligned['s'] - rf_per_period
    excess_b = aligned['b'] - rf_per_period

    slope, intercept, r_value, p_value, std_err = linregress(excess_b, excess_s)

    return {
        'alpha_annualized':  intercept * periods_per_year,
        'beta':              slope,
        'r_squared':         r_value ** 2,
        'p_value_alpha':     p_value,
        'alpha_significant': p_value < 0.05,
    }


def capture_ratios(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict:
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ['s', 'b']

    up_periods = aligned[aligned['b'] > 0]
    down_periods = aligned[aligned['b'] < 0]

    up_capture = (up_periods['s'].mean() / up_periods['b'].mean() * 100
                  if len(up_periods) > 0 and up_periods['b'].mean() > 0 else 0)
    down_capture = (down_periods['s'].mean() / down_periods['b'].mean() * 100
                    if len(down_periods) > 0 and down_periods['b'].mean() < 0 else 0)

    return {
        'up_capture_pct':   up_capture,
        'down_capture_pct': down_capture,
        'capture_ratio':    up_capture / abs(down_capture) if down_capture != 0 else float('inf'),
    }


# ============================================================
# Quick demo
# ============================================================
if __name__ == '__main__':
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1500, freq='D')
    returns = pd.Series(np.random.normal(0.0005, 0.01, 1500), index=dates)

    print("=== Returns Summary ===")
    rs = returns_summary(returns)
    for k, v in rs.items():
        if isinstance(v, float):
            print(f"  {k:30s} {v:.4f}")
        else:
            print(f"  {k:30s} {v}")

    print("\n=== Risk-adjusted ratios ===")
    print(f"  Sharpe:    {sharpe_ratio(returns):.3f}")
    print(f"  Sortino:   {sortino_ratio(returns):.3f}")
    print(f"  Calmar:    {calmar_ratio(returns):.3f}")
    print(f"  Burke:     {burke_ratio(returns):.3f}")
    print(f"  Sterling:  {sterling_ratio(returns):.3f}")
    print(f"  Martin:    {martin_ratio(returns):.3f}")

    print("\n=== Risk ===")
    dd = max_drawdown(returns)
    print(f"  Max DD:    {dd['max_drawdown']*100:.2f}%")
    print(f"  Days u/w:  {dd['days_underwater']}")
    print(f"  Ulcer:     {ulcer_index(returns):.3f}")
    print(f"  Pain:      {pain_index(returns)*100:.3f}%")

    print("\n=== Tail ===")
    print(f"  VaR 95%:   {historical_var(returns, 0.95)*100:.2f}%")
    print(f"  CVaR 95%:  {historical_cvar(returns, 0.95)*100:.2f}%")
    print(f"  Tail ratio: {tail_ratio(returns):.2f}")
    moments = distribution_moments(returns)
    print(f"  Skew:      {moments['skew']:+.3f}")
    print(f"  Kurtosis:  {moments['kurtosis']:+.3f}")
