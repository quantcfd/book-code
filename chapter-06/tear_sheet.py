"""
QuantCFD Chương 6 — tear_sheet.py
Generate full 1-page tear sheet PDF từ returns series.

Usage:
    from tear_sheet import create_tear_sheet
    create_tear_sheet(daily_returns, output_path='my_tear_sheet.pdf')
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

from metrics import (
    returns_summary, sharpe_ratio, sortino_ratio, calmar_ratio,
    max_drawdown, ulcer_index, historical_var, historical_cvar,
    profit_factor, expectancy_per_trade, max_consecutive_losses,
    yearly_breakdown, rolling_sharpe,
)


def create_tear_sheet(
    returns: pd.Series,
    trades: pd.DataFrame = None,
    benchmark_returns: pd.Series = None,
    strategy_name: str = "Strategy",
    instrument: str = "",
    output_path: str = "tear_sheet.pdf",
    periods_per_year: int = 252,
):
    """Generate 1-page tear sheet PDF."""
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns phải có DatetimeIndex")

    fig = plt.figure(figsize=(11, 14))
    gs = GridSpec(6, 2, figure=fig, hspace=0.5, wspace=0.3)

    # === HEADER ===
    fig.suptitle(
        f"{strategy_name}  |  {instrument}  |  {returns.index[0].date()} → {returns.index[-1].date()}",
        fontsize=14, fontweight='bold', y=0.99
    )

    rs = returns_summary(returns, periods_per_year)
    dd_info = max_drawdown(returns)
    sharpe = sharpe_ratio(returns, periods_per_year=periods_per_year)

    pct_pos_str = f"{rs['pct_positive_months']*100:.0f}%" if not np.isnan(rs.get('pct_positive_months', np.nan)) else "N/A"
    key_metrics_text = (
        f"CAGR: {rs['cagr']*100:+.1f}%   |   "
        f"Sharpe: {sharpe:.2f}   |   "
        f"Max DD: {dd_info['max_drawdown']*100:.1f}%   |   "
        f"Days underwater: {dd_info['days_underwater']}   |   "
        f"Win months: {pct_pos_str}"
    )
    fig.text(0.5, 0.96, key_metrics_text, ha='center', fontsize=11)

    # === EQUITY CURVE ===
    ax1 = fig.add_subplot(gs[1, 0])
    equity = (1 + returns).cumprod()
    ax1.plot(equity.index, equity.values, color='#0E7C7B', lw=1.5, label='Strategy')
    if benchmark_returns is not None:
        b_eq = (1 + benchmark_returns.reindex(equity.index).fillna(0)).cumprod()
        ax1.plot(b_eq.index, b_eq.values, color='gray', lw=1, ls='--', label='Benchmark')
    ax1.set_title('Equity Curve', fontweight='bold')
    ax1.set_ylabel('Equity (× initial)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # === DRAWDOWN ===
    ax2 = fig.add_subplot(gs[1, 1])
    dd = dd_info['drawdown_series'] * 100
    ax2.fill_between(dd.index, dd.values, 0, color='red', alpha=0.4)
    ax2.plot(dd.index, dd.values, color='darkred', lw=1)
    ax2.set_title('Drawdown', fontweight='bold')
    ax2.set_ylabel('DD (%)')
    ax2.grid(True, alpha=0.3)

    # === ROLLING SHARPE ===
    ax3 = fig.add_subplot(gs[2, 0])
    rs_series = rolling_sharpe(returns, window=252, periods_per_year=periods_per_year).dropna()
    if len(rs_series) > 0:
        ax3.plot(rs_series.index, rs_series.values, color='#0F1F3D', lw=1.2)
        ax3.axhline(0, color='black', lw=0.5)
        ax3.axhline(1, color='green', lw=0.5, ls='--', label='Sharpe = 1')
        ax3.fill_between(rs_series.index, 0, rs_series.values, alpha=0.2,
                         where=rs_series.values > 0, color='green')
        ax3.fill_between(rs_series.index, 0, rs_series.values, alpha=0.2,
                         where=rs_series.values < 0, color='red')
    ax3.set_title('Rolling 1-Year Sharpe', fontweight='bold')
    ax3.set_ylabel('Sharpe')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # === MONTHLY HEATMAP ===
    ax4 = fig.add_subplot(gs[2, 1])
    monthly = (1 + returns).resample('ME').prod() - 1
    if len(monthly) > 0:
        monthly_df = pd.DataFrame({
            'year':   monthly.index.year,
            'month':  monthly.index.month,
            'return': monthly.values * 100,
        })
        pivot = monthly_df.pivot(index='year', columns='month', values='return')
        if HAS_SNS:
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                        cbar_kws={'label': 'Return (%)'}, ax=ax4)
        else:
            ax4.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
            ax4.set_xticks(range(len(pivot.columns)))
            ax4.set_xticklabels(pivot.columns)
            ax4.set_yticks(range(len(pivot.index)))
            ax4.set_yticklabels(pivot.index)
    ax4.set_title('Monthly Returns (%)', fontweight='bold')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Year')

    # === DETAILED METRICS ===
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis('off')
    sortino = sortino_ratio(returns, periods_per_year=periods_per_year)
    calmar = calmar_ratio(returns, periods_per_year=periods_per_year)
    ulcer = ulcer_index(returns)
    var95 = historical_var(returns, 0.95)
    cvar95 = historical_cvar(returns, 0.95)

    table_data = [
        ['RETURNS', 'RISK', 'RISK-ADJUSTED', 'TAIL'],
        [f"Total: {rs['total_return']*100:+.1f}%",
         f"Annual vol: {returns.std()*np.sqrt(periods_per_year)*100:.1f}%",
         f"Sharpe: {sharpe:.2f}",
         f"VaR 95%: {var95*100:.2f}%"],
        [f"CAGR: {rs['cagr']*100:+.1f}%",
         f"Max DD: {dd_info['max_drawdown']*100:.1f}%",
         f"Sortino: {sortino:.2f}",
         f"CVaR 95%: {cvar95*100:.2f}%"],
        [f"Best month: {rs.get('best_month', 0)*100:+.1f}%",
         f"Days u/w: {dd_info['days_underwater']}",
         f"Calmar: {calmar:.2f}",
         f"Worst day: {rs['worst_day']*100:.2f}%"],
        [f"Worst month: {rs.get('worst_month', 0)*100:+.1f}%",
         f"Ulcer: {ulcer:.2f}",
         f"% positive months: {pct_pos_str}",
         f"Worst month: {rs.get('worst_month', 0)*100:.2f}%"],
    ]
    table = ax5.table(cellText=table_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for i in range(4):
        table[(0, i)].set_facecolor('#0F1F3D')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # === TRADE DISTRIBUTION ===
    ax6 = fig.add_subplot(gs[4, 0])
    if trades is not None and 'pnl' in trades.columns:
        trades['pnl'].hist(bins=40, ax=ax6, color='#0E7C7B', edgecolor='black', alpha=0.7)
        ax6.axvline(0, color='black', lw=1)
        ax6.axvline(trades['pnl'].mean(), color='red', lw=1, ls='--',
                    label=f"Mean: {trades['pnl'].mean():+.1f}")
        ax6.set_title('Trade P&L Distribution', fontweight='bold')
        ax6.set_xlabel('P&L per trade')
        ax6.legend()
    else:
        ax6.text(0.5, 0.5, 'Trade-level data\nnot provided',
                 ha='center', va='center', fontsize=10)
        ax6.set_title('Trade Distribution', fontweight='bold')

    # === YEARLY BREAKDOWN ===
    ax7 = fig.add_subplot(gs[4, 1])
    yb = yearly_breakdown(returns)
    if not yb.empty:
        colors = ['green' if r > 0 else 'red' for r in yb['return']]
        ax7.bar(yb.index.astype(str), yb['return'], color=colors, alpha=0.7)
        ax7.set_title('Yearly Returns (%)', fontweight='bold')
        ax7.axhline(0, color='black', lw=0.5)
        ax7.grid(True, alpha=0.3, axis='y')

    fig.text(0.99, 0.01, 'Generated by QuantCFD tear_sheet.py',
             ha='right', fontsize=8, style='italic', color='gray')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Tear sheet saved: {output_path}")


if __name__ == '__main__':
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1500, freq='D')
    synthetic_returns = pd.Series(np.random.normal(0.0005, 0.012, 1500), index=dates)
    benchmark = pd.Series(np.random.normal(0.0003, 0.010, 1500), index=dates)

    create_tear_sheet(
        synthetic_returns,
        benchmark_returns=benchmark,
        strategy_name='Demo Strategy',
        instrument='Synthetic',
        output_path='/tmp/demo_tear_sheet.pdf',
    )
