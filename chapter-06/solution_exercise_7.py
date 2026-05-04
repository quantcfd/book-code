"""
Bài 7 (BONUS) — Live monitoring dashboard

Streamlit app hiển thị live metrics. Chạy với:
    streamlit run solution_exercise_7.py

Note: yêu cầu pip install streamlit. Nếu không có streamlit, code template
vẫn hoạt động — chỉ cần run như script python để tạo CLI report.
"""
import sys
import numpy as np
import pandas as pd
from metrics import sharpe_ratio, max_drawdown, rolling_sharpe
from live_monitor import live_alerts, StopTradingRules, daily_summary_report


def cli_dashboard(live_returns: pd.Series, baseline: dict):
    """Fallback CLI dashboard nếu không có streamlit."""
    print("=" * 60)
    print("LIVE TRADING DASHBOARD (CLI)")
    print("=" * 60)

    equity = (1 + live_returns).cumprod()
    print(f"\nEquity:        ${10000 * equity.iloc[-1]:,.2f}")
    print(f"Total return:  {(equity.iloc[-1] - 1)*100:+.2f}%")
    print(f"Today:         {live_returns.iloc[-1]*100:+.2f}%")

    dd = max_drawdown(live_returns)
    print(f"Current DD:    {dd['max_drawdown']*100:.2f}%")
    print(f"Days u/w:      {dd['days_underwater']}")

    sharpe = sharpe_ratio(live_returns)
    print(f"Live Sharpe:   {sharpe:.2f}")
    print(f"Baseline:      {baseline['sharpe']:.2f}")

    print("\n--- ALERTS ---")
    alerts = live_alerts(live_returns, baseline)
    if alerts:
        for a in alerts:
            print(f"  {a}")
    else:
        print("  ✓ No alerts")

    print("\n--- STOP TRADING CHECK ---")
    rules = StopTradingRules()
    should_stop, reason = rules.should_stop(live_returns)
    if should_stop:
        print(f"  🔴 STOP: {reason}")
    else:
        print(f"  ✓ Continue trading")


def streamlit_dashboard():
    """Streamlit version. Run với: streamlit run solution_exercise_7.py"""
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit không available. Run as Python script for CLI version.")
        return False

    st.set_page_config(page_title='QuantCFD Live Monitor', layout='wide')
    st.title('Strategy Live Dashboard')

    np.random.seed(42)
    dates = pd.date_range('2024-08-01', periods=180, freq='D')
    live_returns = pd.Series(np.random.normal(0.0005, 0.012, 180), index=dates)

    baseline = {'sharpe': 1.5, 'max_dd': -0.12}

    equity = (1 + live_returns).cumprod() * 10000

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Equity', f"${equity.iloc[-1]:,.0f}",
                f"{(equity.iloc[-1]/10000-1)*100:+.1f}%")
    col2.metric('Today', f"{live_returns.iloc[-1]*100:+.2f}%")
    dd = max_drawdown(live_returns)
    col3.metric('Current DD', f"{dd['max_drawdown']*100:.1f}%",
                delta=f"vs base {baseline['max_dd']*100:.1f}%", delta_color='inverse')
    col4.metric('Live Sharpe', f"{sharpe_ratio(live_returns):.2f}",
                delta=f"vs base {baseline['sharpe']:.2f}")

    st.line_chart(equity, height=300)

    rs_series = rolling_sharpe(live_returns, window=60).dropna()
    if len(rs_series) > 0:
        st.subheader('Rolling 60-day Sharpe')
        st.line_chart(rs_series, height=200)

    st.subheader('Alerts')
    alerts = live_alerts(live_returns, baseline)
    if alerts:
        for a in alerts:
            st.warning(a)
    else:
        st.success('No alerts — strategy operating normally')

    st.subheader('Stop Trading Rules')
    rules = StopTradingRules()
    should_stop, reason = rules.should_stop(live_returns)
    if should_stop:
        st.error(f'🔴 STOP TRADING: {reason}')
    else:
        st.success('✓ Continue trading')

    return True


if __name__ == '__main__':
    np.random.seed(42)
    dates = pd.date_range('2024-08-01', periods=180, freq='D')
    live_returns = pd.Series(np.random.normal(0.0005, 0.012, 180), index=dates)
    baseline = {'sharpe': 1.5, 'max_dd': -0.12}

    # Try streamlit first, fallback to CLI
    if not streamlit_dashboard():
        cli_dashboard(live_returns, baseline)
