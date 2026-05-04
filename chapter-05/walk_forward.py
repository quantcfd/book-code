"""
QuantCFD - Chapter 5 - Walk-Forward Analysis
==============================================
Section 5.7: Validate strategy robustness qua nhiều train/test windows.

Two modes:
    - Anchored: train mở rộng dần, test slide forward.
    - Rolling: train window cố định, slide.

Usage:
    results = walk_forward_analysis(
        df, signal_factory, param_grid,
        train_years=2, test_years=1, anchored=True,
        ...engine kwargs...,
    )
    print(results)  # bảng có in_sample_sharpe, out_of_sample_sharpe per window
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

try:
    from .backtest_engine import run_backtest
except ImportError:
    from backtest_engine import run_backtest


def walk_forward_analysis(
    df: pd.DataFrame,
    signal_fn_factory: Callable,
    param_grid: list,
    train_years: int = 2,
    test_years: int = 1,
    anchored: bool = True,
    metric: str = "sharpe",
    **engine_kwargs,
) -> pd.DataFrame:
    """
    Anchored hoặc rolling walk-forward.

    Mỗi window:
        1. Backtest tất cả params trong param_grid trên train period.
        2. Chọn params có metric (default Sharpe) cao nhất.
        3. Backtest params đó trên test period (out-of-sample).
        4. Lưu kết quả test.

    Args:
        df: full DataFrame OHLCV, index UTC.
        signal_fn_factory: function nhận **params → signal_fn.
        param_grid: list of dicts, mỗi dict là 1 set params.
        train_years: độ dài train window.
        test_years: độ dài test window.
        anchored: True = train mở rộng dần; False = rolling fixed window.
        metric: metric để chọn best params trên train (default 'sharpe').
        **engine_kwargs: truyền cho run_backtest (spread_profile, swap, v.v.)

    Returns:
        DataFrame với mỗi row là 1 window, columns:
            window_start, window_end, best_params,
            in_sample_metric, out_of_sample_metric,
            oos_total_return, oos_max_dd, oos_n_trades
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index phải là DatetimeIndex")

    results = []
    start = df.index[0]
    end = df.index[-1]

    test_start = start + pd.DateOffset(years=train_years)

    while test_start + pd.DateOffset(years=test_years) <= end:
        test_end = test_start + pd.DateOffset(years=test_years)

        # Train window
        if anchored:
            train_data = df.loc[start:test_start]
        else:
            train_window_start = test_start - pd.DateOffset(years=train_years)
            train_data = df.loc[train_window_start:test_start]

        # Test window (out-of-sample)
        test_data = df.loc[test_start:test_end]

        if len(train_data) < 100 or len(test_data) < 30:
            test_start += pd.DateOffset(years=test_years)
            continue

        # Search best params on train
        best_metric = -np.inf
        best_params = None
        for params in param_grid:
            try:
                signal_fn = signal_fn_factory(**params)
                r_train = run_backtest(
                    train_data, signal_fn, enable_asserts=False, **engine_kwargs
                )
                stats = r_train.stats()
                m = stats.get(metric, -np.inf)
                if m > best_metric:
                    best_metric = m
                    best_params = params
            except Exception as e:
                print(f"  Warning: params {params} failed train: {e}")
                continue

        if best_params is None:
            test_start += pd.DateOffset(years=test_years)
            continue

        # Test on out-of-sample
        try:
            r_test = run_backtest(
                test_data,
                signal_fn_factory(**best_params),
                enable_asserts=False,
                **engine_kwargs,
            )
            test_stats = r_test.stats()
        except Exception as e:
            print(f"  Warning: best_params {best_params} failed test: {e}")
            test_start += pd.DateOffset(years=test_years)
            continue

        results.append(
            {
                "window_start": test_start,
                "window_end": test_end,
                "best_params": best_params,
                f"in_sample_{metric}": best_metric,
                f"out_of_sample_{metric}": test_stats.get(metric, np.nan),
                "oos_total_return": test_stats.get("total_return", np.nan),
                "oos_max_dd": test_stats.get("max_drawdown", np.nan),
                "oos_n_trades": test_stats.get("n_trades", 0),
            }
        )

        test_start += pd.DateOffset(years=test_years)

    return pd.DataFrame(results)


def summarize_wfa(wfa_df: pd.DataFrame, metric: str = "sharpe") -> dict:
    """Tóm tắt kết quả walk-forward — câu trả lời chính: 'có robust không?'"""
    oos_col = f"out_of_sample_{metric}"
    is_col = f"in_sample_{metric}"

    if oos_col not in wfa_df.columns:
        return {}

    oos = wfa_df[oos_col].dropna()
    is_ = wfa_df[is_col].dropna()
    n_negative = (oos < 0).sum()

    return {
        "n_windows":              len(oos),
        "mean_in_sample":         float(is_.mean()),
        "mean_out_of_sample":     float(oos.mean()),
        "median_out_of_sample":   float(oos.median()),
        "min_out_of_sample":      float(oos.min()),
        "max_out_of_sample":      float(oos.max()),
        "n_oos_negative":         int(n_negative),
        "is_oos_decay":           float(is_.mean() - oos.mean()),
        "robust_verdict": (
            "ROBUST" if oos.mean() >= 0.5 and n_negative == 0
            else "MARGINAL" if oos.mean() >= 0.3
            else "OVERFITTED"
        ),
    }
