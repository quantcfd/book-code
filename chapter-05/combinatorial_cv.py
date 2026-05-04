"""
QuantCFD Chương 5 — Combinatorial Purged Cross-Validation
(Lopez de Prado, "Advances in Financial Machine Learning", 2018)

Vấn đề walk-forward truyền thống:
    - Chỉ test 1 path (train past → test future)
    - Test set bị contaminated khi indicator có rolling window > gap
    - Số lượng test sets ít → overfit param vẫn dễ

Combinatorial Purged CV (CPCV):
    - Chia data thành N nhóm (vd N=6)
    - Generate ALL possible (train, test) splits với k test groups out of N
    - Total paths = C(N, k)
    - Purge: bỏ samples liền kề test set khỏi train (tránh leak)
    - Embargo: bỏ samples sau test set khỏi next train

Ví dụ N=6, k=2: có C(6,2) = 15 unique paths.
Mỗi sample được test trong nhiều paths → robust statistics.
"""
import numpy as np
import pandas as pd
from itertools import combinations


def purged_train_test_split(
    n_samples: int,
    test_indices: list,
    embargo_pct: float = 0.01,
    purge_buffer: int = 5,
) -> tuple:
    """
    Generate train mask sau khi purge & embargo.

    test_indices: list các (start, end) tuples cho test groups
    embargo_pct: % sau test bị embargo
    purge_buffer: bars trước test bị purge (overlap với rolling window)
    """
    train_mask = np.ones(n_samples, dtype=bool)
    embargo_n = int(n_samples * embargo_pct)

    for start, end in test_indices:
        train_mask[start:end] = False
        # Purge: remove buffer trước test
        purge_start = max(0, start - purge_buffer)
        train_mask[purge_start:start] = False
        # Embargo: remove embargo_n sau test
        embargo_end = min(n_samples, end + embargo_n)
        train_mask[end:embargo_end] = False

    test_mask = np.zeros(n_samples, dtype=bool)
    for start, end in test_indices:
        test_mask[start:end] = True

    return train_mask, test_mask


def combinatorial_purged_cv(
    n_samples: int,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo_pct: float = 0.01,
    purge_buffer: int = 5,
):
    """
    Yield (train_mask, test_mask, group_indices) cho mỗi combination.

    Total combinations: C(n_groups, n_test_groups)
    """
    group_size = n_samples // n_groups
    groups = [(i * group_size, (i + 1) * group_size if i < n_groups - 1 else n_samples)
              for i in range(n_groups)]

    for test_group_combo in combinations(range(n_groups), n_test_groups):
        test_indices = [groups[i] for i in test_group_combo]
        train_mask, test_mask = purged_train_test_split(
            n_samples, test_indices, embargo_pct, purge_buffer
        )
        yield train_mask, test_mask, test_group_combo


def cpcv_evaluate(
    df: pd.DataFrame,
    strategy_fn,
    param_grid: list,
    n_groups: int = 6,
    n_test_groups: int = 2,
    metric: str = 'sharpe',
) -> pd.DataFrame:
    """
    strategy_fn(df_train, params) → strategy fitted
    strategy_fn cần return obj có .predict_signals(df_test) và compute_metric(df_test)

    Để demo đơn giản: strategy_fn nhận (train_df, params) và return dict với 'signals'.
    """
    from itertools import product
    n = len(df)
    results = []

    for path_idx, (train_mask, test_mask, test_groups) in enumerate(
        combinatorial_purged_cv(n, n_groups, n_test_groups)
    ):
        df_train = df[train_mask]
        df_test = df[test_mask]

        for params in param_grid:
            try:
                # User's strategy_fn: train + return test metrics
                metrics = strategy_fn(df_train, df_test, params)
                results.append({
                    'path': path_idx,
                    'test_groups': test_groups,
                    'params': params,
                    'train_size': train_mask.sum(),
                    'test_size': test_mask.sum(),
                    **metrics,
                })
            except Exception as e:
                print(f"Path {path_idx}, params {params}: {e}")

    return pd.DataFrame(results)


def cpcv_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate CPCV results theo params."""
    grouped = results.groupby(results['params'].astype(str))
    summary = grouped.agg(
        mean_sharpe=('sharpe', 'mean'),
        median_sharpe=('sharpe', 'median'),
        std_sharpe=('sharpe', 'std'),
        min_sharpe=('sharpe', 'min'),
        max_sharpe=('sharpe', 'max'),
        n_paths=('path', 'nunique'),
    ).round(3)
    summary['stability_score'] = summary['mean_sharpe'] / summary['std_sharpe']
    return summary.sort_values('stability_score', ascending=False)


def demo():
    """Toy example: simulate CPCV trên SMA crossover."""
    np.random.seed(0)
    n_samples = 1000
    df = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.normal(0, 1, n_samples)),
    })

    def strategy_fn(df_train, df_test, params):
        fast, slow = params['fast'], params['slow']
        signals = (df_test['close'].rolling(fast).mean() >
                   df_test['close'].rolling(slow).mean()).astype(int)
        returns = df_test['close'].pct_change() * signals.shift(1)
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        return {'sharpe': sharpe}

    param_grid = [
        {'fast': 5, 'slow': 20},
        {'fast': 10, 'slow': 30},
        {'fast': 20, 'slow': 50},
    ]

    results = cpcv_evaluate(df, strategy_fn, param_grid, n_groups=6, n_test_groups=2)
    print(f"Total paths × params: {len(results)}")
    print(f"Unique paths: {results['path'].nunique()} (= C(6,2) = 15)")

    print("\n--- CPCV Summary ---")
    print(cpcv_summary(results))


if __name__ == '__main__':
    demo()
