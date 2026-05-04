"""
QuantCFD - Chapter 4 - Storage Format Benchmark
================================================
Section 4.7 + Bài tập 3: So sánh CSV vs Parquet vs SQLite cho data lớn.

Đo:
    - File size mỗi format
    - Time to write
    - Time to read full file
    - Time to read random subset

Yêu cầu:
    pip install pandas pyarrow

Chạy:
    python chapter-04/storage_benchmark.py
"""
from __future__ import annotations

import os
import sqlite3
import time
from typing import Callable

import numpy as np
import pandas as pd


def make_test_data(n_rows: int = 500_000) -> pd.DataFrame:
    """Tạo synthetic OHLCV-like data để benchmark."""
    np.random.seed(42)
    timestamps = pd.date_range("2023-01-01", periods=n_rows, freq="1min", tz="UTC")
    close = 50000 + np.cumsum(np.random.normal(0, 50, n_rows))

    df = pd.DataFrame(
        {
            "open": close + np.random.uniform(-30, 30, n_rows),
            "high": close + np.random.uniform(0, 60, n_rows),
            "low": close - np.random.uniform(0, 60, n_rows),
            "close": close,
            "volume": np.random.randint(1, 10000, n_rows),
        },
        index=timestamps,
    )
    df.index.name = "datetime"
    return df


def time_op(label: str, fn: Callable) -> tuple:
    """Run fn() và đo thời gian. Trả về (elapsed_seconds, result)."""
    start = time.time()
    result = fn()
    elapsed = time.time() - start
    return elapsed, result


def benchmark_csv(df: pd.DataFrame, base: str = "_test_data") -> dict:
    path = f"{base}.csv"
    t_w, _ = time_op("CSV write", lambda: df.to_csv(path))
    t_r, df_loaded = time_op(
        "CSV read full",
        lambda: pd.read_csv(path, index_col=0, parse_dates=True),
    )
    # Random access: filter 1 ngày
    target_day = df.index[len(df) // 2].normalize()
    t_random, _ = time_op(
        "CSV read 1 day",
        lambda: pd.read_csv(path, index_col=0, parse_dates=True).loc[
            str(target_day.date())
        ],
    )
    size_mb = os.path.getsize(path) / 1e6
    return {"size_mb": size_mb, "write_s": t_w, "read_s": t_r, "random_s": t_random}


def benchmark_parquet(df: pd.DataFrame, base: str = "_test_data") -> dict:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return {"error": "Cần cài pyarrow: pip install pyarrow"}

    path = f"{base}.parquet"
    t_w, _ = time_op("Parquet write", lambda: df.to_parquet(path))
    t_r, df_loaded = time_op("Parquet read full", lambda: pd.read_parquet(path))

    target_day = df.index[len(df) // 2].normalize()

    def random_read():
        d = pd.read_parquet(path)
        return d.loc[str(target_day.date())]

    t_random, _ = time_op("Parquet read 1 day", random_read)
    size_mb = os.path.getsize(path) / 1e6
    return {"size_mb": size_mb, "write_s": t_w, "read_s": t_r, "random_s": t_random}


def benchmark_sqlite(df: pd.DataFrame, base: str = "_test_data") -> dict:
    path = f"{base}.db"
    if os.path.exists(path):
        os.remove(path)

    def write():
        with sqlite3.connect(path) as conn:
            df.to_sql("ohlcv", conn, if_exists="replace", index=True)
            conn.execute("CREATE INDEX idx_dt ON ohlcv(datetime)")

    t_w, _ = time_op("SQLite write", write)

    def read_full():
        with sqlite3.connect(path) as conn:
            return pd.read_sql(
                "SELECT * FROM ohlcv", conn, index_col="datetime", parse_dates=["datetime"]
            )

    t_r, df_loaded = time_op("SQLite read full", read_full)

    target_day = df.index[len(df) // 2].normalize()
    target_str = str(target_day.date())

    def random_read():
        with sqlite3.connect(path) as conn:
            return pd.read_sql(
                f"SELECT * FROM ohlcv WHERE datetime LIKE '{target_str}%'",
                conn,
                index_col="datetime",
                parse_dates=["datetime"],
            )

    t_random, _ = time_op("SQLite read 1 day", random_read)
    size_mb = os.path.getsize(path) / 1e6
    return {"size_mb": size_mb, "write_s": t_w, "read_s": t_r, "random_s": t_random}


def cleanup(base: str = "_test_data") -> None:
    for ext in ("csv", "parquet", "db"):
        f = f"{base}.{ext}"
        if os.path.exists(f):
            os.remove(f)


def main() -> None:
    n_rows = 500_000
    print(f"Generating synthetic data: {n_rows:,} rows of OHLCV...")
    df = make_test_data(n_rows)
    print(f"DataFrame in-memory size: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB\n")

    results = {
        "CSV": benchmark_csv(df),
        "Parquet": benchmark_parquet(df),
        "SQLite": benchmark_sqlite(df),
    }

    print(f"\n{'='*78}")
    print(f"{'Format':<12} {'Size (MB)':<12} {'Write (s)':<12} "
          f"{'Read full (s)':<15} {'Random 1d (s)':<15}")
    print("=" * 78)
    for fmt, r in results.items():
        if "error" in r:
            print(f"{fmt:<12} ERROR: {r['error']}")
            continue
        print(
            f"{fmt:<12} "
            f"{r['size_mb']:<12.2f} "
            f"{r['write_s']:<12.3f} "
            f"{r['read_s']:<15.3f} "
            f"{r['random_s']:<15.3f}"
        )

    # So sánh tỷ lệ
    if "Parquet" in results and "error" not in results["Parquet"]:
        csv_size = results["CSV"]["size_mb"]
        pq_size = results["Parquet"]["size_mb"]
        print(f"\nParquet smaller than CSV by: {csv_size / pq_size:.1f}x")

        csv_read = results["CSV"]["read_s"]
        pq_read = results["Parquet"]["read_s"]
        print(f"Parquet faster read than CSV: {csv_read / pq_read:.1f}x")

    # Cleanup
    cleanup()
    print("\n(Cleaned up test files)")
    print("\nKết luận: Parquet là default cho data warehouse quant. "
          "CSV chỉ dùng cho debug. SQLite dùng khi cần query subset thường xuyên.")


if __name__ == "__main__":
    main()
