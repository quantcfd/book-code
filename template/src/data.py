"""
src/data.py — Data loading and cleaning.

Functions:
    load_csv(path)         — Load OHLCV from CSV with proper datetime index
    load_parquet(path)     — Load from Parquet (recommended for large files)
    align_timezone(df, tz) — Align all data to single timezone
    clean_ohlcv(df)        — Remove duplicates, handle missing data, validate

Đầy đủ implementation sẽ học ở Chương 4.
"""
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """Load OHLCV CSV với datetime index."""
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df


def load_parquet(path: str) -> pd.DataFrame:
    """Load OHLCV Parquet — nhanh hơn CSV cho data lớn."""
    return pd.read_parquet(path)


# TODO: implement đầy đủ ở Chương 4
def align_timezone(df: pd.DataFrame, tz: str = "UTC") -> pd.DataFrame:
    raise NotImplementedError("Sẽ implement ở Chương 4")


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("Sẽ implement ở Chương 4")
