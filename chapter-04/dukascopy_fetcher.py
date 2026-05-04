"""
QuantCFD - Chapter 4 - Dukascopy Historical Forex Data
========================================================
Section 4.3: Lấy tick/M1 data Forex miễn phí từ Dukascopy.

Dukascopy là Swiss broker, expose tick data lịch sử miễn phí qua web API.
Library `duka` wrap API thành Python interface.

Yêu cầu:
    pip install duka

Chạy:
    python chapter-04/dukascopy_fetcher.py
"""
import os

try:
    from duka.app import app as duka_app
    from duka.core.utils import TimeFrame
    HAS_DUKA = True
except ImportError:
    HAS_DUKA = False


def fetch_dukascopy(
    symbols: list,
    start_date: str,
    end_date: str,
    timeframe: str = "M1",
    output_folder: str = "./data/raw/dukascopy",
    threads: int = 4,
) -> None:
    """
    Tải data từ Dukascopy về output_folder dạng CSV.

    Args:
        symbols: list symbol, vd ["EURUSD", "GBPUSD"].
        start_date: "YYYY-MM-DD".
        end_date: "YYYY-MM-DD".
        timeframe: "TICK", "M1", "M5", "M15", "M30", "H1", "H4", "D1".
        output_folder: thư mục lưu CSV.
        threads: số thread parallel download.

    Output:
        File CSV trong output_folder, một file/symbol/ngày.
    """
    if not HAS_DUKA:
        raise ImportError(
            "Cần cài duka: pip install duka"
        )

    os.makedirs(output_folder, exist_ok=True)

    tf_map = {
        "TICK": TimeFrame.TICK,
        "M1": TimeFrame.M1,
        "M5": TimeFrame.M5,
        "M15": TimeFrame.M15,
        "M30": TimeFrame.M30,
        "H1": TimeFrame.H1,
        "H4": TimeFrame.H4,
        "D1": TimeFrame.D1,
    }

    if timeframe not in tf_map:
        raise ValueError(f"Timeframe phải là một trong: {list(tf_map.keys())}")

    duka_app(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        threads=threads,
        timeframe=tf_map[timeframe],
        folder=output_folder,
        header=True,
    )

    print(f"\nDownload xong. Files trong {output_folder}/")


def main() -> None:
    if not HAS_DUKA:
        print("CHƯA CÀI duka. Chạy: pip install duka")
        return

    # Tải EURUSD M1 cho 1 tháng — demo
    fetch_dukascopy(
        symbols=["EURUSD"],
        start_date="2024-12-01",
        end_date="2024-12-31",
        timeframe="M1",
        threads=4,
    )

    print(
        "\nLưu ý: Dukascopy data theo giờ broker (UTC+2 mùa hè, UTC+3 mùa đông). "
        "Convert sang UTC trước khi merge với data từ source khác."
    )


if __name__ == "__main__":
    main()
