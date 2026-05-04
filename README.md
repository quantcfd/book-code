# QuantCFD — Book Code Vault

Code đi kèm cuốn sách **QuantCFD: Giao Dịch Định Lượng CFD Cho Trader Việt** của tác giả **Anthony Nguyễn**, xuất bản bởi **Monta Capital Investment Company Limited**.

> *Mọi kỹ thuật trong sách đã được công bố trong các paper từ thập niên 1990s. Renaissance Technologies — Sharpe ~2.5 trong 30+ năm — không có thuật toán bí mật. Họ thắng vì execution và data quality.*

## Cấu trúc

```
book-code/
├── chapter-01/          # Hành trình từ cảm tính đến định lượng
├── chapter-02/          # Tư duy thống kê cho trader
├── chapter-03/          # Python cho trader — crash course
└── template/            # Template cấu trúc dự án quant chuẩn
```

Mỗi thư mục `chapter-XX/` chứa code thực thi được của các ví dụ và bài tập trong chương tương ứng. Đọc README.md trong từng thư mục để biết chi tiết.

Thư mục `template/` chứa cấu trúc dự án quant chuẩn (giảng ở Chương 3, mục 3.6) — anh em fork về làm seed cho dự án nghiên cứu của mình.

## Setup

```bash
# Clone repo
git clone https://github.com/quantcfd/book-code.git
cd book-code

# Tạo môi trường Python riêng (yêu cầu Anaconda/Miniconda)
conda create -n quantcfd python=3.11 -y
conda activate quantcfd

# Cài dependencies
pip install -r requirements.txt
```

Sau đó chạy code của bất kỳ chương nào, ví dụ:

```bash
python chapter-01/hello_market.py
python chapter-02/expectancy_calculator.py
python chapter-03/speed_test.py
```

## Yêu cầu

- Python 3.11+ (3.10 cũng OK, nhưng 3.11 nhanh hơn)
- Internet để tải dữ liệu thị trường (yfinance)
- ~1GB dung lượng đĩa cho dữ liệu mẫu

## Cộng đồng

- **Sách:** [quantcfd.com](https://quantcfd.com)
- **Discord:** [discord.gg/CC6xsZ8tcf](https://discord.gg/CC6xsZ8tcf)
- **Liên hệ tác giả:** book@quantcfd.com

## Khuyến cáo

Code trong repo này dành cho mục đích **giáo dục**. Không phải lời khuyên đầu tư. Giao dịch CFD/Forex/Crypto có rủi ro mất toàn bộ vốn. Tác giả và Monta Capital không chịu trách nhiệm cho khoản lỗ phát sinh từ việc áp dụng code này.

Hiệu suất quá khứ không đảm bảo kết quả tương lai.

## License

MIT License — xem file [LICENSE](LICENSE).

---

*"Quant trading = Rule + Evidence + Risk Control. Thiếu một trong ba = không phải quant, dù có dùng Python hay không."*
— QuantCFD, Chương 1
