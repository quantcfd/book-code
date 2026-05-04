# Chương 1 — Hành trình từ cảm tính đến định lượng

Code cho Chương 1 của sách **QuantCFD**.

## Files

| File | Mô tả | Bài tập / Section trong sách |
|---|---|---|
| `hello_market.py` | Lấy dữ liệu BTC, XAUUSD, US500 và in thống kê cơ bản | Bài tập 3 |

## Chạy

```bash
# Từ thư mục root của repo
python chapter-01/hello_market.py
```

## Kết quả kỳ vọng

```
Asset    | Annual Return  | Annual Vol   | Sharpe | Max DD
------------------------------------------------------------------------------
BTC      | Annual Return:  ~50-150%      | ~50-70%      |  1.5-2.5 | ~ -20% to -30%
XAUUSD   | Annual Return:  ~15-25%       | ~12-18%      |  1.0-1.8 | ~ -8% to -15%
US500    | Annual Return:  ~10-25%       | ~12-18%      |  0.7-1.5 | ~ -10% to -20%
```

(Số liệu thay đổi theo period — đây chỉ là khoảng tham khảo.)

## Bài tập tự làm thêm (Chương 1)

- **Bài 1:** Trading Journal Forensics — phân loại 50 lệnh gần nhất theo Rule vs Discretionary.
- **Bài 2:** Edge Identification — viết 3 lý do anh em kiếm được tiền dài hạn, map vào 4 loại edge.

Hai bài này không có code — làm trên giấy thật.
