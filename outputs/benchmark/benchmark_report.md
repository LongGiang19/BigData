# Spark end-to-end inference benchmark

Đã ghi **21 lượt hoàn tất**. Spark chạy **local trên một máy**.
Spark 3.5.8, Java 11.0.30, Python 3.11.18,
Windows-10-10.0.26300-SP0, 32 CPU logic, driver 4g,
8 shuffle partitions, Arrow tối đa 20000 dòng/batch.
Mỗi Python worker dùng 1 BLAS thread. Các cấu hình master dùng tiến trình JVM riêng.

| Master | Flows | Lượt | Scoring partitions | Read (s) | Prep (s) | Score (s) | Write (s) | E2E (s) | Score flow/s | E2E flow/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| local[*] | 100,000 | 3 | 7 | 2.87 ± 1.79 | 7.34 ± 1.05 | 5.49 ± 0.15 | 0.62 ± 0.37 | 16.43 ± 3.27 | 18,216 | 6,239 |
| local[*] | 500,000 | 3 | 16 | 2.65 ± 0.13 | 7.19 ± 0.95 | 10.73 ± 0.21 | 0.42 ± 0.06 | 21.06 ± 1.12 | 46,622 | 23,790 |
| local[*] | 1,000,000 | 3 | 30 | 6.65 ± 6.22 | 15.50 ± 7.40 | 18.93 ± 1.16 | 0.97 ± 0.73 | 42.22 ± 10.82 | 52,972 | 24,947 |
| local[*] | 4,653,501 | 3 | 139 | 6.51 ± 0.33 | 13.90 ± 0.12 | 82.99 ± 0.45 | 2.73 ± 0.18 | 106.30 ± 0.98 | 56,071 | 43,780 |
| local[1] | 100,000 | 3 | 7 | 2.94 ± 1.98 | 8.45 ± 1.54 | 13.20 ± 0.25 | 0.58 ± 0.30 | 25.29 ± 3.98 | 7,580 | 4,017 |
| local[2] | 100,000 | 3 | 7 | 2.77 ± 1.77 | 7.62 ± 1.48 | 7.97 ± 0.10 | 0.56 ± 0.42 | 19.04 ± 3.71 | 12,546 | 5,383 |
| local[4] | 100,000 | 3 | 7 | 2.85 ± 1.82 | 7.48 ± 1.46 | 5.74 ± 0.33 | 0.56 ± 0.46 | 16.75 ± 4.03 | 17,470 | 6,198 |

Thời gian là trung bình ± độ lệch chuẩn mẫu. Throughput là trung bình
của số dòng chia thời gian từng lượt. CSV lưu cả throughput từng pha
và độ lệch chuẩn.

## Phương pháp

- Input nhỏ là tiền tố vật lý cố định của từng ngày, với quota tỷ lệ theo
  số dòng. Full test đọc trực tiếp snapshot 100%. Việc chuẩn bị input,
  nạp model và khởi động Spark diễn ra ngoài phần đo.
- **Read:** Đọc Parquet và materialize toàn bộ cột dùng cho preprocessing
  vào cache Spark `DISK_ONLY` bằng action đếm dòng. **Prep:** Chuẩn hóa nhãn,
  protocol và làm sạch theo pipeline, rồi materialize cột scoring.
- **Score:** Dùng IF, AE, Ensemble và threshold đã lưu. Signed-log và
  z-score chạy bên trong `mapInPandas`. Action đếm từng cột kết quả buộc Spark
  thực thi scoring và cache output. **Write:** Ghi predictions phân vùng
  theo ngày từ output đã materialize. Đọc footer Parquet sau khi dừng đồng
  hồ để kiểm tra số dòng và marker `_SUCCESS`.
- **E2E:** Từ lúc bắt đầu scan đến khi ghi xong, gồm vật liệu hóa cache
  và chi phí chuyển pha. Không gồm training, baseline, tính AUC hay đọc lại.
  Vì có cache đĩa giữa các pha, đây là benchmark cô lập pha và sẽ khác
  runtime của một DAG chạy liền không materialize.
- Cache Spark được giải phóng sau mỗi lượt. `first-size-pass` là lượt đầu
  của kích thước trong cùng JVM sau probe nhỏ. `warm-repeat` là lượt sau
  trên cùng input. Cache hệ điều hành không bị xóa, nên không đo cold disk
  thực sự. Lượt đầu vẫn có thể chịu JIT và khởi động worker.
- So sánh số lõi chỉ thực hiện ở kích thước nhỏ nhất trong kế hoạch.
  Spark slots là giới hạn đồng thời của local master, trong khi máy vẫn
  có thể chạy tiến trình khác. Kết quả phản ánh máy, ổ đĩa và layout file
  này, cùng chi phí Spark khi workload nhỏ.

## Kiểm chứng và giới hạn

- Đủ 3 lượt cho mỗi cấu hình. Với full test, E2E lượt đầu là 107,42 giây
  và hai lượt warm là 105,90 và 105,57 giây. Lượt đầu 100.000 dòng có
  chênh lệch lớn hơn, 20,08 giây so với 15,46 và 13,76 giây.
- Trên 100.000 dòng, local[2] và local[4] lần lượt đạt speedup scoring
  1,66× và 2,30× so với local[1]. Speedup E2E tương ứng là 1,33× và 1,51×.
  Chỉ có 7 scoring partitions ở kích thước này. Chi phí đọc, materialize
  và lập lịch vẫn chiếm phần đáng kể trong E2E.
- Đã đọc lại cả ba Parquet full test. Số dòng theo ngày, cờ Ensemble,
  điểm hữu hạn và toàn bộ TP/FP/TN/FN của ba mô hình khớp lượt full test
  chính thức. Bằng chứng nằm trong `prediction_verification.json`.
- Log có cảnh báo broadcast task binary lớn, rút gọn chuỗi biểu diễn plan
  và cấu hình local dir. Lượt full test thứ ba có hai cảnh báo Parquet
  writer vượt ngưỡng cấp phát 95% heap, dẫn tới giảm row group còn 98,06%
  với 31 writers. Các lượt ghi hoàn tất và số dòng đọc lại khớp. Chi phí
  tài nguyên và lập lịch này nằm trong số đo, chưa được tách riêng.
- Các input nhỏ có layout Parquet được ghi lại khi chuẩn bị benchmark.
  Tốc độ không chỉ phụ thuộc số dòng mà còn phụ thuộc layout, phân vùng,
  cache hệ điều hành và nội dung dữ liệu. Không suy rộng speedup 100.000
  dòng sang full test hoặc cluster phân tán.
