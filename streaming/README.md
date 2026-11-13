# Local Kafka Replay Demo

```text
Parquet snapshot → JSON replay producer → Kafka (loopback)
 → Spark Structured Streaming → SQLite predictions → Traffic Lab Live Detection
```

Đây là replay từ CSE-CIC-IDS2018, không phải thu thập mạng trực tuyến.
Producer xác minh checksum của file snapshot, giữ `record_id` làm Kafka key
và event ID, thêm timestamp replay UTC (epoch milliseconds) và `source_day`.
JSON chỉ chứa các cột model cần. Broker và controller bind loopback.
Producer và consumer từ chối bootstrap ngoài `127.0.0.1` và kiểm tra địa
chỉ broker được quảng bá trước khi xử lý. Tải dependency chỉ tải phần mềm.

Consumer đọc schema JSON từ danh sách feature của model. Producer tạo
protocol indicators giống ingest, chuyển giá trị không hữu hạn sang
JSON null. `src.scoring.score_pandas` áp dụng cleaning, signed-log,
normalization, clipping, IF/AE, Ensemble và threshold từ artifact chính
thức. SQLite lưu event ID, thời điểm replay, ngày nguồn, score, alert,
top feature, thời điểm xử lý và Kafka partition/offset.

## Chạy trong VS Code

Mở các terminal PowerShell tại thư mục chứa `run_pipeline.py`. Dùng môi trường
`.conda-py311` đã chạy offline pipeline. Cài client và dependency dashboard:

```powershell
.\.conda-py311\python.exe -m pip install -r streaming\requirements.txt -r dashboard\requirements.txt
```

**Terminal 1 — Kafka.** Tải [Apache Kafka binary](https://kafka.apache.org/downloads/),
kiểm tra SHA-512 trước khi giải nén. Dùng Kafka 4.4.0 với JDK 17 hoặc
mới hơn cho broker. Trên máy đã kiểm chứng demo, chạy lệnh sau từ thư mục
chứa `run_pipeline.py`. Trên máy khác, thay hai đường dẫn bằng vị trí cài đặt thực tế.

```powershell
.\streaming\start_local_broker.ps1 `
  -KafkaHome '.\.cleaning_check\kafka_demo_tools\kafka_2.13-4.4.0' `
  -JavaHome 'C:\Java\openjdk-17.0.0.1+2_windows-x64_bin\jdk-17.0.0.1'
```

**Terminal 2 — topic và Spark.** Tạo topic một partition một lần trên
broker local:

```powershell
.\.conda-py311\python.exe -c "from confluent_kafka.admin import AdminClient, NewTopic; a=AdminClient({'bootstrap.servers':'127.0.0.1:9092'}); a.create_topics([NewTopic('traffic-replay',1,1)])['traffic-replay'].result(15)"
$env:PYSPARK_SUBMIT_ARGS = '--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8 pyspark-shell'
.\.conda-py311\python.exe -X utf8 -u -m streaming.consumer
```

Spark dùng phiên bản **3.5.8 / Scala 2.12** của dự án. Lần đầu
`--packages` tải connector. Nếu đã chuẩn bị các JAR tương ứng, có thể dùng
`--jars` với danh sách URI `file:///D:/...` phân cách bằng dấu phẩy. Trên
Windows tránh đường dẫn `D:/...` trần trong `--jars` vì Hadoop có thể đọc
`D` như filesystem scheme. Connector cần Spark Kafka SQL, Spark Kafka
token provider, Kafka clients 3.4.1 và commons-pool2 2.11.1 ngoài Spark
distribution đã có. [Spark Kafka connector](https://spark.apache.org/docs/3.5.8/structured-streaming-kafka-integration.html)

**Terminal 3 — producer.** Mặc định đọc snapshot full test và model chính
thức của project. Đổi `--day`, `--limit` và `--rate` khi trình bày:

```powershell
.\.conda-py311\python.exe -X utf8 -u -m streaming.replay --day 2018-02-21 --limit 200 --rate 100
```

**Terminal 4 — Dashboard.**

```powershell
.\.conda-py311\python.exe -m streamlit run dashboard\app.py
```

Mở **Live Detection**. Trang đọc SQLite mỗi 3 giây, hiển thị tổng ID duy nhất,
alerts, flow/giây trong cửa sổ trượt 60 giây, số cảnh báo theo phút trong 60
phút gần nhất và tối đa 25 cảnh báo mới nhất. Để dùng thư mục khác, truyền
`--state <thư_mục>` cho consumer và đặt `TRAFFIC_REPLAY_DB` thành
`<thư_mục>/predictions.sqlite` trước khi chạy Streamlit.

## Khởi động lại và giới hạn

- Giữ cùng thư mục `streaming/runtime`, topic, broker và model. Chỉ chạy một
  consumer trên mỗi cặp checkpoint–store. Đổi topic/model thì dùng thư mục
  state mới. Không xóa hoặc tái tạo topic trong khi giữ checkpoint cũ.
- Spark lưu offsets ở checkpoint. Nếu commit SQLite thành công rồi Spark
  chạy lại batch, khóa chính `event_id` chặn ghi trùng. Predictions,
  counter và rollup được commit cùng transaction. Nếu mất store, cần khôi
  phục cùng checkpoint, hoặc tạo một state mới để replay lại.
- Consumer chạy `local[2]`, micro-batch mặc định 5 giây và giới hạn
  1.000 Kafka offsets mỗi trigger. Batch score được collect về driver để
  ghi SQLite local. Throughput thực tế chịu chi phí startup Python, Kafka,
  Spark, scoring và I/O SQLite. Flow/giây trên UI là số flow mới đã commit
  trong 60 giây chia 60, không phải công suất inference thuần.
- Payload thiếu trường hoặc sai schema làm batch thất bại. Kiểm tra log và
  record theo Kafka partition/offset trước khi khôi phục. `failOnDataLoss=true`
  để lỗi retention/offset được báo. Demo chưa có DLQ, cơ chế HA hoặc chính
  sách dọn state dài hạn.
- Các feature và threshold giữ nguyên nên chất lượng phát hiện có cùng
  giới hạn như model offline. Replay ngày 21/02 là minh họa luồng xử lý.

## Kiểm thử

```powershell
.\.conda-py311\python.exe -m pytest streaming/tests dashboard/tests -q
# Để chạy integration test, bật Kafka local và cấu hình connector ở trên.
$env:REPLAY_TEST_BROKER = '127.0.0.1:9092'
.\.conda-py311\python.exe -m pytest streaming/tests/test_kafka_integration.py -q -s
```

Integration test tạo topic riêng, gửi flow thật từ snapshot, chạy Spark,
restart từ checkpoint sau khi gửi lại các ID cũ và thêm ID mới, đối chiếu
SQLite với Spark batch preprocessing + `score_pandas`, và kiểm tra số liệu
trên trang Live Detection.

## Kết quả kiểm chứng

Kafka 4.4.0 KRaft chạy với Java 17 trên loopback. Consumer dùng Spark 3.5.8
và `local[2]`. Bộ unit/integration và dashboard đạt **9 passed**.

- Lượt đầu Kafka xác nhận 200 messages. Sau restart, Kafka nhận thêm 230,
  Spark tiếp tục từ batch 2, loại 200 ID trùng và lưu thêm 30 ID.
- Đối chiếu 230 dòng với batch Spark preprocessing cho vector đầu vào khớp
  từng phần tử. Score lệch tối đa **2,57492×10⁻⁵** trong dung sai tuyệt đối
  **3×10⁻⁵**. Cả 230 alert flags khớp. Sai lệch tương thích với tính toán
  AE float32 trên các batch khác kích thước. Với flow rất sát ngưỡng,
  cần kiểm tra riêng nếu yêu cầu quyết định khớp tuyệt đối.
- Chromium mở Live Detection ở 230 dòng. Sau lần replay 240 messages tiếp
  theo, trang tự tăng lên **240 flow duy nhất, 180 alerts** mà không reload.
  Tổng ba lượt có 670 messages và 430 lần ghi trùng bị bỏ qua.
- Batch 100 messages trong lượt kiểm chứng mất khoảng 1,86–2,95 giây
  cho scoring và ghi SQLite, tương đương khoảng 34–54 messages/giây
  trong pha đó. Trung vị từ timestamp replay tới thời điểm xử lý ghi
  trong store là 14,64 giây. Test gửi dữ liệu trước khi khởi động query
  `--available-now`, nên số đo latency bao gồm khởi động và backlog.
  Chưa đo tải liên tục hoặc độ trễ phân vị cao.
- Windows ghi lỗi dọn JAR tạm lúc Spark tắt. Checkpoint đã commit,
  SQLite đọc lại được và restart tiếp tục đúng offset. Nếu cần thu hồi
  dung lượng, chỉ dọn thư mục Spark tạm sau khi tiến trình đã dừng.

[Số liệu kiểm chứng](verification.json) · [Ảnh Live Detection](live_detection.png)
