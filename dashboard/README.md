# Traffic Lab — Streamlit Dashboard

Ứng dụng offline đọc các artifact nghiên cứu đã kiểm chứng của CSE-CIC-IDS2018.
Dashboard sử dụng CSV tổng hợp và PNG. Không cần khởi động Spark, nạp model
hoặc đọc predictions Parquet để mở bất kỳ trang nào.

## Chạy trong VS Code

Mở terminal PowerShell tại thư mục chứa `run_pipeline.py` và dùng môi trường
Python 3.11 của dự án:

```powershell
.\.conda-py311\python.exe -m pip install -r dashboard\requirements.txt
.\.conda-py311\python.exe -m streamlit run dashboard\app.py
```

Mở địa chỉ local mà Streamlit in trong terminal. Dùng bộ lọc ở sidebar để
chọn mô hình và ngày. Bộ lọc áp dụng cho dữ liệu tương tác. Các ảnh audit đã
lưu giữ nguyên phạm vi phân tích gốc và có chú thích riêng.

## Nguồn dữ liệu

| Trang | Nguồn |
|---|---|
| Overview | `metrics_by_day_independent.csv` của lượt full test. Volume lấy từ confusion counts. Alerts = TP + FP của mô hình đang chọn. |
| Model Evaluation | `independent_metrics.csv` cho toàn kỳ và bảng theo ngày cho ngày riêng. Random split xuất hiện trong phần đối chứng riêng. |
| Attack Analysis | `outputs/temporal_drift/attack_analysis.csv`, gồm HOIC, LOIC-HTTP, LOIC-UDP, Bot và Infilteration. |
| Temporal Drift | CSV feature, score và performance theo ngày của drift audit, cùng CSV và CDF của ranking reversal ngày 21/02. |
| Live Detection | SQLite của demo replay Kafka/Spark. Tự cập nhật mỗi 3 giây khi consumer ghi dữ liệu. |

Mặc định thư mục run là `.cleaning_check/test100_20261109_233550_850520`.
Các audit nằm trong `outputs`. Khi di chuyển các artifacts, đặt hai biến
sau thành đường dẫn tương ứng rồi chạy lại app:

```powershell
$env:DASHBOARD_FULL_RUN = 'D:\path\to\full_test_run'
$env:DASHBOARD_RESEARCH_DIR = 'D:\path\to\outputs'
```

Chọn các artifact cùng lượt model, snapshot và threshold. Dashboard báo
thiếu file hoặc sai schema thay vì thay thế bằng dữ liệu mô phỏng. Metrics
gộp lấy từ đánh giá full test, không tính bằng trung bình các ngày.
FPR, Recall và Precision hiển thị theo phần trăm. ROC-AUC và PR-AUC trong
caption hiển thị trên thang 0–1. Threshold là giá trị đã lưu từ Benign
Validation.

## Kiểm thử

```powershell
.\.conda-py311\python.exe -m pip install pytest
.\.conda-py311\python.exe -m pytest dashboard/tests -q
```

Unit test kiểm tra file thiếu, định dạng, schema và confusion counts.
Smoke test Streamlit AppTest mở cả năm trang, đổi ngày và mô hình, và
kiểm tra thông báo khi artifacts thiếu. Smoke test dữ liệu thật cần đủ
các CSV nguồn. Ứng dụng chỉ hỗ trợ kết quả offline của lượt đã chỉ định,
và không cung cấp drill-down theo flow hoặc tính lại predictions.
Riêng Live Detection đọc trạng thái demo replay. Xem [hướng dẫn streaming](../streaming/README.md).

Đã kiểm tra trên Python 3.11.18, Streamlit 1.67.0 và Altair 6.3.0.
Bộ test đạt 4/4. AppTest cần socket loopback của asyncio, nên hãy chạy
trong terminal VS Code khi sandbox không cho mở socket cục bộ.

## Ảnh minh họa

Ảnh chụp từ Streamlit chạy trên localhost với artifacts thật:
[Overview](screenshots/overview.png),
[Model Evaluation](screenshots/model_evaluation.png),
[Attack Analysis](screenshots/attack_analysis.png),
[Temporal Drift](screenshots/temporal_drift.png),
[Live Detection replay](../streaming/live_detection.png).

## Live Detection cập nhật

Bốn KPI gồm Total Processed Flows, Total Alerts, Alert Rate và Processing
Rate. Alert Rate = alerts / flow duy nhất. Processing Rate = flow mới ghi
trong 60 giây / 60. Tooltip giải thích từng chỉ số.

System Status kiểm tra metadata Kafka qua loopback tối đa mỗi 20 giây.
Offline nghĩa là không nhận phản hồi trong một giây ở lần kiểm tra đó.
Spark Consumer hiển thị Unknown vì chưa có heartbeat độc lập đáng tin cậy.
Thời điểm ghi SQLite gần nhất chỉ mô tả dữ liệu, không suy ra tiến trình còn chạy.

Hai biểu đồ truy vấn theo processed time của SQLite, gồm 60 ô phút
(phút hiện tại và 59 phút trước). Phút không có bản ghi có giá trị 0,
được suy ra từ truy vấn, không tạo event mới. Bảng Recent Alerts giữ tối
đa 25 dòng, sắp xếp mới nhất trước. ID rút gọn ở bảng và ID đầy đủ
trong mục sao chép/tải CSV. Mọi thời gian trên trang dùng Asia/Ho_Chi_Minh.
Các truy vấn chỉ đọc, dùng index thời gian sẵn có, không cần migration.

Đã kiểm chứng SQLite hiện tại giữ 1.000 flow và 602 alerts sau khi đọc UI,
Alert Rate 60,20%. Bộ unit/smoke liên quan đạt 13 passed, bao gồm cả năm
trang và các trạng thái. Chromium xác nhận tự cập nhật từ 1 lên 2 dòng
trên SQLite test riêng mà không reload. Chưa có heartbeat để phân biệt
Spark Running với Stopped, nên giao diện giữ Unknown cho cả hai trường hợp.

[Ảnh giao diện mới](screenshots/live_detection_improved.png) ·
[Bảng cảnh báo và ID đầy đủ](screenshots/live_recent_alerts.png)
