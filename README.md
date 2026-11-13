# Network Anomaly Detection — Hybrid VS Code (`.ipynb` + `.py`)

Project nghiên cứu phát hiện bất thường trên **CSE-CIC-IDS2018**, gồm pipeline PySpark, Isolation Forest, Autoencoder, các audit, Traffic Lab và demo Kafka replay. Đã kiểm chứng full test 4.653.501 flow. Dữ liệu Parquet gốc được giữ ngoài Git.

## Nội dung đưa lên GitHub

Repository giữ mã nguồn, notebook đã xóa output, unit/integration tests, model nhỏ,
CSV tổng hợp, báo cáo và biểu đồ nghiên cứu. Dataset, predictions Parquet, môi trường
Python, cache, Kafka broker, checkpoint và SQLite runtime được bỏ qua bằng `.gitignore`.
Các file local vẫn có thể được dùng để chạy demo trên máy hiện tại.

Khi clone trên máy mới, cần tạo môi trường Python và tải dataset theo phần chuẩn bị.
Dashboard đọc các CSV từ lượt full test chỉ định qua `DASHBOARD_FULL_RUN` và
`DASHBOARD_RESEARCH_DIR`. Các đường dẫn `.cleaning_check/...` là artifacts local
của máy nghiên cứu, không có trong Git. Model và snapshot nguồn cho streaming
cũng cần được tạo hoặc cung cấp lại theo hướng dẫn tương ứng.

- [Hướng dẫn Traffic Lab](dashboard/README.md)
- [Hướng dẫn Kafka replay](streaming/README.md)
- Báo cáo và CSV nghiên cứu ở `outputs/temporal_drift/`, `outputs/ablation_study/`,
  `outputs/benchmark/`, `outputs/fpr_sensitivity/` và các thư mục audit.

Notebook gốc trước khi xóa output được sao lưu local trong
`.cleaning_check/github_cleanup_20261113_162632/`.

## Ý tưởng tổ chức

- **`src/*.py`**: nguồn hàm dùng chung, tái sử dụng cho notebook, CLI, dashboard hoặc streaming về sau.
- **`notebooks/*.ipynb`**: các bước học, chạy thử, quan sát bảng và hình.
- **`run_pipeline.py`**: chạy toàn bộ pipeline. Notebook 03 gọi cùng hàm `main()` để tránh copy/đồng bộ code.
- Các notebook **độc lập về kernel**; không dùng biến của notebook khác. Notebook 03 tạo artifacts trên đĩa; notebooks 04–05 đọc lại.

## Bản đồ từ notebook Kaggle gốc

| Bước cũ | Module `.py` | Notebook xem/chạy |
|---|---|---|
| 0. Config, parse ngày, Spark | `src/config.py`, `src/utils.py`, `src/spark_session.py` | 01, 03 |
| 1. Ingest, curated Parquet | `src/data_io.py` | 01, 03 |
| 2. Data quality, EDA | `src/preprocessing.py` | 01, 02 |
| 3. Feature selection & scaling | `src/preprocessing.py` | 02, 03 |
| 4. Isolation Forest & Autoencoder | `src/models.py` | 03 |
| 5. Calibrate thresholds, ensemble | `src/scoring.py` | 03, 04 |
| 6. Inference + evaluation + figures | `src/scoring.py`, `src/evaluation.py`, `src/reports.py` | 03, 04 |
| 7. GBT baseline & random split | `src/baselines.py` | 03 (05 đọc kết quả) |
| 8. Artifacts & batch inference | `run_pipeline.py`, `predict_batch.py`, `src/scoring.py` | 03, 05 |

## Chuẩn bị trên VS Code

1. Cài **Python 3.11**, **Java JDK 17**, extension **Python** và **Jupyter** trong VS Code.
2. Mở folder này làm **workspace root** trong VS Code.
3. Tạo môi trường ảo và cài thư viện:

Windows PowerShell:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name network-anomaly --display-name "Python (network-anomaly)"
```
macOS/Linux:
```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name network-anomaly --display-name "Python (network-anomaly)"
```
4. Chọn Python interpreter trong `.venv`, mở notebook và chọn **Select Kernel → Python (network-anomaly)** hoặc `.venv`.
5. Đặt tất cả file Parquet của Kaggle `dhoogla/csecicids2018` vào **`data/raw/`**, giữ nguyên tên có hậu tố `_TrafficForML_CICFlowMeter.parquet`.
6. Chạy notebooks **01 → 02 → 03 → 04 → 05**, hoặc chạy **03 trực tiếp** nếu chỉ muốn full pipeline.

**Chạy full pipeline bằng terminal** (ở project root):
```bash
python run_pipeline.py
```

**Windows PowerShell máy yếu** (trước khi chạy; notebook phải restart kernel để nhận biến mới):
```powershell
$env:SAMPLE="30000"
$env:TEST_FRAC="0.05"
$env:WITH_SUPERVISED="0"
$env:SPARK_DRIVER_MEMORY="4g"
python run_pipeline.py
```

**Lưu ý:** `SKIP_INGEST=1` chỉ sử dụng khi curated Parquet đã có và cùng schema/dataset. Khi thay dataset hoặc chỉnh ingestion phải build lại curated.

## Output

### Khi gặp `Python worker failed to connect back`

Đây là lỗi khởi động/kết nối Python worker dùng trong `mapInPandas`; cảnh báo
Arrow từ `toPandas()` là lỗi phát sinh tiếp theo. Tắt Arrow cho `toPandas`
không bỏ được worker của `mapInPandas`.

`create_spark()` chỉ định worker dùng `sys.executable` của pipeline và chạy
thử một batch Arrow trước khi ingest/train. Sau khi cập nhật mã, **restart
kernel** và chọn môi trường Python 3.11 đã cài `requirements.txt`, rồi chạy
lại notebook 03. Nếu chạy terminal, dùng trực tiếp:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py
```

Nếu kiểm tra worker vẫn lỗi, xem traceback đầy đủ và stderr của worker để
kiểm tra thư viện thiếu, Python không khởi động được hoặc firewall/antivirus
chặn kết nối local giữa Java và Python.

Windows + Python 3.12 trở lên có thể gặp lỗi socket worker
[SPARK-53759](https://issues.apache.org/jira/browse/SPARK-53759) trên Spark
3.5.8. Môi trường riêng `.conda-py311` dùng Python 3.11 để tránh lỗi này.
Chọn **Select Kernel → Python Environments → `.conda-py311/python.exe`**
trong notebook và restart kernel; cấu hình interpreter mặc định không tự
thay kernel notebook đã chọn. Chạy terminal bằng:

```powershell
.\.conda-py311\python.exe run_pipeline.py
```

- `data/curated/cse2018_curated/`: dataset sau ingestion, partition day.
- `outputs/models/model.joblib` và `feature_spec.joblib`: mô hình và preprocessing.
- `outputs/metrics/*.json, *.csv`: kết quả đánh giá.
- `outputs/predictions/test_scored/`: prediction Parquet.
- `outputs/figures/*.png`: biểu đồ.

## Những giới hạn hiện có

- Đây là Spark **local** và scikit-learn train trên dữ liệu lấy mẫu, không phải distributed model training.
- `TEST_FRAC=0.25` mặc định tạo snapshot Parquet lấy mẫu Bernoulli theo thứ tự dòng trong file nguồn. Pipeline tái sử dụng snapshot trong `data/curated/test_snapshots/` khi SHA-256 nguồn, seed và fraction khớp. Manifest đi kèm ghi schema, số dòng và checksum của từng file. Bản sao manifest nằm trong `metrics/test_snapshot_manifest.json`.
- HistGradientBoosting random split 70/15/15 fit feature selection, scaling và classifier trên train của baseline. Ngưỡng lấy từ Benign Validation của cùng split. Đây là đối chứng với dữ liệu trộn ngày, có thể chứa các loại attack giống nhau trong các tập.

Để lưu một lượt chạy riêng trên PowerShell, chọn thư mục đầu ra mới trước khi chạy:

```powershell
$env:OUTPUT_DIR = Join-Path (Get-Location) ("outputs/repro_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
.\.conda-py311\python.exe -X utf8 run_pipeline.py
```
- `notebooks/02_preprocessing.ipynb` là notebook kiểm tra quy trình; notebook 03 fit lại preprocessing trong full pipeline.
- Kafka replay và Streamlit chạy trên máy local. Hướng dẫn và giới hạn kiểm thử nằm trong README của từng thành phần.

## Test cú pháp không cần dataset

```bash
python -m compileall -q src run_pipeline.py predict_batch.py
python -m pip install pytest
python -m pytest -q tests
```
