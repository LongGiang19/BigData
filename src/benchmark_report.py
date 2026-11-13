"""Render benchmark artifacts from completed trials, including partial runs."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .benchmarking import STAGES


def write_report(output):
    output = Path(output)
    runs = pd.read_json(output / "trials.jsonl", lines=True)
    runs.to_csv(output / "benchmark_results.csv", index=False)
    measures = [f"{stage}_{unit}" for stage in STAGES for unit in ("seconds", "flows_per_second")]
    grouped = runs.groupby(["master", "rows"], sort=False)
    summary = grouped[measures].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary = summary.join(grouped.agg(repeats=("repeat", "count"),
                                      read_partitions=("read_partitions", "first"),
                                      scoring_partitions=("scoring_partitions", "first"),
                                      spark_slots=("spark_slots", "first"),
                                      driver_memory=("driver_memory", "first"))).reset_index()
    summary.to_csv(output / "benchmark_summary.csv", index=False)
    scale = summary[summary.master.eq("local[*]")].sort_values("rows")
    if not scale.empty:
        for unit, label, filename in (
            ("flows_per_second", "Flows / second", "throughput_by_size.png"),
            ("seconds", "Elapsed seconds", "processing_time_by_size.png"),
        ):
            fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
            for stage in STAGES:
                ax.errorbar(scale.rows, scale[f"{stage}_{unit}_mean"],
                            yerr=scale[f"{stage}_{unit}_std"].fillna(0), marker="o",
                            capsize=3, label=stage.replace("_", " "))
            ax.set(xscale="log", xlabel="Input flows", ylabel=label,
                   title="Spark local[*] · mean ± sample standard deviation")
            ax.set_xticks(scale.rows, [f"{n:,}" for n in scale.rows])
            ax.grid(alpha=.25)
            ax.legend()
            fig.savefig(output / filename, dpi=170)
            plt.close(fig)
    cores = summary[summary.master.isin(["local[1]", "local[2]", "local[4]"])].sort_values("spark_slots").copy()
    if 1 in cores.spark_slots.to_list():
        base = cores[cores.spark_slots.eq(1)].iloc[0]
        fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
        for stage in ("scoring", "end_to_end"):
            cores[f"{stage}_speedup"] = base[f"{stage}_seconds_mean"] / cores[f"{stage}_seconds_mean"]
            ax.plot(cores.spark_slots, cores[f"{stage}_speedup"], "o-", label=stage.replace("_", " "))
        ax.plot(cores.spark_slots, cores.spark_slots, ":", color="gray", label="ideal linear")
        ax.set(xlabel="Spark local execution slots", ylabel="Speedup vs local[1]",
               xticks=cores.spark_slots, title=f"Core comparison · {int(base.rows):,} flows")
        ax.grid(alpha=.25)
        ax.legend()
        fig.savefig(output / "speedup_by_cores.png", dpi=170)
        plt.close(fig)
        cores.to_csv(output / "core_speedup.csv", index=False)
    env = runs.iloc[0]
    rows = []
    for row in summary.itertuples(index=False):
        timings = [f"{getattr(row, f'{s}_seconds_mean'):.2f} ± {getattr(row, f'{s}_seconds_std'):.2f}"
                   for s in STAGES]
        rows.append(f"| {row.master} | {row.rows:,} | {row.repeats} | {row.scoring_partitions} | "
                    + " | ".join(timings)
                    + f" | {row.scoring_flows_per_second_mean:,.0f} | {row.end_to_end_flows_per_second_mean:,.0f} |")
    table = "\n".join(rows)
    report = f"""# Spark end-to-end inference benchmark

Đã ghi **{len(runs)} lượt hoàn tất**. Spark chạy **local trên một máy**.
Spark {env.spark_version}, Java {env.java}, Python {env.python},
{env.os}, {env.logical_cpu} CPU logic, driver {env.driver_memory},
{env.shuffle_partitions} shuffle partitions, Arrow tối đa {env.arrow_batch_rows} dòng/batch.
Mỗi Python worker dùng 1 BLAS thread. Các cấu hình master dùng tiến trình JVM riêng.

| Master | Flows | Lượt | Scoring partitions | Read (s) | Prep (s) | Score (s) | Write (s) | E2E (s) | Score flow/s | E2E flow/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

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
"""
    (output / "benchmark_report.md").write_text(report, encoding="utf-8")
