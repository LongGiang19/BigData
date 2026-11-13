"""Evaluate the predeclared ablations and render their comparison."""
import argparse
import json
from pathlib import Path
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def measure(y, score, threshold):
    hit = score > threshold
    tp = int(np.count_nonzero(hit & y))
    fp = int(np.count_nonzero(hit & ~y))
    pos = int(y.sum())
    neg = len(y) - pos
    p, r, _ = precision_recall_curve(y, score)
    return dict(n=len(y), TP=tp, FP=fp, TN=neg-fp, FN=pos-tp,
                recall=tp/pos, fpr=fp/neg, precision=tp/(tp+fp) if tp+fp else 0.,
                f1=2*tp/(pos+tp+fp), roc_auc=roc_auc_score(y, score), pr_auc=auc(r, p))


def read_predictions(path):
    cols = ["day", "label_raw", "y", "s_iso", "s_ae"]
    return ds.dataset(path, format="parquet", partitioning="hive").to_table(columns=cols).to_pandas()


def ensemble(data, norms, weight):
    m0, s0, m1, s1 = norms
    return (weight * (np.log(data.s_iso.to_numpy() + 1e-9) - m0) / s0
            + (1-weight) * (np.log(data.s_ae.to_numpy() + 1e-9) - m1) / s1)


def run(out):
    protocol = json.loads((out / "experiment_config.json").read_text(encoding="utf-8"))
    if (out / "ablation_results.csv").exists():
        raise FileExistsError("Ablation report already exists")
    official = Path(protocol["official_reference"])
    official_predictions = official / "outputs/predictions/test_scored"
    official_labels = ds.dataset(official_predictions, format="parquet", partitioning="hive").to_table(
        columns=["day", "label_raw", "y"]).to_pandas()
    counts = official_labels.groupby(["day", "label_raw", "y"]).size()
    del official_labels
    assert counts.sum() == 4653501
    rows, day_rows, attack_rows = [], [], []
    for variant in protocol["variants"]:
        folder = out / variant
        art = joblib.load(folder / "model.joblib")
        timing = json.loads((folder / "runtime.json").read_text(encoding="utf-8"))
        assert timing["features"] == len(art["cols"])
        val = pd.read_parquet(folder / "validation_scores.parquet")
        assert len(val) == 446619
        pred = read_predictions(folder / "predictions")
        pd.testing.assert_series_equal(pred.groupby(["day", "label_raw", "y"]).size(), counts)
        y = pred.y.to_numpy() == 1
        days = pred.groupby("day").indices
        attacks = pred.loc[y].groupby("label_raw").groups
        options = [(variant, "ensemble", .5)]
        if variant == "paired_baseline":
            options += [("if_only", "s_iso", None), ("ae_only", "s_ae", None)]
            options += [(f"ensemble_if_{int(w*100)}", "ensemble", w)
                        for w in protocol["if_weights"] if w != .5]
        for name, kind, weight in options:
            start = time.perf_counter()
            if kind == "ensemble":
                validation = ensemble(val, art["score_norm"], weight)
                score = ensemble(pred, art["score_norm"], weight)
            else:
                validation, score = val[kind].to_numpy(), pred[kind].to_numpy()
            assert np.isfinite(score).all() and np.isfinite(validation).all()
            threshold = float(np.quantile(validation, 1-protocol["fpr_target"]))
            if weight == .5 or kind != "ensemble":
                key = "s_ens" if kind == "ensemble" else kind
                np.testing.assert_allclose(threshold, art["thresholds"][key], rtol=0, atol=2e-12)
            meta = dict(experiment=name, trained_variant=variant, if_weight=weight,
                        feature_count=len(art["cols"]), threshold=threshold,
                        validation_fpr=float((validation > threshold).mean()))
            summary = measure(y, score, threshold)
            for day, idx in days.items():
                day_rows.append(dict(**meta, day=day, **measure(y[idx], score[idx], threshold)))
            for attack, idx in attacks.items():
                tp = int(np.count_nonzero(score[idx] > threshold))
                attack_rows.append(dict(**meta, attack=attack, n=len(idx), TP=tp,
                                        FN=len(idx)-tp, recall=tp/len(idx)))
            rows.append(dict(**meta, **summary, **timing, evaluation_seconds=time.perf_counter()-start))
            print("EVALUATED", name, summary["recall"], summary["fpr"], flush=True)
        del pred, y, val, score
    results = pd.DataFrame(rows)
    daily = pd.DataFrame(day_rows)
    per_attack = pd.DataFrame(attack_rows)
    key = "experiment"
    cols = ["TP", "FP", "TN", "FN"]
    pd.testing.assert_frame_equal(daily.groupby(key)[cols].sum(), results.set_index(key)[cols].sort_index())
    pd.testing.assert_frame_equal(per_attack.groupby(key)[["TP", "FN"]].sum(),
                                  results.set_index(key)[["TP", "FN"]].sort_index())
    results.to_csv(out / "ablation_results.csv", index=False)
    daily.to_csv(out / "ablation_by_day.csv", index=False)
    per_attack.to_csv(out / "ablation_by_attack.csv", index=False)
    for metrics, filename in [
        (("recall", "fpr", "precision"), "operating_metrics.png"),
        (("roc_auc", "pr_auc"), "ranking_metrics.png"),
    ]:
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5), layout="constrained")
        for ax, metric in zip(axes, metrics):
            ax.barh(results.experiment, results[metric], color="#2563eb")
            ax.invert_yaxis()
            ax.set_xlabel(metric)
            ax.grid(axis="x", alpha=.2)
        fig.savefig(out / filename, dpi=165)
        plt.close(fig)
    official_metrics = json.loads((official / "verification_metrics.json").read_text(encoding="utf-8"))
    pd.DataFrame(official_metrics["independent_metrics"][:3]).to_csv(
        out / "official_reference.csv", index=False)
    write_markdown(out, protocol, results, daily, per_attack, official_metrics)
    return results


def write_markdown(out, protocol, results, daily, attacks, official):
    def table(headers, values):
        return "\n".join(["| " + " | ".join(headers) + " |",
                          "|" + " --- |" * len(headers),
                          *["| " + " | ".join(map(str, row)) + " |" for row in values]])

    sample_rows = ds.dataset(out / "shared_training_sample.parquet", format="parquet").count_rows()
    corr_rows = ds.dataset(out / "shared_correlation_sample.parquet", format="parquet").count_rows()
    base = results.set_index("experiment").loc["paired_baseline"]
    metrics = [[r.experiment, r.feature_count, f"{r.threshold:.8g}",
                f"{r.validation_fpr:.4%}", f"{r.recall:.4%}", f"{r.precision:.4%}",
                f"{r.fpr:.4%}", f"{r.f1:.5f}", f"{r.roc_auc:.5f}", f"{r.pr_auc:.5f}"]
               for r in results.itertuples()]
    contrasts = []
    for r in results.itertuples():
        if r.experiment == "paired_baseline":
            continue
        contrasts.append([r.experiment, f"{100*(r.recall-base.recall):+.4f}",
                          f"{100*(r.fpr-base.fpr):+.4f}", f"{r.pr_auc-base.pr_auc:+.5f}"])
    runtimes = results.drop_duplicates("trained_variant")
    timing_rows = [[r.trained_variant, f"{r.fit_preprocessing_seconds:.1f}", f"{r.training_seconds:.1f}",
                    f"{r.calibration_seconds:.1f}", f"{r.test_score_write_seconds:.1f}",
                    f"{r.total_seconds:.1f}"] for r in runtimes.itertuples()]
    ref = next(r for r in official["independent_metrics"] if r["model"] == "Ensemble AE+IF")
    daily_ranges = [[name, f"{group.fpr.min():.4%}–{group.fpr.max():.4%}",
                     f"{group.recall.min():.4%}–{group.recall.max():.4%}"]
                    for name, group in daily.groupby("experiment", sort=False)]
    attack_rows = [[r.experiment, r.attack, r.n, r.TP, f"{r.recall:.4%}"]
                   for r in attacks[attacks.attack.str.contains("HOIC|LOIC-HTTP", regex=True)].itertuples()]
    seconds = json.loads((out / "runtime.json").read_text(encoding="utf-8"))["seconds"]
    report = f"""# Ablation Study — Network Traffic Anomaly Detection

Cả tám điểm vận hành được đánh giá trên cùng **4.653.501 dòng full test**.
Số dòng theo ngày, loại nhãn và nhãn nhị phân khớp predictions chính thức
ở cả bốn lượt chấm điểm. Snapshot nguồn được xác nhận bằng checksum.

Train 14–15/02, Benign Validation 16/02 và bảy ngày test giữ nguyên.
Mỗi feature spec fit trên **1.268.743 dòng Benign Train**. Mẫu fit model chung
gồm **{sample_rows:,}** dòng và mẫu tính tương quan chung gồm **{corr_rows:,}**
dòng, đều được lưu. Seed {protocol['seed']}. Ngưỡng là phân vị
{1-protocol['fpr_target']:.1%} của **446.619 Benign Validation** cho từng điểm.
Các cấu hình được khai báo trước khi chấm test.

## Kết quả gộp

{table(["Biến thể", "Features", "Threshold", "Val FPR", "Recall", "Precision", "Test FPR", "F1", "ROC-AUC", "PR-AUC"], metrics)}

PR-AUC tính bằng tích phân hình thang trên đường PR đầy đủ. Threshold của
IF-only, AE-only và Ensemble có đơn vị khác nhau. Các dòng IF-only, AE-only
và Ensemble trọng số 25/75, 75/25 dùng lại component scores của paired baseline.
Mỗi trọng số được hiệu chuẩn riêng trên Benign Validation.

Paired baseline áp dụng signed-log, lọc tương quan 0,98 và clipping ±10,
rồi train trên cùng mẫu với các ablation. Tham chiếu chính thức đã có Recall
**{ref['recall']:.4%}**, FPR **{ref['fpr']:.4%}** và PR-AUC **{ref['pr_auc']:.5f}**.
Mẫu train và cách giữ mẫu tương quan có thể khác giữa lượt chính thức và paired baseline.
Dùng paired baseline để phân tích đối chiếu trong thí nghiệm này.

{table(["Biến thể", "Δ Recall (điểm %)", "Δ FPR (điểm %)", "Δ PR-AUC"], contrasts)}

`no_signed_log` fit lại thống kê và lọc tương quan trên thang đo gốc, nên hiệu ứng
bao gồm thay đổi các feature được giữ. `no_correlation_filter` vẫn bỏ feature hằng
và hiếm. Khi số feature thay đổi, AE dùng quy tắc kích thước lớp theo số feature
của pipeline. `clip_5` dùng ±5 nhất quán ở train, validation và test, so với ±10.

## Theo ngày và loại tấn công

{table(["Biến thể", "Dải FPR theo ngày", "Dải Recall theo ngày"], daily_ranges)}

Ví dụ hai nhóm tấn công lớn. CSV theo loại ghi support, TP, FN và Recall.
Nhóm attack chỉ chứa mẫu dương, nên FPR và ROC-AUC theo loại cần một tập Benign
tham chiếu khác và không được tính trong bảng này.

{table(["Biến thể", "Attack", "Số dòng", "TP", "Recall"], attack_rows)}

## Runtime

Spark local, driver {protocol['driver_memory']}, {protocol['shuffle_partitions']}
shuffle partitions, BLAS một thread. Tổng lượt chuẩn bị và train/chấm điểm Spark
**{seconds:.1f} giây**, cộng thời gian tính metrics độc lập trong CSV.

{table(["Lượt train", "Fit prep (s)", "Train (s)", "Calibration (s)", "Score + write (s)", "Tổng (s)"], timing_rows)}

Runtime fit và scoring trong các dòng tái dùng là chi phí của lượt cha
`paired_baseline`. Chi phí riêng để tính metrics nằm ở `evaluation_seconds`.

## Phạm vi diễn giải

Đây là một seed, một split thời gian và một snapshot. Recall, FPR, Precision và
ranking cần đọc cùng nhau. Thay đổi điểm vận hành trên một ngày calibration có
thể chuyển sang các ngày test với FPR khác. Kết quả final test chỉ dùng để mô tả.
Nếu cần chọn trọng số, hãy khóa một split development theo thời gian riêng trước
khi đánh giá trên một tập test được giữ kín.
"""
    (out / "ablation_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
