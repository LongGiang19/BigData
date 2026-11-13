import os
import sys
import json
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
out = Path(__file__).resolve().parent
root = out.parents[1]
sys.path.insert(0, str(root))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn import metrics as sk
from src.preprocessing import to_z
from src.scoring import calibrate

run = root / ".cleaning_check/full_20261109_215910_140315"
source = run / "outputs"
art = joblib.load(source / "models/model.joblib")
config = json.loads((source / "metrics/config.json").read_text())
published = json.loads((source / "metrics/metrics.json").read_text(encoding="utf-8"))
thresholds = art["thresholds"]
assert thresholds == config["thresholds"]
names = {"s_iso": "Isolation Forest", "s_ae": "Autoencoder", "s_ens": "Ensemble"}
pred = ds.dataset(source / "predictions/test_scored", format="parquet",
                  partitioning="hive").to_table().to_pandas()
expected = pred.label_raw.str.strip().str.upper().ne("BENIGN").astype(int)
assert pred.label_raw.notna().all() and expected.equals(pred.y.astype(int))
assert set(pred.day) == set(config["test_days"])
assert np.isfinite(pred[list(names)]).all().all()
assert (pred[["s_iso", "s_ae"]] >= 0).all().all()
m0, s0, m1, s1 = art["score_norm"]
assert s0 > 0 and s1 > 0
ensemble = .5 * (np.log(pred.s_iso + 1e-9) - m0) / s0 + .5 * (np.log(pred.s_ae + 1e-9) - m1) / s1
ensemble_error = float(np.max(np.abs(ensemble - pred.s_ens)))
np.testing.assert_allclose(ensemble, pred.s_ens, atol=1e-12, rtol=0)
assert (pred.is_anomaly == (pred.s_ens > thresholds["s_ens"])).all()

def measure(frame, score):
    y, values = frame.y.to_numpy(), frame[score].to_numpy()
    alert = values > thresholds[score]
    tn, fp, fn, tp = map(int, sk.confusion_matrix(y, alert, labels=[0, 1]).ravel())
    both = bool((tp + fn) and (tn + fp))
    p, r, _ = sk.precision_recall_curve(y, values) if both else (None, None, None)
    return dict(n=len(y), TP=tp, FP=fp, TN=tn, FN=fn,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                fpr=fp/(tn+fp) if tn+fp else None,
                roc_auc=sk.roc_auc_score(y, values) if both else None,
                pr_auc=sk.auc(r, p) if both else None,
                average_precision=sk.average_precision_score(y, values) if both else None,
                prevalence=float(y.mean()))

overall = {key: measure(pred, key) for key in names}
for key, row in overall.items():
    label = "Ensemble AE+IF" if key == "s_ens" else names[key]
    for metric in ("recall", "fpr"):
        np.testing.assert_allclose(row[metric], published[label]["ALL"][metric], atol=1e-12)
pd.DataFrame([dict(model=names[k], threshold=thresholds[k], **v)
              for k, v in overall.items()]).to_csv(out / "overall.csv", index=False)
pd.DataFrame([dict(day=day, model=name, **measure(frame, key))
              for day, frame in pred.groupby("day")
              for key, name in names.items()]).to_csv(out / "by_day.csv", index=False)
attacks = pred[pred.y == 1]
pd.DataFrame([dict(attack=label, model=name, n=len(group),
                   detected=int((group[key] > thresholds[key]).sum()),
                   recall=float((group[key] > thresholds[key]).mean()))
              for label, group in attacks.groupby("label_raw")
              for key, name in names.items()]).to_csv(out / "by_attack.csv", index=False)
pred.groupby(["day", "label_raw", "y"]).size().rename("n").to_csv(out / "labels.csv")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
for ax, (key, name) in zip(axes, names.items()):
    values = pred[key].to_numpy()
    # A logarithmic scale exposes the AE's concentrated body and long tail.
    log_axis = key == "s_ae" and values.min() > 0
    edges = (np.geomspace(values.min(), values.max(), 100) if log_axis
             else np.linspace(values.min(), values.max(), 100))
    edges = np.unique(np.r_[edges, thresholds[key]])
    for label, color in [(0, "#2563eb"), (1, "#c2410c")]:
        group = values[pred.y.to_numpy() == label]
        counts, _ = np.histogram(group, bins=edges)
        ax.stairs(counts / len(group), edges, color=color, label=["Benign", "Attack"][label])
    ax.axvline(thresholds[key], ls="--", color="black", label="Current threshold")
    ax.set(title=name, xlabel="Anomaly score", ylabel="Fraction per bin", yscale="log")
    if log_axis:
        ax.set_xscale("log")
    ax.legend(fontsize=8)
fig.savefig(out / "score_distributions.png", dpi=170)
plt.close(fig)

fig, (pr, roc) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
for key, name in names.items():
    y, values = pred.y, pred[key]
    precision, recall, _ = sk.precision_recall_curve(y, values)
    fpr, tpr, _ = sk.roc_curve(y, values)
    row = overall[key]
    line, = pr.plot(recall, precision, label=f"{name} ({row['pr_auc']:.3f})")
    pr.scatter(row["recall"], row["precision"], color=line.get_color())
    line, = roc.plot(fpr, tpr, label=f"{name} ({row['roc_auc']:.3f})")
    roc.scatter(row["fpr"], row["recall"], color=line.get_color())
pr.axhline(pred.y.mean(), ls=":", color="gray", label="Attack prevalence")
roc.plot([0, 1], [0, 1], ":", color="gray")
pr.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall")
roc.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
for ax in (pr, roc):
    ax.legend(fontsize=8)
fig.savefig(out / "ranking_curves.png", dpi=170)
plt.close(fig)
print("TEST_AUDIT_DONE", len(pred), flush=True)

# Reproduce calibration on every validation benign record, without Spark or fitting.
val = ds.dataset(run / "curated" / ("day=" + config["val_days"][0]), format="parquet")
chunks = []
validation_rows = 0
for batch in val.to_batches(columns=["y", "label_raw", *art["cols"]], batch_size=20000):
    pdf = batch.to_pandas()
    validation_rows += len(pdf)
    benign = pdf.label_raw.str.strip().str.upper().eq("BENIGN")
    assert np.array_equal(pdf.y.to_numpy(), (~benign).astype(int).to_numpy())
    if not benign.any():
        continue
    z = to_z(pdf[benign], art["cols"], art["mean"], art["std"])
    chunks.append(pd.DataFrame({"s_iso": -art["iso"].score_samples(z),
                                "s_ae": ((z - art["ae"].predict(z)) ** 2).mean(axis=1)}))
norms, recomputed, val_scores = calibrate(pd.concat(chunks, ignore_index=True), config["fpr_target"])
calibration = {key: dict(saved=thresholds[key], recomputed=recomputed[key],
                        delta=recomputed[key]-thresholds[key],
                        benign_above=int((val_scores[key] > thresholds[key]).sum()),
                        validation_fpr=float((val_scores[key] > thresholds[key]).mean()))
               for key in names}
audit = dict(prediction_rows=len(pred), label_mismatches=0, ensemble_max_abs_error=ensemble_error,
             validation_rows=validation_rows, validation_benign_rows=len(val_scores),
             calibration=calibration, score_norm_saved=art["score_norm"], score_norm_recomputed=norms)
(out / "checks.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print("AUDIT_DONE", audit, flush=True)
