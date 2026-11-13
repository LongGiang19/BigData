"""Reproduce validation-only FPR sensitivity from saved full-test artifacts."""
import argparse
import json
import os
import sys
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from src.preprocessing import to_z

TARGETS = (.001, .005, .01, .02, .05)
MODELS = {"s_iso": "Isolation Forest", "s_ae": "Autoencoder", "s_ens": "Ensemble"}


def metrics(tp, fp, pos, neg):
    fn, tn = pos - tp, neg - fp
    return dict(TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
                precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / pos if pos else np.nan,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                fpr=fp / neg if neg else np.nan)


def validation_scores(run, art, day):
    val = ds.dataset(run / "curated" / f"day={day}", format="parquet")
    chunks = []
    cols = art["cols"]
    m0, s0, m1, s1 = art["score_norm"]
    for batch in val.to_batches(columns=["y", "label_raw", *cols], batch_size=20000):
        pdf = batch.to_pandas()
        benign = pdf.label_raw.str.strip().str.upper().eq("BENIGN")
        assert pdf.y.eq((~benign).astype(int)).all()
        if not benign.any():
            continue
        z = to_z(pdf.loc[benign], cols, art["mean"], art["std"])
        iso = -art["iso"].score_samples(z)
        ae = ((z - art["ae"].predict(z)) ** 2).mean(axis=1)
        ens = .5 * (np.log(iso + 1e-9) - m0) / s0 + .5 * (np.log(ae + 1e-9) - m1) / s1
        chunks.append(pd.DataFrame(dict(s_iso=iso, s_ae=ae, s_ens=ens)))
    return pd.concat(chunks, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=root / ".cleaning_check/test100_20261109_233550_850520")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("thresholds.csv", "overall_metrics.csv", "report.md")):
        raise FileExistsError("Choose a fresh --output directory")
    source = args.run / "outputs"
    cfg = json.loads((source / "metrics/config.json").read_text(encoding="utf-8"))
    verified = json.loads((args.run / "verification_metrics.json").read_text(encoding="utf-8"))
    art = joblib.load(source / "models/model.joblib")
    assert art["thresholds"] == cfg["thresholds"] and len(cfg["val_days"]) == 1
    val = validation_scores(args.run, art, cfg["val_days"][0])
    assert len(val) == verified["validation_benign_rows"] == 446619
    np.testing.assert_allclose(val.quantile(.99).loc[list(MODELS)].to_numpy(),
                               [art["thresholds"][k] for k in MODELS], atol=2e-6, rtol=0)
    print("VALIDATION_VERIFIED", len(val), flush=True)
    pred = ds.dataset(source / "predictions/test_scored", format="parquet",
                      partitioning="hive").to_table(columns=["y", "day", "label_raw", *MODELS]).to_pandas()
    y = pred.y.to_numpy() == 1
    assert len(pred) == verified["test_rows"] == 4653501
    assert pred.y.eq(pred.label_raw.str.strip().str.upper().ne("BENIGN").astype(int)).all()
    assert pred.groupby("day").size().to_dict() == verified["test_counts_by_day"]
    days = pred.groupby("day").indices
    attacks = pred.loc[y].groupby("label_raw").groups
    pos, neg = int(y.sum()), int((~y).sum())
    thresholds, overall, daily, attack_rows, ranking = [], [], [], [], []
    for key, model in MODELS.items():
        score = pred[key].to_numpy()
        assert np.isfinite(score).all() and np.isfinite(val[key]).all()
        for day, idx in days.items():
            yd, sd = y[idx], score[idx]
            p, r, _ = precision_recall_curve(yd, sd)
            ranking.append(dict(model=model, day=day, n=len(idx), prevalence=yd.mean(),
                                roc_auc=roc_auc_score(yd, sd), pr_auc=auc(r, p)))
        for target in TARGETS:
            threshold = float(val[key].quantile(1 - target))
            validation_fpr = float(val[key].gt(threshold).mean())
            meta = dict(model=model, fpr_target=target, threshold=threshold, validation_fpr=validation_fpr)
            thresholds.append(dict(**meta, validation_n=len(val), validation_fp=int(val[key].gt(threshold).sum())))
            hit = score > threshold
            fp = int((hit & ~y).sum())
            overall.append(dict(**meta, **metrics((hit & y).sum(), fp, pos, neg)))
            for day, idx in days.items():
                yd, hd = y[idx], hit[idx]
                daily.append(dict(day=day, **meta, **metrics((hd & yd).sum(), (hd & ~yd).sum(),
                                                           yd.sum(), (~yd).sum())))
            # Each attack is compared against the same full-test benign pool.
            for label, idx in attacks.items():
                attack_rows.append(dict(attack=label, attack_n=len(idx), benign_reference_n=neg,
                                        **meta, **metrics(hit[idx].sum(), fp, len(idx), neg)))
    tables = {
        "thresholds": pd.DataFrame(thresholds), "overall_metrics": pd.DataFrame(overall),
        "metrics_by_day": pd.DataFrame(daily), "metrics_by_attack": pd.DataFrame(attack_rows),
        "ranking_by_day": pd.DataFrame(ranking),
    }
    keys = ["model", "fpr_target"]
    totals = tables["overall_metrics"].set_index(keys).sort_index()
    for name, columns in (("metrics_by_day", ["TP", "FP", "TN", "FN"]),
                          ("metrics_by_attack", ["TP", "FN"])):
        pd.testing.assert_frame_equal(tables[name].groupby(keys)[columns].sum().sort_index(),
                                      totals[columns])
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    results = tables["overall_metrics"]
    for x, z, title in [
        ("fpr_target", "recall", "Recall vs validation FPR target"),
        ("fpr_target", "fpr", "Actual test FPR vs validation FPR target"),
        ("recall", "precision", "Precision–Recall at validation thresholds"),
        ("threshold", "recall", "Threshold vs recall"),
    ]:
        separate = x == "threshold"
        fig, axes = plt.subplots(1, 3 if separate else 1, figsize=(13 if separate else 7, 4.5),
                                 layout="constrained", squeeze=False)
        for i, (model, group) in enumerate(results.groupby("model", sort=False)):
            ax = axes[0, i if separate else 0]
            a = group[x].to_numpy() * (1 if separate else 100)
            b = group[z].to_numpy() * 100
            ax.plot(a, b, marker="o", label=model)
            if x == "recall":
                for xx, yy, target in zip(a, b, group.fpr_target):
                    ax.annotate(f"{target:.1%}", (xx, yy), xytext=(4, 4), textcoords="offset points", fontsize=7)
            if x == "fpr_target":
                ax.set_xscale("log")
                ax.set_xticks([t*100 for t in TARGETS], [str(t*100) for t in TARGETS])
            ax.set(xlabel={"fpr_target": "Validation FPR target (%)", "recall": "Recall (%)",
                           "threshold": "Score threshold"}[x],
                   ylabel={"recall": "Recall (%)", "fpr": "Actual test FPR (%)",
                           "precision": "Precision (%)"}[z])
            ax.grid(alpha=.25)
            ax.legend()
        if z == "fpr":
            reference = np.geomspace(.1, 5, 100)
            axes[0, 0].plot(reference, reference, ":", color="gray", label="Test = target")
            axes[0, 0].legend()
        fig.suptitle(title)
        fig.savefig(out / f"{z}_vs_{x}.png", dpi=170)
        plt.close(fig)
    assert len(daily) == 105
    print(results.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
