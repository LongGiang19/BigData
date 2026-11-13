"""Descriptive temporal drift audit of the official full-test model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from . import config as cfg
from .preprocessing import to_z

OFFICIAL = cfg.ROOT / ".cleaning_check/test100_20261109_233550_850520"
MODELS = {"s_iso": "IF", "s_ae": "AE", "s_ens": "Ensemble"}
FOCUS = ("HOIC", "LOIC-HTTP", "Bot", "Infilteration")


def shift(ref, target):
    """Return two-sample KS, Wasserstein-1, and smoothed quantile-bin PSI."""
    cuts = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
    edges = np.r_[-np.inf, cuts, np.nextafter(cuts[-1], np.inf), np.inf]
    a, b = [np.histogram(values, bins=edges)[0] + .5 for values in (ref, target)]
    a, b = a/a.sum(), b/b.sum()
    return dict(ks=float(ks_2samp(ref, target, method="asymp").statistic),
                wasserstein=float(wasserstein_distance(ref, target)),
                psi=float(np.sum((b-a)*np.log(b/a))))


def sample_benign(path, cols, day, n, seed):
    files = sorted(map(str, path.glob("*.parquet")))
    table = ds.dataset(files, format="parquet").to_table(columns=cols + ["y", "label_raw"],
                                                         filter=ds.field("y") == 0)
    count = len(table)
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(day.replace("-", ""))]))
    indices = np.sort(rng.choice(count, size=min(n, count), replace=False))
    pdf = table.take(indices).to_pandas()
    assert pdf.y.eq(0).all() and pdf.label_raw.str.strip().str.upper().eq("BENIGN").all()
    return pdf, dict(day=day, population=count, sample_n=len(indices),
                     ordered_files=[str(Path(p).name) for p in files],
                     benign_ordinal_sha256=hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest())


def score_features(x, art):
    err = (x - art["ae"].predict(x)) ** 2
    iso, ae = -art["iso"].score_samples(x), err.mean(axis=1)
    m0, s0, m1, s1 = art["score_norm"]
    ens = .5*(np.log(iso+1e-9)-m0)/s0 + .5*(np.log(ae+1e-9)-m1)/s1
    return err.mean(axis=0), dict(s_iso=iso, s_ae=ae, s_ens=ens)


def summarize_score(score, threshold):
    p = np.quantile(score, [.1, .5, .9, .99])
    return dict(n=len(score), p10=p[0], median=p[1], p90=p[2], p99=p[3],
                above_threshold=int((score > threshold).sum()), alert_rate=float((score > threshold).mean()))


def performance(y, score, threshold):
    hit = score > threshold
    tp, fp = int((hit & y).sum()), int((hit & ~y).sum())
    pos, neg = int(y.sum()), int((~y).sum())
    p, r, _ = precision_recall_curve(y, score)
    return dict(n=len(y), attacks=pos, prevalence=y.mean(), TP=tp, FP=fp, TN=neg-fp, FN=pos-tp,
                recall=tp/pos, fpr=fp/neg, precision=tp/(tp+fp) if tp+fp else 0,
                roc_auc=roc_auc_score(y, score), pr_auc=auc(r, p))


def run(source, out, n=50000, seed=42):
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    official = source / "outputs"
    conf = json.loads((official / "metrics/config.json").read_text(encoding="utf-8"))
    verified = json.loads((source / "verification_metrics.json").read_text(encoding="utf-8"))
    art = joblib.load(official / "models/model.joblib")
    assert art["thresholds"] == conf["thresholds"] and len(conf["val_days"]) == 1
    cols = art["cols"]
    split = [conf[k + "_days"] for k in ("train", "val", "test")]
    train_days, val_days, test_days = split
    val_day = val_days[0]
    assert max(train_days) < val_day < min(test_days)
    raw_quality = pd.read_csv(official / "metrics/data_quality_by_day.csv")
    benign_counts = raw_quality[raw_quality.y.eq(0)].set_index("day").total_rows
    inputs, errors, baseline_scores, samples = {}, {}, {}, []
    for day in train_days + val_days + test_days:
        pdf, info = sample_benign(source / "curated" / f"day={day}", cols, day, n, seed)
        assert info["population"] == benign_counts.loc[day]
        samples.append(info)
        inputs[day] = to_z(pdf, cols, art["mean"], art["std"], clip=np.inf)
        errors[day], baseline_scores[day] = score_features(np.clip(inputs[day], -cfg.Z_CLIP, cfg.Z_CLIP), art)
    # Each train day contributes proportionally to its Benign population.
    first_quota = round(n * benign_counts.loc[train_days[0]] / benign_counts.loc[train_days].sum())
    quotas = [first_quota, n-first_quota]
    rng = np.random.default_rng(seed)
    train = np.concatenate([inputs[day][rng.choice(len(inputs[day]), size=q, replace=False)]
                            for day, q in zip(train_days, quotas)])
    train_error, train_scores = score_features(np.clip(train, -cfg.Z_CLIP, cfg.Z_CLIP), art)
    feature_rows, relation_rows = [], []
    for day in val_days + test_days:
        for reference, ref, ref_error in (("train", train, train_error),
                                         (val_day, inputs[val_day], errors[val_day])):
            if day == reference:
                continue
            rows = []
            for i, name in enumerate(cols):
                m = shift(ref[:, i], inputs[day][:, i])
                rows.append(dict(day=day, reference=reference, feature=name,
                                 reference_n=len(ref), n=len(inputs[day]), **m,
                                 recon_mse=errors[day][i], reference_recon_mse=ref_error[i],
                                 delta_recon_mse=errors[day][i]-ref_error[i],
                                 clip_fraction=float((np.abs(inputs[day][:, i]) > cfg.Z_CLIP).mean())))
            feature_rows.extend(rows)
            relation_rows.append(dict(day=day, reference=reference,
                ks_vs_recon_mse=spearmanr([r["ks"] for r in rows], [r["recon_mse"] for r in rows]).statistic,
                ks_vs_delta_mse=spearmanr([r["ks"] for r in rows], [r["delta_recon_mse"] for r in rows]).statistic))

    pred = ds.dataset(official / "predictions/test_scored", format="parquet", partitioning="hive").to_table(
        columns=["day", "y", "label_raw", "is_anomaly", *MODELS]).to_pandas()
    assert len(pred) == verified["test_rows"] == 4653501
    assert pred.groupby("day").size().to_dict() == verified["test_counts_by_day"]
    assert pred.y.eq(pred.label_raw.str.strip().str.upper().ne("BENIGN").astype(int)).all()
    assert pred.is_anomaly.eq(pred.s_ens.gt(art["thresholds"]["s_ens"])).all()
    assert np.isfinite(pred[list(MODELS)]).all().all()
    score_rows, perf_rows, attack_rows = [], [], []
    references = {"train": train_scores, val_day: baseline_scores[val_day]}
    for key, model in MODELS.items():
        threshold = art["thresholds"][key]
        for day, ref in references.items():
            score_rows.append(dict(day=day, model=model, label="Benign", reference="",
                                   threshold=threshold, **summarize_score(ref[key], threshold)))
        for day, frame in pred.groupby("day", sort=True):
            y = frame.y.to_numpy() == 1
            values = frame[key].to_numpy()
            perf_rows.append(dict(day=day, model=model, threshold=threshold, **performance(y, values, threshold)))
            for label, score in (("Benign", values[~y]), ("Attack", values[y])):
                for ref_day, ref in references.items():
                    score_rows.append(dict(day=day, model=model, label=label, reference=ref_day,
                                           threshold=threshold, **summarize_score(score, threshold),
                                           **shift(ref[key], score)))
        attack = pred.loc[pred.y.eq(1), ["day", "label_raw", key]]
        groups = list(attack.groupby(["day", "label_raw"], sort=True))
        groups += [(("ALL", label), frame) for label, frame in attack.groupby("label_raw", sort=True)]
        for (day, label), frame in groups:
            row = summarize_score(frame[key].to_numpy(), threshold)
            attack_rows.append(dict(day=day, attack=label, model=model, threshold=threshold,
                                    focus=any(tag.casefold() in label.casefold() for tag in FOCUS),
                                    FN=row["n"]-row["above_threshold"], **row))
    features, scores, perf, attacks, relation = map(pd.DataFrame,
        (feature_rows, score_rows, perf_rows, attack_rows, relation_rows))
    official_by_day = pd.read_csv(source / "metrics_by_day_independent.csv")
    official_by_day["model"] = official_by_day.model.map(
        {"Isolation Forest": "IF", "Autoencoder": "AE", "Ensemble AE+IF": "Ensemble"})
    expected = perf.merge(official_by_day, on=["day", "model"], suffixes=("", "_official"))
    assert len(expected) == 21
    for metric in ("TP", "FP", "TN", "FN", "recall", "fpr", "precision", "roc_auc", "pr_auc"):
        np.testing.assert_allclose(expected[metric], expected[metric+"_official"], atol=1e-12)
    assert len(features) == len(cols) * 15 and len(scores) == 90
    for name, df in (("feature_drift_by_day", features), ("score_drift_by_day", scores),
                     ("performance_by_day", perf), ("attack_analysis", attacks),
                     ("drift_error_association", relation)):
        df.to_csv(out / f"{name}.csv", index=False)
    (out / "analysis_config.json").write_text(json.dumps(dict(
        official_run=str(source.resolve()), sample_size=n, seed=seed, train_quotas=dict(zip(train_days, quotas)),
        thresholds=art["thresholds"], features=cols, samples=samples, test_rows=len(pred),
        runtime_seconds=time.perf_counter()-start), indent=2), encoding="utf-8")
    plot(out, train, inputs, cols, features, scores, perf, test_days, val_day)
    print("DRIFT_VERIFIED", len(pred), len(features), len(scores), len(perf), len(attacks), flush=True)


def plot(out, train, inputs, cols, features, scores, perf, test_days, val_day):
    subset = features[features.reference.eq("train")]
    order = subset[subset.day.isin(test_days)].groupby("feature").ks.median().nlargest(12).index
    heat = subset.pivot(index="feature", columns="day", values="ks").loc[order, [val_day, *test_days]]
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    im = ax.imshow(heat, aspect="auto", vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(range(heat.shape[1]), [d[5:] for d in heat.columns], rotation=30)
    ax.set_yticks(range(len(order)), order)
    ax.set(title="Benign feature KS vs pooled train (top 12)", xlabel="Validation and test day")
    fig.colorbar(im, ax=ax, label="KS statistic")
    fig.savefig(out / "feature_drift_heatmap.png", dpi=170)
    plt.close(fig)

    worst = subset[subset.day.isin(test_days)].groupby("day").ks.median().idxmax()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    for ax, feature in zip(axes, order[:3]):
        idx = cols.index(feature)
        for label, matrix in (("Pooled train", train), (val_day, inputs[val_day]), (worst, inputs[worst])):
            x = np.sort(matrix[:, idx])
            ax.plot(x, np.arange(1, len(x)+1)/len(x), label=label)
        ax.set(title=feature, xlabel="Signed-log z-score before clipping", ylabel="Empirical CDF")
        ax.legend(fontsize=8)
    fig.savefig(out / "feature_distributions.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, layout="constrained")
    for ax, model in zip(axes, MODELS.values()):
        data = scores[scores.model.eq(model) & scores.reference.eq(val_day)]
        for label, color in (("Benign", "#2563eb"), ("Attack", "#c2410c")):
            group = data[data.label.eq(label)].set_index("day").loc[test_days]
            x = np.arange(len(group))
            ax.plot(x, group["median"], "o-", color=color, label=label)
            ax.fill_between(x, group.p10, group.p90, color=color, alpha=.17)
        ax.axhline(data.threshold.iloc[0], color="black", ls="--", label="Saved threshold")
        if model == "AE":
            ax.set_yscale("symlog", linthresh=.001)
        ax.set(ylabel=model + " score")
        ax.legend(loc="best", ncol=3)
    axes[-1].set_xticks(range(len(test_days)), [d[5:] for d in test_days])
    fig.suptitle("Full test score distributions by day · median and p10–p90")
    fig.savefig(out / "score_distributions.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, layout="constrained")
    for ax, metric in zip(axes, ("recall", "fpr", "pr_auc")):
        for model, group in perf.groupby("model", sort=False):
            group = group.set_index("day").loc[test_days]
            ax.plot(range(len(group)), group[metric]*100, "o-", label=model)
        if metric == "pr_auc":
            ax.plot(range(len(group)), group.prevalence*100, ":", color="gray", label="Attack prevalence")
        ax.set_ylabel(metric.upper() + " (%)")
        ax.grid(alpha=.25)
        ax.legend(ncol=4)
    axes[-1].set_xticks(range(len(test_days)), [d[5:] for d in test_days])
    fig.savefig(out / "performance_by_day.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=OFFICIAL)
    parser.add_argument("--output", type=Path, default=cfg.ROOT / "outputs/temporal_drift")
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.run, args.output, args.sample_size, args.seed)
