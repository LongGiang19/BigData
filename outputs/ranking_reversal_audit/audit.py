"""Independent ranking audit of the saved predictions for 2018-02-21 only."""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / ".cleaning_check/test100_20261109_233550_850520"
DAY = "2018-02-21"
NAMES = {"s_iso": "Isolation Forest", "s_ae": "Autoencoder", "s_ens": "Ensemble"}
QUANTILES = [0, .01, .10, .50, .90, .99, 1]


def pr_area(y, score):
    precision, recall, _ = precision_recall_curve(y, score)
    return auc(recall, precision)


def run(source, out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.csv").exists():
        raise FileExistsError("Choose a fresh output directory")
    partition = source / "outputs/predictions/test_scored" / f"day={DAY}"
    pred = ds.dataset(partition, format="parquet").to_table(
        columns=["label_raw", "y", "is_anomaly", *NAMES]).to_pandas()
    assert pred.label_raw.notna().all()
    y = pred.label_raw.str.strip().str.upper().ne("BENIGN").to_numpy()
    np.testing.assert_array_equal(pred.y.to_numpy(), y.astype(int))
    assert np.isfinite(pred[list(NAMES)]).all().all()
    assert (pred[["s_iso", "s_ae"]] >= 0).all().all()
    config = json.loads((source / "outputs/metrics/config.json").read_text(encoding="utf-8"))
    assert pred.is_anomaly.eq(pred.s_ens.gt(config["thresholds"]["s_ens"])).all()
    art = joblib.load(source / "outputs/models/model.joblib")
    assert art["thresholds"] == config["thresholds"]
    m0, s0, m1, s1 = art["score_norm"]
    assert min(s0, s1) > 0
    ensemble = .5*(np.log(pred.s_iso+1e-9)-m0)/s0 + .5*(np.log(pred.s_ae+1e-9)-m1)/s1
    ensemble_error = float(np.max(np.abs(ensemble-pred.s_ens)))
    assert ensemble_error < 1e-12
    counts = pred.groupby(["label_raw", "y"]).size().rename("n").reset_index()
    counts.to_csv(out / "label_counts.csv", index=False)
    neg, pos = int((~y).sum()), int(y.sum())
    overall, quantiles, contributions = [], [], []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout="constrained")
    groups = pred.loc[y].groupby("label_raw").groups
    for (key, model), ax in zip(NAMES.items(), axes):
        s = pred[key].to_numpy()
        neg_scores = s[~y]
        roc = roc_auc_score(y, s)
        reverse = roc_auc_score(y, -s)
        np.testing.assert_allclose(roc + reverse, 1, atol=1e-12, rtol=0)
        overall.append(dict(day=DAY, model=model, benign=neg, attack=pos,
                            prevalence=pos / len(pred), roc_auc=roc,
                            roc_auc_negated=reverse, pr_auc=pr_area(y, s),
                            pr_auc_negated=pr_area(y, -s)))
        for label, values, color in [("Benign", neg_scores, "#2563eb"), ("Attack", s[y], "#c2410c")]:
            q = np.quantile(values, QUANTILES)
            quantiles.append(dict(model=model, label=label, n=len(values),
                                  **{f"p{int(p*100):02d}": v for p, v in zip(QUANTILES, q)}))
            x, count = np.unique(values, return_counts=True)
            ax.step(x, np.cumsum(count)/len(values), where="post", color=color, label=label)
        ax.axvline(config["thresholds"][key], color="black", ls="--", label="Saved threshold")
        if key == "s_ae":
            ax.set_xscale("symlog", linthresh=.001)
        ax.set(title=model, xlabel="Anomaly score", ylabel="Empirical CDF")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=.2)
        model_rows = []
        for attack, idx in groups.items():
            attack_scores = s[idx]
            support = len(idx)
            # All attack types use exactly the same Benign comparison pool.
            pair_scores = np.r_[neg_scores, attack_scores]
            pair_y = np.r_[np.zeros(neg, dtype=bool), np.ones(support, dtype=bool)]
            pair_auc = roc_auc_score(pair_y, pair_scores)
            model_rows.append(dict(model=model, attack=attack, n=support, share_of_attacks=support/pos,
                                   auc_vs_same_benign=pair_auc,
                                   wrong_order_mass=(support/pos)*(1-pair_auc),
                                   median= float(np.median(attack_scores)),
                                   p99=float(np.quantile(attack_scores,.99)),
                                   above_threshold=int((attack_scores > config["thresholds"][key]).sum())))
        np.testing.assert_allclose(sum(r["share_of_attacks"]*r["auc_vs_same_benign"] for r in model_rows),
                                   roc, atol=1e-12, rtol=0)
        for row in model_rows:
            row["share_of_total_wrong_order_mass"] = row["wrong_order_mass"]/(1-roc)
        contributions.extend(model_rows)
    pd.DataFrame(overall).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(quantiles).to_csv(out / "score_quantiles.csv", index=False)
    pd.DataFrame(contributions).to_csv(out / "attack_contributions.csv", index=False)
    fig.suptitle(f"{DAY} · anomaly score distributions (empirical CDF)")
    fig.savefig(out / "score_distributions.png", dpi=170)
    plt.close(fig)
    (out / "audit_config.json").write_text(json.dumps(
        dict(day=DAY, source_partition=str(partition.resolve()), benign=neg, attack=pos,
             thresholds=config["thresholds"], label_mismatches=0,
             ensemble_max_abs_error=ensemble_error), indent=2), encoding="utf-8")
    print(pd.DataFrame(overall).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    run(args.run, args.output)
