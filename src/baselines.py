"""Optional supervised comparisons retained from the original notebook.
Random-split comparison is exploratory, not a causal upper bound.
"""
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.functions import vector_to_array
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from . import config as cfg
from .preprocessing import fit_feature_spec, valid_cond, to_z
from .scoring import select_scoring_columns as sel
from .evaluation import evaluate, metrics_df

def run_baselines(all_df, train_days, val_days, test, feats, cols, metrics):
    sup_all = all_df.filter(F.col("day").isin(train_days + val_days)).filter(valid_cond(feats)).select(*cols, "y")
    n_sup = sup_all.count()
    sup = sup_all.sample(False, min(1.0, 500_000 / max(n_sup, 1)), cfg.SEED).cache()
    n_pos = sup.filter("y = 1").count(); n_neg = sup.count() - n_pos
    sup = sup.withColumn("w", F.when(F.col("y") == 1, n_neg / max(n_pos, 1)).otherwise(1.0))
    asm = VectorAssembler(inputCols=cols, outputCol="features", handleInvalid="keep")
    gbt = GBTClassifier(labelCol="y", featuresCol="features", weightCol="w", maxIter=60, maxDepth=6, seed=cfg.SEED).fit(asm.transform(sup))
    pg = gbt.transform(asm.transform(sel(test, cols))).withColumn("s_gbt", vector_to_array("probability")[1])
    res = evaluate(pg.select("day", "y", "s_gbt"), "s_gbt", 0.5)
    metrics["GBT cross-day (supervised, tham chiếu)"] = res
    g = metrics_df(res, "GBT cross-day")
    print(g.round(3))

    n_all = all_df.count()
    pdf = all_df.sample(False, min(1.0, 1_000_000 / n_all), cfg.SEED).select(*feats, "label_raw", "y").toPandas()
    metrics["HistGradientBoosting random-split 70/15/15"] = run_random_baseline(
        all_df.sparkSession, pdf, feats)
    return metrics


def run_random_baseline(spark, pdf, feats):
    """Fit preprocessing and classifier on this baseline's training rows only."""
    y = pdf["y"].to_numpy()
    vc = pdf.label_raw.value_counts()
    strat = pdf.label_raw.where(pdf.label_raw.map(vc) >= 10, "RARE")          # lớp quá hiếm gộp lại để stratify được
    idx = np.arange(len(pdf))
    i_tr, i_tmp = train_test_split(idx, test_size=0.30, stratify=strat, random_state=cfg.SEED)
    i_va, i_te = train_test_split(i_tmp, test_size=0.50, stratify=strat.iloc[i_tmp], random_state=cfg.SEED)

    train = spark.createDataFrame(pdf.iloc[i_tr][feats])
    cols, mean, std, _ = fit_feature_spec(train, feats)
    if not cols:
        raise ValueError("Random baseline feature selection returned no features")
    Zs = to_z(pdf, cols, mean, std)

    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=True,
                                         class_weight="balanced", random_state=cfg.SEED).fit(Zs[i_tr], y[i_tr])
    s_va, s_te = clf.predict_proba(Zs[i_va])[:, 1], clf.predict_proba(Zs[i_te])[:, 1]
    thr_sup = float(np.quantile(s_va[y[i_va] == 0], 1 - cfg.FPR_TARGET))

    def eval_bin(yt, s, t):
        pred = s > t
        tp, fp = int((pred & (yt == 1)).sum()), int((pred & (yt == 0)).sum())
        pos, neg = int((yt == 1).sum()), int((yt == 0).sum())
        rec, prec = tp / max(pos, 1), tp / max(tp + fp, 1)
        return {"recall": rec, "fpr": fp / max(neg, 1), "precision": prec, "f1": 2 * prec * rec / max(prec + rec, 1e-12),
                "roc_auc": roc_auc_score(yt, s), "pr_auc": average_precision_score(yt, s)}

    r_rand = eval_bin(y[i_te], s_te, thr_sup)
    print("Random stratified split @FPR mục tiêu:", {k: round(v, 4) for k, v in r_rand.items()})
    per = pd.DataFrame({"label": pdf.label_raw.iloc[i_te].values, "hit": s_te > thr_sup})
    per = per[per.label.str.upper() != "BENIGN"].groupby("label").hit.agg(["size", "mean"]).rename(columns={"mean": "recall"})
    print(per.round(3))


    return dict(r_rand, threshold=thr_sup, features=cols,
                split_rows={"train": len(i_tr), "validation": len(i_va), "test": len(i_te)})
