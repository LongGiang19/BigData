import pandas as pd
from pyspark.sql import functions as F
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def evaluate(sdf, score_col, thr):
    ev = lambda d, m: BinaryClassificationEvaluator(rawPredictionCol=score_col, labelCol="y", metricName=m).evaluate(d)
    rows = sdf.groupBy("day").agg(
        F.count("*").alias("n"), F.sum("y").alias("pos"),
        F.sum(((F.col("y") == 1) & (F.col(score_col) > thr)).cast("int")).alias("tp"),
        F.sum(((F.col("y") == 0) & (F.col(score_col) > thr)).cast("int")).alias("fp")).collect()
    out = {}
    for r in sorted(rows, key=lambda r: r["day"]):
        n, pos, tp, fp = r["n"], r["pos"], r["tp"], r["fp"]
        neg = n - pos
        rec = tp / pos if pos else None
        prec = tp / (tp + fp) if tp + fp else None
        d = {"n": n, "attack_rate": pos / n, "recall": rec, "fpr": fp / neg if neg else None, "precision": prec,
             "f1": (2 * prec * rec / (prec + rec)) if prec and rec else None}
        if pos and neg:
            sub = sdf.filter(F.col("day") == r["day"])
            d["roc_auc"], d["pr_auc"] = ev(sub, "areaUnderROC"), ev(sub, "areaUnderPR")
        out[r["day"]] = d
    return out

def metrics_df(res, model):
    return pd.DataFrame([{"model": model, "day": d, **v} for d, v in res.items()])
