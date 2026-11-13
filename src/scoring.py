import numpy as np
import pandas as pd
from pyspark.sql import functions as F, types as T
from . import config as cfg
from .preprocessing import to_z

SCORE_SCHEMA = T.StructType([
    T.StructField("day", T.StringType()), T.StructField("label_raw", T.StringType()),
    T.StructField("y", T.IntegerType()), T.StructField("s_iso", T.DoubleType()),
    T.StructField("s_ae", T.DoubleType()), T.StructField("top_feat", T.StringType())
])

def make_scorer(bc, cols, *, clip=cfg.Z_CLIP, signed_log=True):
    def fn(it):
        iso, ae, mean, std, err_ref = bc.value
        names = np.array(cols)
        for pdf in it:
            Z = to_z(pdf, cols, mean, std, clip=clip, signed_log=signed_log)
            err = (Z - ae.predict(Z)) ** 2
            out = pdf[["day", "label_raw", "y"]].copy()
            out["s_iso"] = -iso.score_samples(Z)
            out["s_ae"] = err.mean(axis=1)
            out["top_feat"] = names[(err / err_ref).argmax(axis=1)]
            yield out
    return fn

def score_pandas(pdf, art):
    Z = to_z(pdf, art["cols"], art["mean"], art["std"])
    s_iso = -art["iso"].score_samples(Z)
    err = (Z - art["ae"].predict(Z)) ** 2
    m0, s0, m1, s1 = art["score_norm"]
    ens = 0.5 * (np.log(s_iso + 1e-9) - m0) / s0 + 0.5 * (np.log(err.mean(1) + 1e-9) - m1) / s1
    out = pd.DataFrame({"score": ens, "alert": ens > art["thresholds"]["s_ens"]})
    out["top_feat"] = np.array(art["cols"])[(err / art["err_ref"]).argmax(1)]
    return out

def select_scoring_columns(df, cols):
    return df.select(*cols, "day", "label_raw", "y")

def fuse_scores(df, score_norm):
    mu_i, sd_i, mu_a, sd_a = score_norm
    return df.withColumn("s_ens", 0.5 * (F.log(F.col("s_iso") + 1e-9) - mu_i) / sd_i
                         + 0.5 * (F.log(F.col("s_ae") + 1e-9) - mu_a) / sd_a)

def calibrate(val_sc, fpr_target):
    lg = lambda a: np.log(np.asarray(a) + 1e-9)
    mu_i, sd_i = lg(val_sc.s_iso).mean(), lg(val_sc.s_iso).std()
    mu_a, sd_a = lg(val_sc.s_ae).mean(), lg(val_sc.s_ae).std()
    if min(sd_i, sd_a) <= 0:
        raise ValueError("Validation scores have zero variance")
    val_sc = val_sc.copy()
    val_sc["s_ens"] = 0.5 * (lg(val_sc.s_iso) - mu_i) / sd_i + 0.5 * (lg(val_sc.s_ae) - mu_a) / sd_a
    norms = [float(mu_i), float(sd_i), float(mu_a), float(sd_a)]
    thresholds = {k: float(np.quantile(val_sc[k], 1 - fpr_target)) for k in ["s_iso", "s_ae", "s_ens"]}
    return norms, thresholds, val_sc
