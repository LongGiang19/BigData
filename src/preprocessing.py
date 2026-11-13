import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from . import config as cfg

def _finite(c):
    x = F.col(c).cast("double")
    return ~(x.isNull() | F.isnan(x) | (F.abs(x) == float("inf")))


def invalid_value(c):
    """Non-null boolean predicate shared by cleaning and quality reporting."""
    x = F.col(c).cast("double")
    bad = ~_finite(c)
    if c not in cfg.ALLOW_NEG:
        bad = bad | (x < 0)
    return F.coalesce(bad, F.lit(True))


def sanitize_features(df, feats):
    """Replace invalid feature cells with zero without dropping any rows."""
    feats = set(feats)
    return df.select(*[
        F.when(invalid_value(c), F.lit(0.0))
        .otherwise(F.col(c).cast("double")).alias(c) if c in feats else F.col(c)
        for c in df.columns
    ])


def quality_by_day(df, feats):
    """Aggregate BEFORE sanitizing; return a Spark DataFrame by day and y."""
    cells = F.lit(0).cast("long")
    for c in feats:
        cells = cells + invalid_value(c).cast("long")
    return (df.select("day", "y", cells.alias("__invalid_cells"))
            .groupBy("day", "y")
            .agg(F.count("*").alias("total_rows"),
                 F.sum((F.col("__invalid_cells") > 0).cast("long")).alias("invalid_rows"),
                 F.sum("__invalid_cells").alias("invalid_cells"))
            .withColumn("invalid_row_rate", F.col("invalid_rows") / F.col("total_rows"))
            .orderBy("day", "y"))

def profile(df, feats):
    exprs = [F.count("*").alias("__n")]
    for c in feats:
        x = F.col(c).cast("double")
        exprs += [F.sum((~_finite(c)).cast("long")).alias(f"bad|{c}"),
                  F.sum((x < 0).cast("long")).alias(f"neg|{c}"),
                  F.avg((x == 0).cast("double")).alias(f"zero|{c}"),
                  F.min(F.when(_finite(c), x)).alias(f"min|{c}"),
                  F.max(F.when(_finite(c), x)).alias(f"max|{c}")]
    r = df.agg(*exprs).first()
    return pd.DataFrame([{"feature": c, "nonfinite": r[f"bad|{c}"] or 0, "neg": r[f"neg|{c}"] or 0,
                          "zero_frac": r[f"zero|{c}"], "min": r[f"min|{c}"], "max": r[f"max|{c}"],
                          "constant": r[f"min|{c}"] == r[f"max|{c}"]} for c in feats])

def valid_cond(feats):
    """Hợp lệ = hữu hạn và không âm (trừ cfg.ALLOW_NEG)."""
    cond = F.lit(True)
    for c in feats:
        cond = cond & ~invalid_value(c)
    return cond

def slog_spark(c):
    x = F.col(c).cast("double")
    return F.signum(x) * F.log1p(F.abs(x))

def slog_np(X):
    return np.sign(X) * np.log1p(np.abs(X))

def fit_feature_spec(ben_df, feats, std_floor=cfg.STD_FLOOR, min_nonzero=cfg.MIN_NONZERO, corr_thr=cfg.CORR_THR,
                     sample_n=100_000, seed=cfg.SEED, *, signed_log=True, correlation_sample=None):
    exprs = [F.count("*").alias("__n")]
    for c in feats:
        x = slog_spark(c) if signed_log else F.col(c).cast("double")
        exprs += [F.mean(x).alias(f"mean|{c}"), F.stddev(x).alias(f"std|{c}"),
                  F.avg((F.col(c) != 0).cast("double")).alias(f"nz|{c}")]
    r = ben_df.agg(*exprs).first()
    n_benign = r["__n"]
    stat = {c: (float(r[f"mean|{c}"] or 0.0), float(r[f"std|{c}"] or 0.0), float(r[f"nz|{c}"] or 0.0)) for c in feats}

    status = {}
    for c in feats:
        _, std, nz = stat[c]
        if std < 1e-9:
            status[c] = ("drop", "feature hằng")
        elif nz < min_nonzero:
            status[c] = ("drop", f"quá hiếm, nonzero={nz:.4%}")
    candidates = [c for c in feats if c not in status]

    frac = min(1.0, 1.05 * sample_n / max(n_benign, 1))
    pdf = (ben_df.select(*candidates).sample(False, frac, seed).limit(sample_n).toPandas()
           if correlation_sample is None else correlation_sample[candidates])
    values = pdf.to_numpy(np.float64)
    corr = pd.DataFrame(slog_np(values) if signed_log else values, columns=candidates).corr().abs().fillna(0.0)

    kept = []
    for c in candidates:
        hit = next((k for k in kept if corr.loc[c, k] > corr_thr), None)
        if hit is None:
            kept.append(c)
        else:
            status[c] = ("drop", f"|rho|={corr.loc[c, hit]:.3f} với {hit}")

    mean = np.array([stat[c][0] for c in kept])
    std = np.maximum(np.array([stat[c][1] for c in kept]), std_floor)
    report = pd.DataFrame([{"feature": c, "decision": status.get(c, ("keep", ""))[0], "reason": status.get(c, ("keep", ""))[1],
                            "std_slog": stat[c][1], "nonzero_frac": stat[c][2],
                            "std_floored": c in kept and stat[c][1] < std_floor} for c in feats])
    return kept, mean, std, report

def to_z(pdf, cols, mean, std, clip=cfg.Z_CLIP, *, signed_log=True):
    """Biến đổi signed-log + z-score. Tự vô hiệu hoá giá trị lỗi (NaN/inf -> 0, âm -> 0 trừ cfg.ALLOW_NEG)
    để val/test không cần xoá dòng."""
    X = np.array(pdf[cols].to_numpy(dtype=np.float64, na_value=np.nan), copy=True)
    X[~np.isfinite(X)] = 0.0
    neg_idx = [j for j, c in enumerate(cols) if c not in cfg.ALLOW_NEG]
    X[:, neg_idx] = np.maximum(X[:, neg_idx], 0.0)
    Z = np.clip(((slog_np(X) if signed_log else X) - mean) / std, -clip, clip)
    return Z.astype(np.float32)
