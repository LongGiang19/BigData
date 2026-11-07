import os
from pyspark.sql import functions as F, types as T
from . import config as cfg
from .utils import norm, day_of

def read_day(spark, path):
    df = spark.read.parquet(str(path))
    new = [norm(c) for c in df.columns]
    assert len(set(new)) == len(new), f"Tên cột trùng sau chuẩn hoá: {path}"
    return df.toDF(*new)

def standardize(df, day):
    if "label" not in df.columns:
        raise ValueError("Không có cột 'label' sau khi chuẩn hoá tên.")
    df = df.filter(F.col("label").isNotNull() & (F.trim(F.col("label").cast("string")) != "Label"))
    df = (df.withColumn("day", F.lit(day))
            .withColumn("label_raw", F.trim(F.col("label").cast("string")))
            .withColumn("y", (F.upper(F.col("label_raw")) != "BENIGN").cast("int"))
            .drop("label"))
    feats = [f.name for f in df.schema.fields
             if isinstance(f.dataType, T.NumericType) and f.name not in cfg.EXCLUDE_FEATURES]
    if "protocol" in feats:
        df = (df.withColumn("is_tcp",   (F.col("protocol") == 6).cast("double"))
                .withColumn("is_udp",   (F.col("protocol") == 17).cast("double"))
                .withColumn("is_other_proto", (~F.col("protocol").isin(6, 17)).cast("double"))
                .drop("protocol"))
        feats.remove("protocol")
        feats += ["is_tcp", "is_udp", "is_other_proto"]
    return df, feats

def load_days(spark, paths):
    parts, feats0 = [], None
    for p in paths:
        df, feats = standardize(read_day(spark, p), day_of(p))
        feats0 = feats0 or feats
        assert feats == feats0, f"Schema lệch ở {os.path.basename(p)}"
        parts.append(df)
    out = parts[0]
    for d in parts[1:]:
        out = out.unionByName(d)
    return out, feats0

def prepare_curated(spark, files):
    if cfg.SKIP_INGEST and any(cfg.CURATED_DIR.glob("day=*")):
        print("Using cached curated data")
    else:
        raw, _ = load_days(spark, files)
        raw.write.mode("overwrite").partitionBy("day").parquet(str(cfg.CURATED_DIR))
    return spark.read.parquet(str(cfg.CURATED_DIR))
