"""Entrypoint: python run_pipeline.py (run from project root)."""
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from pyspark.sql import functions as F

from src import config as cfg
from src.utils import day_of, log
from src.spark_session import create_spark
from src.data_io import prepare_curated
from src.preprocessing import profile, valid_cond, fit_feature_spec, to_z
from src.models import fit_models
from src.scoring import (SCORE_SCHEMA, make_scorer, select_scoring_columns,
                         fuse_scores, calibrate, score_pandas)
from src.evaluation import evaluate, metrics_df
from src.reports import plot_ae_loss, plot_metrics


def main():
    for d in [cfg.OUT_DIR / "models", cfg.OUT_DIR / "metrics", cfg.OUT_DIR / "predictions", cfg.OUT_DIR / "figures", cfg.CURATED_DIR.parent]:
        d.mkdir(parents=True, exist_ok=True)
    files = cfg.TRAIN_FILES + [cfg.VAL_FILE] + cfg.TEST_FILES
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset files in DATA_DIR:\n" + "\n".join(missing))
    train_days = sorted(day_of(p) for p in cfg.TRAIN_FILES)
    val_days = [day_of(cfg.VAL_FILE)]
    test_days = sorted(day_of(p) for p in cfg.TEST_FILES)
    assert max(train_days) < min(val_days) < min(test_days), "Incorrect temporal split"
    print("Train:", train_days, "Validation:", val_days, "Test:", test_days)
    spark = create_spark()
    try:
        log("Ingest and prepare parquet")
        all_df = prepare_curated(spark, files)
        feats = [c for c in all_df.columns if c not in cfg.EXCLUDE_FEATURES]
        train_all = all_df.filter(F.col("day").isin(train_days))
        ben_train_raw = train_all.filter(F.col("y") == 0)
        report = profile(ben_train_raw, feats)
        report.to_csv(cfg.OUT_DIR / "metrics" / "data_quality_train.csv", index=False)
        ben_train = ben_train_raw.filter(valid_cond(feats)).persist()
        n_raw, n_ben = ben_train_raw.count(), ben_train.count()
        if not n_ben:
            raise ValueError("No valid benign training records")
        log(f"Benign training {n_raw:,} -> {n_ben:,} records")
        cols, mean, std, feature_report = fit_feature_spec(ben_train, feats)
        if not cols:
            raise ValueError("Feature selection returned no features")
        feature_report.to_csv(cfg.OUT_DIR / "metrics" / "feature_report.csv", index=False)
        joblib.dump({"cols": cols, "mean": mean, "std": std, "z_clip": cfg.Z_CLIP}, cfg.OUT_DIR / "models" / "feature_spec.joblib")
        label_counts = all_df.groupBy("day", "label_raw").count().orderBy("day", "label_raw").toPandas()
        label_counts.to_csv(cfg.OUT_DIR / "metrics" / "label_distribution.csv", index=False)

        val_all = all_df.filter(F.col("day").isin(val_days))
        ben_val = val_all.filter(F.col("y") == 0)
        test = all_df.filter(F.col("day").isin(test_days))
        if cfg.TEST_FRAC < 1:
            test = test.sample(False, cfg.TEST_FRAC, cfg.SEED)
        sample = (ben_train.sample(False, min(1., cfg.SAMPLE / n_ben), cfg.SEED)
                  .select(*cols).toPandas())
        log(f"Training models on {len(sample):,} benign records")
        t0 = time.perf_counter()
        iso, ae = fit_models(sample, cols, mean, std)
        log(f"Models trained in {time.perf_counter() - t0:.1f}s")
        plot_ae_loss(ae, cfg.OUT_DIR / "figures" / "ae_loss.png")

        n_val = ben_val.count()
        if not n_val:
            raise ValueError("No benign validation records")
        vs = (ben_val.sample(False, min(1., 100_000 / n_val), cfg.SEED)
              .select(*cols).toPandas())
        z_val = to_z(vs, cols, mean, std)
        err_ref = ((z_val - ae.predict(z_val)) ** 2).mean(axis=0) + 1e-6
        bc = spark.sparkContext.broadcast((iso, ae, mean, std, err_ref))
        scorer = make_scorer(bc, cols)
        val_sc = (select_scoring_columns(ben_val, cols)
                  .mapInPandas(scorer, SCORE_SCHEMA)
                  .select("s_iso", "s_ae").toPandas())
        score_norm, thr, val_sc = calibrate(val_sc, cfg.FPR_TARGET)
        log(f"Calibrated threshold: {thr}")

        t0 = time.perf_counter()
        test_s = (fuse_scores(select_scoring_columns(test, cols).mapInPandas(scorer, SCORE_SCHEMA), score_norm)
                  .persist())
        n_test = test_s.count()
        valfull_s = fuse_scores(select_scoring_columns(val_all, cols).mapInPandas(scorer, SCORE_SCHEMA), score_norm).persist()
        names = {"s_iso": "Isolation Forest", "s_ae": "Autoencoder", "s_ens": "Ensemble AE+IF"}
        metrics = {"fpr_target": cfg.FPR_TARGET, "test_frac": cfg.TEST_FRAC,
                   "days": {"train": train_days, "val": val_days, "test": test_days}}
        frames = []
        for c, name in names.items():
            result = evaluate(test_s, c, thr[c])
            result["ALL"] = evaluate(test_s.withColumn("day", F.lit("ALL")), c, thr[c])["ALL"]
            metrics[name] = result
            frames.append(metrics_df(result, name))
        mdf = pd.concat(frames, ignore_index=True)
        mdf.to_csv(cfg.OUT_DIR / "metrics" / "metrics_by_day.csv", index=False)
        plot_metrics(mdf, cfg.FPR_TARGET, cfg.OUT_DIR / "figures" / "metrics_by_day.png")
        metrics["val_day_ensemble"] = evaluate(valfull_s, "s_ens", thr["s_ens"])
        by_attack = (test_s.filter(F.col("y") == 1).groupBy("day", "label_raw")
                     .agg(F.count("*").alias("n"), F.avg((F.col("s_ens") > thr["s_ens"]).cast("double")).alias("recall"))
                     .toPandas())
        metrics["recall_by_attack"] = by_attack.to_dict("records")
        by_attack.to_csv(cfg.OUT_DIR / "metrics" / "recall_by_attack.csv", index=False)
        top_feat = (test_s.filter((F.col("y") == 1) & (F.col("s_ens") > thr["s_ens"]))
                    .groupBy("top_feat").count().orderBy(F.desc("count")).limit(8).toPandas())
        metrics["top_features_of_detected_attacks"] = dict(zip(top_feat.top_feat, top_feat["count"]))

        if cfg.WITH_SUPERVISED:
            from src.baselines import run_baselines
            metrics = run_baselines(all_df, train_days, val_days, test, feats, cols, mean, std, metrics)

        model_art = {"iso": iso, "ae": ae, "mean": mean, "std": std, "cols": cols,
                     "err_ref": err_ref, "score_norm": score_norm, "thresholds": thr,
                     "sklearn": sklearn.__version__}
        joblib.dump(model_art, cfg.OUT_DIR / "models" / "model.joblib")
        (cfg.OUT_DIR / "metrics" / "config.json").write_text(
            json.dumps({"feats_all": feats, "cols": cols, "train_days": train_days,
                        "val_days": val_days, "test_days": test_days, "thresholds": thr,
                        "fpr_target": cfg.FPR_TARGET, "sample": len(sample)}, indent=2), encoding="utf-8")
        (cfg.OUT_DIR / "metrics" / "metrics.json").write_text(json.dumps(
            metrics, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)), encoding="utf-8")

        # Persist prediction parquet for future dashboard
        predictions = test_s.withColumn("is_anomaly", (F.col("s_ens") > thr["s_ens"]).cast("int"))
        predictions.write.mode("overwrite").partitionBy("day").parquet(str(cfg.OUT_DIR / "predictions" / "test_scored"))
        log(f"Scored {n_test:,} flows in {time.perf_counter() - t0:.1f}s (including evaluation & saves)")
        log(f"Outputs saved to {cfg.OUT_DIR}")
        valfull_s.unpersist()
        test_s.unpersist()
        ben_train.unpersist()
        bc.unpersist()
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
