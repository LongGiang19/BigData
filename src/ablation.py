"""Predeclared temporal ablations on the fixed 100% test snapshot."""
import argparse
import json
import os
import time
from pathlib import Path

from . import config as cfg

VARIANTS = {
    "paired_baseline": dict(signed_log=True, corr_thr=cfg.CORR_THR, clip=cfg.Z_CLIP),
    "no_signed_log": dict(signed_log=False, corr_thr=cfg.CORR_THR, clip=cfg.Z_CLIP),
    "no_correlation_filter": dict(signed_log=True, corr_thr=1.0, clip=cfg.Z_CLIP),
    "clip_5": dict(signed_log=True, corr_thr=cfg.CORR_THR, clip=5.0),
}
WEIGHTS = (.25, .50, .75)
SNAPSHOT = cfg.TEST_SNAPSHOT_DIR / "c5af9b674f2f1eb8d6aacecc"
OFFICIAL = cfg.ROOT / ".cleaning_check/test100_20261109_233550_850520"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=cfg.ROOT / "outputs/ablation_study")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    temp = out / "temp"
    temp.mkdir()
    os.environ.update(TMP=str(temp), TEMP=str(temp), TMPDIR=str(temp),
                      OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    import joblib
    import numpy as np
    from pyspark import StorageLevel
    from pyspark.sql import functions as F
    from .data_io import load_days
    from .models import fit_models
    from .preprocessing import fit_feature_spec, sanitize_features, to_z
    from .scoring import SCORE_SCHEMA, calibrate, make_scorer, select_scoring_columns
    from .spark_session import create_spark
    from .test_snapshot import _sha256
    from .utils import day_of

    manifest = json.loads((SNAPSHOT / "manifest.json").read_text(encoding="utf-8"))
    paths = [SNAPSHOT / row["file"] for row in manifest["counts"]]
    assert manifest["settings"]["fraction"] == 1.0
    assert {day_of(p) for p in paths} == {day_of(p) for p in cfg.TEST_FILES}
    assert all(_sha256(SNAPSHOT / r["file"]) == r["sample_sha256"] for r in manifest["counts"])
    source = {p.name: _sha256(p) for p in cfg.TRAIN_FILES + [cfg.VAL_FILE]}
    protocol = dict(variants=VARIANTS, if_weights=WEIGHTS, seed=cfg.SEED, fpr_target=cfg.FPR_TARGET,
                    snapshot=str(SNAPSHOT), official_reference=str(OFFICIAL), train_source_hashes=source,
                    train_days=[day_of(p) for p in cfg.TRAIN_FILES], val_days=[day_of(cfg.VAL_FILE)],
                    test_days=sorted(day_of(p) for p in paths), sample_target=cfg.SAMPLE,
                    train_sample_method="one Spark Bernoulli sample shared by all variants",
                    correlation_sample_method="one shared Spark train-only sample, capped at 100000",
                    blas_threads=1, driver_memory=cfg.SPARK_DRIVER_MEMORY,
                    shuffle_partitions=cfg.SPARK_SHUFFLE_PARTITIONS,
                    selection="predeclared descriptive comparisons; no test-based selection")
    (out / "experiment_config.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    started = time.perf_counter()
    cfg.TMP_DIR = out / "spark_tmp"
    spark = create_spark()
    try:
        train, feats = load_days(spark, cfg.TRAIN_FILES)
        train = sanitize_features(train.filter("y = 0"), feats).select(*feats).persist(StorageLevel.DISK_ONLY)
        n_train = train.count()
        sample = train.sample(False, min(1., cfg.SAMPLE/n_train), cfg.SEED).toPandas()
        corr = train.sample(False, min(1., 105000/n_train), cfg.SEED).limit(100000).toPandas()
        sample.to_parquet(out / "shared_training_sample.parquet", index=False)
        corr.to_parquet(out / "shared_correlation_sample.parquet", index=False)
        val, _ = load_days(spark, [cfg.VAL_FILE])
        val = sanitize_features(val.filter("y = 0"), feats).persist(StorageLevel.DISK_ONLY)
        n_val = val.count()
        val_ref = val.sample(False, min(1., 100000/n_val), cfg.SEED).select(*feats).toPandas()
        test, _ = load_days(spark, paths)
        test = sanitize_features(test, feats).persist(StorageLevel.DISK_ONLY)
        n_test = test.count()
        assert (n_train, n_val, n_test) == (1268743, 446619, 4653501)
        print("SHARED_DATA", n_train, len(sample), len(corr), n_val, n_test, flush=True)
        for name, settings in VARIANTS.items():
            start = time.perf_counter()
            folder = out / name
            folder.mkdir()
            opts = {key: settings[key] for key in ("signed_log", "clip")}
            cols, mean, std, report = fit_feature_spec(
                train, feats, signed_log=opts["signed_log"], corr_thr=settings["corr_thr"],
                correlation_sample=corr)
            report.to_csv(folder / "feature_report.csv", index=False)
            fit_start = time.perf_counter()
            iso, ae = fit_models(sample, cols, mean, std, **opts)
            fit_end = time.perf_counter()
            z = to_z(val_ref, cols, mean, std, **opts)
            err_ref = ((z - ae.predict(z))**2).mean(axis=0) + 1e-6
            bc = spark.sparkContext.broadcast((iso, ae, mean, std, err_ref))
            scorer = make_scorer(bc, cols, **opts)
            val_scores = (select_scoring_columns(val, cols).mapInPandas(scorer, SCORE_SCHEMA)
                          .select("s_iso", "s_ae").toPandas())
            assert len(val_scores) == n_val and np.isfinite(val_scores.to_numpy()).all()
            norms, thresholds, _ = calibrate(val_scores, cfg.FPR_TARGET)
            val_scores.to_parquet(folder / "validation_scores.parquet", index=False)
            joblib.dump(dict(iso=iso, ae=ae, cols=cols, mean=mean, std=std, err_ref=err_ref,
                             score_norm=norms, thresholds=thresholds, **opts), folder / "model.joblib")
            calibration_end = time.perf_counter()
            (select_scoring_columns(test, cols).mapInPandas(scorer, SCORE_SCHEMA)
             .select("day", "label_raw", "y", "s_iso", "s_ae").write.mode("error")
             .partitionBy("day").parquet(str(folder / "predictions")))
            end = time.perf_counter()
            bc.destroy(blocking=True)
            timing = dict(features=len(cols), fit_preprocessing_seconds=fit_start-start,
                          training_seconds=fit_end-fit_start, calibration_seconds=calibration_end-fit_end,
                          test_score_write_seconds=end-calibration_end, total_seconds=end-start)
            (folder / "runtime.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
            print("VARIANT_COMPLETE", name, timing, flush=True)
    finally:
        spark.stop()
    (out / "runtime.json").write_text(json.dumps(dict(seconds=time.perf_counter()-started)), encoding="utf-8")


if __name__ == "__main__":
    main()
