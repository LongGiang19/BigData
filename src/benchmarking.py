"""Stage-isolated, inference-only benchmark. Run with python -m src.benchmarking."""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from functools import reduce
from pathlib import Path

from . import config as cfg

SNAPSHOT = cfg.TEST_SNAPSHOT_DIR / "c5af9b674f2f1eb8d6aacecc"
MODEL = cfg.ROOT / ".cleaning_check/test100_20261109_233550_850520/outputs/models/model.joblib"
STAGES = ("read", "preprocessing", "scoring", "write", "end_to_end")


def prepare_inputs(snapshot, output, sizes):
    """Create exact, nested physical prefixes per day outside the timed runs."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .test_snapshot import _sha256

    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    counts = manifest["counts"]
    total = sum(r["sample_rows"] for r in counts)
    assert manifest["settings"]["fraction"] == 1 and len(counts) == 7
    assert all(r["sample_rows"] == r["source_rows"] for r in counts)
    for row in counts:
        assert _sha256(snapshot / row["file"]) == row["sample_sha256"]
    inputs = {}
    for size in sizes:
        if not 0 < size <= total:
            raise ValueError(f"Unsupported input size {size}; snapshot has {total} rows")
        paths = []
        cumulative = 0
        for row in counts:
            source = snapshot / row["file"]
            start = cumulative * size // total
            cumulative += row["sample_rows"]
            quota = cumulative * size // total - start
            if size == total:
                paths.append(str(source.resolve()))
                continue
            path = output / "inputs" / str(size) / source.name
            path.parent.mkdir(parents=True, exist_ok=True)
            parquet = pq.ParquetFile(source)
            with pq.ParquetWriter(path, parquet.schema_arrow, compression="snappy") as writer:
                left = quota
                for batch in parquet.iter_batches(batch_size=32768):
                    if not left:
                        break
                    part = batch.slice(0, min(left, len(batch)))
                    writer.write_table(pa.Table.from_batches([part]))
                    left -= len(part)
                assert left == 0
            assert pq.ParquetFile(path).metadata.num_rows == quota
            paths.append(str(path.resolve()))
        inputs[str(size)] = paths
    return inputs


def timed_trial(spark, paths, target, destination, art):
    from pyspark import StorageLevel
    from pyspark.sql import functions as F
    from .data_io import read_day, standardize
    from .preprocessing import sanitize_features
    from .scoring import SCORE_SCHEMA, fuse_scores, make_scorer, select_scoring_columns
    from .utils import day_of

    cached = []
    bc = None
    def cache(frame):
        frame.persist(StorageLevel.DISK_ONLY)
        cached.append(frame)
        return frame
    def union(parts):
        return reduce(lambda left, right: left.unionByName(right), parts)

    try:
        start = time.perf_counter()
        raw = [cache(read_day(spark, p).drop("record_id")) for p in paths]
        raw_count = sum(frame.count() for frame in raw)
        read_end = time.perf_counter()
        assert raw_count == target
        read_partitions = sum(frame.rdd.getNumPartitions() for frame in raw)

        prep_start = time.perf_counter()
        parts = []
        for frame, path in zip(raw, paths):
            standardized, feats = standardize(frame, day_of(path))
            parts.append(select_scoring_columns(sanitize_features(standardized, feats), art["cols"]))
        clean = cache(union(parts))
        assert clean.count() == target
        prep_end = time.perf_counter()
        scoring_partitions = clean.rdd.getNumPartitions()
        for frame in raw:
            frame.unpersist(blocking=True)

        score_start = time.perf_counter()
        bc = spark.sparkContext.broadcast(tuple(art[k] for k in ("iso", "ae", "mean", "std", "err_ref")))
        scored = clean.mapInPandas(make_scorer(bc, art["cols"]), SCORE_SCHEMA)
        scored = fuse_scores(scored, art["score_norm"]).withColumn(
            "is_anomaly", (F.col("s_ens") > art["thresholds"]["s_ens"]).cast("int"))
        scored = cache(scored)
        totals = scored.agg(F.count("*").alias("n"), *[F.count(k).alias(k) for k in
                             ("s_iso", "s_ae", "s_ens", "top_feat", "is_anomaly")]).first()
        assert all(v == target for v in totals)
        score_end = time.perf_counter()
        clean.unpersist(blocking=True)

        write_start = time.perf_counter()
        scored.write.mode("error").partitionBy("day").parquet(str(destination))
        end = time.perf_counter()
        # Readback is outside timing. Parquet footers independently verify row counts.
        import pyarrow.parquet as pq
        files = list(destination.rglob("*.parquet"))
        assert (destination / "_SUCCESS").exists()
        assert sum(pq.ParquetFile(p).metadata.num_rows for p in files) == target
        durations = dict(read=read_end-start, preprocessing=prep_end-prep_start,
                         scoring=score_end-score_start, write=end-write_start, end_to_end=end-start)
        return dict(rows=target, read_partitions=read_partitions, scoring_partitions=scoring_partitions,
                    parquet_files=len(files), output_bytes=sum(p.stat().st_size for p in files),
                    **{f"{k}_seconds": v for k, v in durations.items()},
                    **{f"{k}_flows_per_second": target/v for k, v in durations.items()})
    finally:
        for frame in reversed(cached):
            frame.unpersist(blocking=True)
        if bc is not None:
            bc.destroy(blocking=True)


def worker(output, master, sizes, repeats):
    import joblib
    from .spark_session import create_spark

    plan = json.loads((output / "input_manifest.json").read_text(encoding="utf-8"))
    tag = master.replace("[", "_").replace("]", "").replace("*", "all")
    cfg.TMP_DIR = output / "scratch" / tag
    art = joblib.load(plan["model"])
    spark = create_spark(master)
    spark.sparkContext.setLogLevel("WARN")
    try:
        environment = dict(master=spark.sparkContext.master, logical_cpu=os.cpu_count(),
                           spark_slots=spark.sparkContext.defaultParallelism,
                           driver_memory=spark.sparkContext.getConf().get("spark.driver.memory"),
                           shuffle_partitions=spark.conf.get("spark.sql.shuffle.partitions"),
                           arrow_batch_rows=spark.conf.get("spark.sql.execution.arrow.maxRecordsPerBatch"),
                           spark_version=spark.version, python=platform.python_version(), os=platform.platform(),
                           java=spark._jvm.java.lang.System.getProperty("java.version"),
                           blas_threads=1, cache_storage="DISK_ONLY")
        for size in sizes:
            for repetition in range(1, repeats + 1):
                dest = output / "predictions" / tag / str(size) / str(repetition)
                print("START", master, size, repetition, flush=True)
                result = timed_trial(spark, plan["inputs"][str(size)], size, dest, art)
                row = dict(**environment, requested_rows=size, repeat=repetition,
                           run_type="first-size-pass" if repetition == 1 else "warm-repeat", **result)
                with (output / "trials.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                print("DONE", json.dumps(row), flush=True)
    finally:
        spark.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=cfg.OUT_DIR / "benchmark")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100000, 500000, 1000000, 4653501])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--core-comparison", action="store_true")
    parser.add_argument("--worker")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.worker:
        worker(output, args.worker, args.sizes, args.repeats)
        return
    output.mkdir(parents=True, exist_ok=False)
    temp = output / "temp"
    temp.mkdir()
    inputs = prepare_inputs(args.snapshot, output, args.sizes)
    (output / "input_manifest.json").write_text(json.dumps(dict(
        model=str(args.model.resolve()), snapshot=str(args.snapshot.resolve()), inputs=inputs,
        selection="proportional physical prefix per day; full snapshot read directly"), indent=2), encoding="utf-8")
    env = dict(os.environ, TMP=str(temp), TEMP=str(temp), TMPDIR=str(temp),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    # A new process is required to isolate JVM gateway and SparkContext.
    plans = [("local[*]", args.sizes)]
    if args.core_comparison:
        plans += [(f"local[{cores}]", [min(args.sizes)]) for cores in (1, 2, 4)]
    try:
        for master, sizes in plans:
            subprocess.run([sys.executable, "-X", "utf8", "-u", "-m", "src.benchmarking",
                            "--output", str(output), "--worker", master, "--repeats", str(args.repeats),
                            "--sizes", *map(str, sizes)], cwd=cfg.ROOT, env=env, check=True)
    finally:
        if (output / "trials.jsonl").exists():
            summarize(output)


def summarize(output):
    from .benchmark_report import write_report
    write_report(output)


if __name__ == "__main__":
    main()
