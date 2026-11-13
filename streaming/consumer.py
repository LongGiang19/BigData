"""Spark Structured Streaming micro-batches with an idempotent local sink."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from pyspark.sql import functions as F, types as T

from src import config as cfg
from src.scoring import score_pandas
from src.spark_session import create_spark
from .common import MODEL, STATE, TOPIC, check_topic, local_broker
from .store import initialize, write_predictions

META = [("event_id", T.StringType()), ("event_ms", T.LongType()), ("source_day", T.StringType())]
KAFKA = [("partition", T.IntegerType()), ("offset", T.LongType())]
RESULT = T.StructType([T.StructField(k, t, False) for k, t in [
    *META, *KAFKA, ("score", T.DoubleType()), ("alert", T.BooleanType()), ("top_feat", T.StringType())]])


def event_schema(cols):
    features = T.StructType([T.StructField(c, T.DoubleType()) for c in cols])
    return T.StructType([T.StructField(k, t) for k, t in [*META, ("features", features)]])


def parse_messages(frame, cols):
    raw = frame.select(F.col("value").cast("string").alias("json"), "partition", "offset")
    parsed = raw.withColumn("event", F.from_json("json", event_schema(cols)))
    keys = F.json_object_keys(F.get_json_object("json", "$.features"))
    required = F.array(*[F.lit(c) for c in cols])
    valid = (F.col("event.event_id").rlike(r"^.+:[0-9a-f]{64}:[0-9]+$")
             & (F.col("event.event_ms") > 0)
             & F.col("event.source_day").rlike(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
             & keys.isNotNull() & (F.size(F.array_except(required, keys)) == 0))
    return parsed.select(
        *[F.col(f"event.{k}").alias(k) for k, _ in META], "partition", "offset",
        F.coalesce(valid, F.lit(False)).alias("valid"),
        *[F.col(f"event.features.{c}").alias(c) for c in cols])


def score_frames(frames, art):
    for frame in frames:
        if not frame.valid.all():
            raise ValueError("Invalid replay event; inspect Kafka partition/offset before retrying")
        scores = score_pandas(frame, art)
        if not np.isfinite(scores.score).all():
            raise ValueError("Model returned non-finite replay scores")
        result = frame[[k for k, _ in META + KAFKA]].copy()
        for key in ("score", "alert", "top_feat"):
            result[key] = scores[key].to_numpy()
        yield result


def run(spark, model, broker, topic, state, max_offsets, trigger, available_now=False):
    from confluent_kafka.admin import AdminClient
    broker = local_broker(broker)
    check_topic(AdminClient({"bootstrap.servers": broker}), topic)
    if not 1 <= max_offsets <= 10000:
        raise ValueError("max_offsets must be between 1 and 10000")
    state = state.resolve()
    checkpoint, db = state / "checkpoint", state / "predictions.sqlite"
    if (checkpoint / "metadata").exists() and not db.is_file():
        raise ValueError("Checkpoint exists without its prediction store; restore the pair")
    with model.open("rb") as f:
        signature = json.dumps(dict(version=1, broker=broker, topic=topic,
                                    model=hashlib.file_digest(f, "sha256").hexdigest(),
                                    checkpoint=str(checkpoint)), sort_keys=True)
    initialize(db, signature)
    art = joblib.load(model)
    bc = spark.sparkContext.broadcast(art)
    stream = (spark.readStream.format("kafka").option("kafka.bootstrap.servers", broker)
              .option("subscribe", topic).option("startingOffsets", "earliest")
              .option("failOnDataLoss", "true").option("maxOffsetsPerTrigger", max_offsets).load())
    parsed = parse_messages(stream, art["cols"])

    def process(batch, batch_id):
        started = time.perf_counter()
        scored = batch.mapInPandas(lambda frames: score_frames(frames, bc.value), RESULT).collect()
        inserted = write_predictions(db, scored)
        print(json.dumps(dict(batch=int(batch_id), received=len(scored), inserted=inserted,
                              duplicates=len(scored)-inserted,
                              seconds=time.perf_counter()-started)), flush=True)

    writer = parsed.writeStream.option("checkpointLocation", checkpoint.as_uri()).foreachBatch(process)
    writer = writer.trigger(availableNow=True) if available_now else writer.trigger(processingTime=trigger)
    query = writer.start()
    try:
        query.awaitTermination()
    finally:
        if query.isActive:
            query.stop()
        bc.destroy(blocking=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=MODEL)
    p.add_argument("--state", type=Path, default=STATE)
    p.add_argument("--broker", default="127.0.0.1:9092")
    p.add_argument("--topic", default=TOPIC)
    p.add_argument("--max-offsets", type=int, default=1000)
    p.add_argument("--trigger", default="5 seconds")
    p.add_argument("--available-now", action="store_true")
    a = p.parse_args()
    cfg.TMP_DIR = a.state.resolve() / "spark_tmp"
    spark = create_spark("local[2]")
    try:
        run(spark, a.model, a.broker, a.topic, a.state, a.max_offsets, a.trigger, a.available_now)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
