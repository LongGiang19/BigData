"""Replay the official source snapshot as bounded, rate-controlled JSON."""
import argparse
import json
import math
from pathlib import Path
import time

import joblib
import pyarrow.parquet as pq

from src.test_snapshot import _sha256
from src.utils import norm
from .common import MODEL, SNAPSHOT, TOPIC, check_topic, local_broker

PROTOCOLS = {"is_tcp", "is_udp", "is_other_proto"}


def event_rows(snapshot, cols, day, limit):
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    rows = [row for row in manifest["counts"] if row["day"] == day]
    if len(rows) != 1:
        raise ValueError(f"Expected one snapshot file for {day}")
    path = snapshot / rows[0]["file"]
    if _sha256(path) != rows[0]["sample_sha256"]:
        raise ValueError(f"Snapshot checksum mismatch: {path.name}")
    reader = pq.ParquetFile(path)
    names = {norm(name): name for name in reader.schema_arrow.names}
    required = set(cols) - PROTOCOLS
    if set(cols) & PROTOCOLS:
        required.add("protocol")
    fields = [names[k] for k in sorted(required | {"record_id"})]
    remaining = limit
    for batch in reader.iter_batches(columns=fields, batch_size=min(2048, limit)):
        for row in batch.slice(0, remaining).to_pylist():
            protocol = row.get(names.get("protocol"))
            features = {}
            for name in cols:
                if name in PROTOCOLS:
                    value = None if protocol is None else {
                        "is_tcp": protocol == 6, "is_udp": protocol == 17,
                        "is_other_proto": protocol not in (6, 17),
                    }[name]
                else:
                    value = row[names[name]]
                features[name] = float(value) if value is not None and math.isfinite(value) else None
            event_id = row[names["record_id"]]
            if not event_id:
                raise ValueError("Snapshot row has no record_id")
            yield dict(event_id=event_id, source_day=day, features=features)
            remaining -= 1
        if remaining == 0:
            break


def replay(snapshot, model, day, limit, rate, broker, topic):
    from confluent_kafka import Producer
    if limit < 1 or not math.isfinite(rate) or rate <= 0:
        raise ValueError("Limit and rate must be positive")
    client = Producer({"bootstrap.servers": local_broker(broker), "enable.idempotence": True,
                       "allow.auto.create.topics": False, "message.timeout.ms": 15000})
    check_topic(client, topic)
    cols = joblib.load(model)["cols"]
    failures = []
    acknowledged = 0
    def delivered(error, message):
        nonlocal acknowledged
        if error:
            failures.append(str(error))
        else:
            acknowledged += 1
    started = time.monotonic()
    next_send = started
    sent = 0
    try:
        for event in event_rows(snapshot, cols, day, limit):
            time.sleep(max(0., next_send - time.monotonic()))
            event["event_ms"] = time.time_ns() // 1_000_000
            body = json.dumps(event, allow_nan=False, separators=(",", ":"))
            while True:
                try:
                    client.produce(topic, key=event["event_id"], value=body, on_delivery=delivered)
                    break
                except BufferError:
                    client.poll(.2)
            sent += 1
            next_send = time.monotonic() + 1/rate
            client.poll(0)
            if failures:
                break
    finally:
        pending = client.flush(20)
    if pending or failures or acknowledged != sent or not sent:
        raise RuntimeError(f"Kafka delivery incomplete: sent={sent}, ack={acknowledged}, pending={pending}, errors={failures[:3]}")
    print(json.dumps(dict(sent=sent, acknowledged=acknowledged, seconds=time.monotonic()-started)), flush=True)
    return acknowledged


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    p.add_argument("--model", type=Path, default=MODEL)
    p.add_argument("--day", default="2018-02-21")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--rate", type=float, default=100)
    p.add_argument("--broker", default="127.0.0.1:9092")
    p.add_argument("--topic", default=TOPIC)
    a = p.parse_args()
    replay(a.snapshot, a.model, a.day, a.limit, a.rate, a.broker, a.topic)


if __name__ == "__main__":
    main()
