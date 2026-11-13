"""Opt-in test using a real local Kafka broker and Spark Kafka connector."""
import json
import os
from contextlib import closing
from pathlib import Path
import subprocess
import sys
import uuid

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data_io import load_days
from src.preprocessing import sanitize_features, to_z
from src.scoring import score_pandas
from src.spark_session import create_spark
from streaming.common import MODEL, ROOT, SNAPSHOT, local_broker
from streaming.replay import event_rows, replay
from streaming.store import connect, dashboard_snapshot


def test_kafka_checkpoint_restart_and_dashboard(tmp_path, monkeypatch):
    broker = os.getenv("REPLAY_TEST_BROKER")
    if not broker:
        pytest.skip("Set REPLAY_TEST_BROKER and Spark Kafka connector to run the broker test")
    from confluent_kafka import Consumer, TopicPartition
    from confluent_kafka.admin import AdminClient, NewTopic
    from streamlit.testing.v1 import AppTest

    broker = local_broker(broker)
    topic = "replay-integration-" + uuid.uuid4().hex
    admin = AdminClient({"bootstrap.servers": broker})
    admin.create_topics([NewTopic(topic, num_partitions=1, replication_factor=1)])[topic].result(15)
    day = "2018-02-21"
    state = Path(os.getenv("REPLAY_TEST_STATE", str(tmp_path / "state")))
    state.mkdir(parents=True, exist_ok=False)
    db = state / "predictions.sqlite"
    monkeypatch.setenv("TRAFFIC_REPLAY_DB", str(db))
    app = AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=20).run()
    app.radio(key="page").set_value("Live Detection").run()
    assert "đang chờ prediction store" in app.info[0].value

    def consume(log_name):
        with (state / log_name).open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, "-X", "utf8", "-u", "-m", "streaming.consumer",
                "--broker", broker, "--topic", topic, "--state", str(state), "--max-offsets", "100",
                "--available-now"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=150)
        assert result.returncode == 0, (state / log_name).read_text(encoding="utf-8")

    assert replay(SNAPSHOT, MODEL, day, 200, 500, broker, topic) == 200
    consume("first_consumer.log")
    assert dashboard_snapshot(db)["flows"] == 200
    app.run()
    assert not app.exception and app.metric[0].value == "200"
    assert int(app.metric[1].value.replace(",", "")) > 0

    # Replay the original 200 and append 30 distinct rows, then restart the JVM.
    assert replay(SNAPSHOT, MODEL, day, 230, 500, broker, topic) == 230
    client = Consumer({"bootstrap.servers": broker, "group.id": topic + "-audit"})
    try:
        assert client.get_watermark_offsets(TopicPartition(topic, 0), timeout=10) == (0, 430)
    finally:
        client.close()
    consume("restart_consumer.log")
    with closing(connect(db, readonly=True)) as connection:
        streamed = pd.read_sql_query("SELECT * FROM predictions ORDER BY event_id", connection)
    assert len(streamed) == streamed.event_id.nunique() == 230
    assert (state / "checkpoint" / "commits").is_dir()
    app.run()
    assert not app.exception and app.metric[0].value == "230"
    assert dashboard_snapshot(db)["alerts"] == int(streamed.alert.sum())

    # Independent batch preprocessing through the existing Spark path.
    art = joblib.load(MODEL)
    events = list(event_rows(SNAPSHOT, art["cols"], day, 230))
    ids = [r["event_id"] for r in events]
    manifest = json.loads((SNAPSHOT / "manifest.json").read_text())
    source = next(SNAPSHOT / r["file"] for r in manifest["counts"] if r["day"] == day)
    spark = create_spark("local[2]")
    try:
        raw, feats = load_days(spark, [source])
        batch = sanitize_features(raw.filter(raw.record_id.isin(ids)), feats).select(
            "record_id", *art["cols"]).toPandas().sort_values("record_id")
    finally:
        spark.stop()
    assert batch.record_id.tolist() == streamed.event_id.tolist()
    wire_features = pd.DataFrame([e["features"] for e in events], index=ids).loc[streamed.event_id]
    transform = (art["cols"], art["mean"], art["std"])
    np.testing.assert_array_equal(to_z(batch, *transform), to_z(wire_features, *transform))
    expected = score_pandas(batch, art)
    # Float32 AE reconstruction can vary slightly with batch size.
    np.testing.assert_allclose(streamed.score, expected.score, rtol=0, atol=3e-5)
    np.testing.assert_array_equal(streamed.alert, expected.alert)
    print("BATCH_STREAM_MATCH", len(streamed),
          "max_abs_error", float(np.max(np.abs(streamed.score.to_numpy()-expected.score.to_numpy()))),
          "alerts", int(streamed.alert.sum()), "state", state)
