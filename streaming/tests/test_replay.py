import json
import sqlite3

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.scoring import score_pandas
from src.test_snapshot import _sha256
from streaming.common import MODEL, SNAPSHOT, local_broker
from streaming.consumer import score_frames
from streaming.replay import event_rows
from streaming.store import dashboard_snapshot, initialize, write_predictions


def test_only_loopback():
    assert local_broker("127.0.0.1:9092") == "127.0.0.1:9092"
    for value in ("example.org:9092", "0.0.0.0:9092", "127.0.0.1:0",
                  "127.0.0.1:9092,example.org:9092"):
        with pytest.raises(ValueError):
            local_broker(value)


def test_replay_id_and_protocol(tmp_path):
    path = tmp_path / "flows.parquet"
    pq.write_table(pa.table({"record_id": ["stable-a", "stable-b"], "Protocol": [6, None],
                            "Flow Duration": [float("inf"), -2.]}), path)
    (tmp_path / "manifest.json").write_text(json.dumps(dict(counts=[
        dict(day="2018-02-21", file=path.name, sample_sha256=_sha256(path))])))
    cols = ["is_tcp", "is_udp", "is_other_proto", "flow_duration"]
    rows = list(event_rows(tmp_path, cols, "2018-02-21", 2))
    assert rows == list(event_rows(tmp_path, cols, "2018-02-21", 2))
    assert rows[0]["event_id"] == "stable-a"
    assert rows[0]["features"] == dict(is_tcp=1, is_udp=0, is_other_proto=0, flow_duration=None)
    assert rows[1]["features"] == dict(is_tcp=None, is_udp=None, is_other_proto=None, flow_duration=-2)
    json.dumps(rows, allow_nan=False)


def test_transaction_and_idempotent_dashboard(tmp_path):
    db = tmp_path / "predictions.sqlite"
    initialize(db, "test")
    row = dict(event_id="one", event_ms=120001, source_day="2018-02-21",
               score=6., alert=True, top_feat="f", partition=0, offset=0)
    assert write_predictions(db, [row, row], now_ms=120005) == 1
    assert write_predictions(db, [dict(row, offset=1)], now_ms=120010) == 0
    with pytest.raises(sqlite3.IntegrityError):
        write_predictions(db, [dict(row, event_id="two"), dict(row, event_id="bad", score=None)])
    view = dashboard_snapshot(db, now_ms=120020)
    assert (view["flows"], view["alerts"], view["flows_per_second"]) == (1, 1, 1/60)
    assert view["trend"] == [dict(minute_ms=120000, alerts=1)]
    assert view["latest"][0]["event_id"] == "one"
    with pytest.raises(ValueError, match="another checkpoint"):
        initialize(db, "another")


def test_saved_model_inference_matches_existing_batch():
    if not MODEL.is_file() or not SNAPSHOT.is_dir():
        pytest.skip("Official snapshot and model are required")
    art = joblib.load(MODEL)
    flows = list(event_rows(SNAPSHOT, art["cols"], "2018-02-21", 12))
    frame = pd.DataFrame([dict(**r["features"], event_id=r["event_id"],
                              source_day=r["source_day"], event_ms=123, partition=0,
                              offset=i, valid=True) for i, r in enumerate(flows)])
    batch = score_pandas(frame, art)
    streamed = next(score_frames(iter([frame]), art))
    np.testing.assert_allclose(streamed.score, batch.score, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(streamed.alert, batch.alert)
    np.testing.assert_array_equal(streamed.event_id, frame.event_id)
    with pytest.raises(ValueError, match="Invalid replay event"):
        next(score_frames(iter([frame.assign(valid=False)]), art))
