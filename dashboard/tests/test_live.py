from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from dashboard.live import alert_table, kafka_status, read_live, vietnam_time
from streaming.store import connect, initialize, write_predictions


def test_readonly_window_timezone_and_full_id(tmp_path):
    db = tmp_path / "live.sqlite"
    initialize(db, "test")
    now = 1794564000000
    row = dict(event_id="source:" + "a"*64 + ":100", event_ms=now-1000, source_day="2018-02-21",
               score=5.246547, alert=True, top_feat="active_std", partition=0, offset=0)
    write_predictions(db, [dict(row, event_id="old")], now_ms=now-3600001)
    write_predictions(db, [row, dict(row, event_id="clean", alert=False)], now_ms=now-1000)
    with closing(connect(db, readonly=True)) as con:
        before = con.execute("SELECT * FROM predictions ORDER BY event_id").fetchall()
    data = read_live(db, now)
    assert (data["flows"], data["alerts"], data["alert_rate"], data["rate"]) == (3, 2, 2/3, 2/60)
    assert len(data["timeline"]) == 60
    assert data["timeline"]["Processed Flows"].sum() == 2
    assert data["timeline"].Alerts.sum() == 1
    assert data["last_ms"] == now-1000
    view = alert_table(data["latest"])
    assert view.iloc[0]["Event ID"] == "…" + row["event_id"][-10:]
    assert data["latest"].iloc[0].event_id == row["event_id"]
    assert vietnam_time(1767225600000) == "01/01/2026 07:00:00"
    assert view.iloc[0]["Replay Time"] == vietnam_time(now-1000)
    with closing(connect(db, readonly=True)) as con:
        assert before == con.execute("SELECT * FROM predictions ORDER BY event_id").fetchall()


def test_empty_and_status_evidence(tmp_path, monkeypatch):
    db = tmp_path / "empty.sqlite"
    initialize(db, "test")
    data = read_live(db)
    assert data["flows"] == data["alerts"] == data["alert_rate"] == 0
    assert data["last_ms"] is None
    assert kafka_status(None)[0] == kafka_status("outside:9092")[0] == "Unknown"
    class Client:
        def __init__(self, settings):
            pass
        def list_topics(self, timeout):
            assert timeout == 1
            return SimpleNamespace(brokers={1: SimpleNamespace(host="127.0.0.1")})
    monkeypatch.setitem(sys.modules, "confluent_kafka.admin", SimpleNamespace(AdminClient=Client))
    assert kafka_status("127.0.0.1:9092")[0] == "Online"
    class Offline(Client):
        def list_topics(self, timeout):
            raise TimeoutError()
    monkeypatch.setitem(sys.modules, "confluent_kafka.admin", SimpleNamespace(AdminClient=Offline))
    assert kafka_status("127.0.0.1:9092")[0] == "Offline"


def test_live_refresh_uses_new_rows_and_preserves_ids(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    db = tmp_path / "live.sqlite"
    initialize(db, "test")
    monkeypatch.setenv("TRAFFIC_REPLAY_DB", str(db))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
    app.radio(key="page").set_value("Live Detection").run()
    assert not app.exception
    assert [m.value for m in app.metric] == ["0", "0", "0.00%", "0.00"]
    row = dict(event_id="source:" + "b"*64 + ":123", event_ms=1, source_day="2018-02-21",
               score=6., alert=True, top_feat="active_std", partition=0, offset=0)
    write_predictions(db, [row])
    app.run()
    assert not app.exception
    assert [m.value for m in app.metric[:3]] == ["1", "1", "100.00%"]
    assert any(row["event_id"] in code.value for code in app.code)
    assert any("Spark Consumer" in m.value and "Unknown" in m.value for m in app.markdown)
