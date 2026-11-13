"""Read-only presentation queries for the replay dashboard."""
from contextlib import closing
import json
from pathlib import Path
import time

import pandas as pd

from streaming.store import connect

ZONE = "Asia/Ho_Chi_Minh"


def vietnam_time(milliseconds):
    if milliseconds is None:
        return "Chưa có dữ liệu"
    return pd.Timestamp(milliseconds, unit="ms", tz="UTC").tz_convert(ZONE).strftime("%d/%m/%Y %H:%M:%S")


def read_live(path, now_ms=None):
    """One read transaction; indexed queries limited to current time windows."""
    now = time.time_ns() // 1_000_000 if now_ms is None else now_ms
    minute = now // 60000 * 60000
    start = minute - 59 * 60000
    with closing(connect(path, readonly=True)) as db:
        db.execute("BEGIN")
        flows, alerts = db.execute("SELECT flows, alerts FROM totals WHERE id=1").fetchone()
        last = db.execute("SELECT processed_ms FROM predictions ORDER BY processed_ms DESC LIMIT 1").fetchone()
        recent = db.execute("SELECT COUNT(*) FROM predictions WHERE processed_ms > ? AND processed_ms <= ?",
                            (now - 60000, now)).fetchone()[0]
        buckets = db.execute("""SELECT CAST(processed_ms / 60000 AS INTEGER)*60000, COUNT(*), SUM(alert)
            FROM predictions WHERE processed_ms >= ? AND processed_ms <= ?
            GROUP BY CAST(processed_ms / 60000 AS INTEGER)""", (start, now)).fetchall()
        latest = pd.read_sql_query("""SELECT event_id, source_day, score, top_feat, event_ms, processed_ms
            FROM predictions WHERE alert=1 ORDER BY processed_ms DESC, rowid DESC LIMIT 25""", db)
        binding = db.execute("SELECT signature FROM binding WHERE id=1").fetchone()
    try:
        broker = json.loads(binding[0]).get("broker") if binding else None
    except (ValueError, AttributeError):
        broker = None
    timeline = pd.DataFrame({"minute_ms": range(start, minute+1, 60000)}).merge(
        pd.DataFrame(buckets, columns=["minute_ms", "Processed Flows", "Alerts"]), how="left", on="minute_ms")
    for column in ("Processed Flows", "Alerts"):
        timeline[column] = pd.to_numeric(timeline[column]).fillna(0).astype(int)
    # Local wall-time is explicit, so browser timezone cannot shift the labels.
    timeline["Time"] = pd.to_datetime(timeline.minute_ms, unit="ms", utc=True).dt.tz_convert(
        ZONE).dt.strftime("%H:%M")
    return dict(flows=flows, alerts=alerts, alert_rate=alerts/flows if flows else 0.,
                rate=recent/60, last_ms=last[0] if last else None, broker=broker,
                timeline=timeline, latest=latest)


def alert_table(latest):
    view = latest.rename(columns={"source_day": "Source Day", "score": "Anomaly Score", "top_feat": "Top Feature"}).copy()
    view["Event ID"] = view.event_id.map(lambda s: s if len(s) <= 14 else "…" + s[-10:])
    for source, label in (("event_ms", "Replay Time"), ("processed_ms", "Processed Time")):
        view[label] = view[source].map(vietnam_time)
    return view[["Event ID", "Source Day", "Anomaly Score", "Top Feature", "Replay Time", "Processed Time"]]


def kafka_status(address):
    """Bounded Kafka metadata probe. No network calls to external addresses."""
    from streaming.common import local_broker
    try:
        local_broker(address or "")
    except ValueError:
        return "Unknown", "Không có địa chỉ loopback hợp lệ trong binding."
    try:
        from confluent_kafka.admin import AdminClient
        client = AdminClient({"bootstrap.servers": address, "socket.timeout.ms": 1000,
                              "socket.connection.setup.timeout.ms": 1000})
        metadata = client.list_topics(timeout=1)
        if metadata.brokers and all(b.host in ("127.0.0.1", "localhost") for b in metadata.brokers.values()):
            return "Online", "Kafka phản hồi metadata qua loopback."
        return "Unknown", "Metadata không xác nhận được broker local."
    except ImportError:
        return "Unknown", "Chưa cài Kafka client để kiểm tra metadata."
    except Exception:
        return "Offline", "Kafka không phản hồi metadata trong 1 giây; kết quả tại lần kiểm tra gần nhất."
