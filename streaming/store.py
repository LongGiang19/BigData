"""Transactional SQLite prediction sink and bounded dashboard queries."""
from contextlib import closing
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS binding (id INTEGER PRIMARY KEY CHECK(id=1), signature TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS predictions (
    event_id TEXT PRIMARY KEY, event_ms INTEGER NOT NULL, source_day TEXT NOT NULL,
    score REAL NOT NULL, alert INTEGER NOT NULL CHECK(alert IN (0,1)),
    top_feat TEXT NOT NULL, processed_ms INTEGER NOT NULL,
    kafka_partition INTEGER NOT NULL, kafka_offset INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS prediction_time ON predictions(processed_ms);
CREATE INDEX IF NOT EXISTS alerts_time ON predictions(alert, processed_ms DESC);
CREATE TABLE IF NOT EXISTS totals (id INTEGER PRIMARY KEY CHECK(id=1), flows INTEGER NOT NULL, alerts INTEGER NOT NULL);
INSERT OR IGNORE INTO totals VALUES (1, 0, 0);
CREATE TABLE IF NOT EXISTS minutes (minute_ms INTEGER PRIMARY KEY, alerts INTEGER NOT NULL);
"""


def connect(path, readonly=False):
    if readonly:
        return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    return sqlite3.connect(path, timeout=10)


def initialize(path, signature):
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(connect(path)) as db, db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript(SCHEMA)
        db.execute("INSERT OR IGNORE INTO binding VALUES (1, ?)", (signature,))
        if db.execute("SELECT signature FROM binding").fetchone()[0] != signature:
            raise ValueError("Prediction store belongs to another checkpoint, topic or model")


def write_predictions(path, rows, now_ms=None):
    now = time.time_ns() // 1_000_000 if now_ms is None else now_ms
    inserted = alerts = 0
    with closing(connect(path)) as db, db:
        for row in rows:
            changed = db.execute(
                """INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(event_id) DO NOTHING""",
                (row["event_id"], row["event_ms"], row["source_day"], row["score"],
                 int(row["alert"]), row["top_feat"], now, row["partition"], row["offset"])).rowcount
            inserted += changed
            alerts += changed * int(row["alert"])
        db.execute("UPDATE totals SET flows=flows+?, alerts=alerts+? WHERE id=1", (inserted, alerts))
        if alerts:
            db.execute("""INSERT INTO minutes VALUES (?, ?) ON CONFLICT(minute_ms)
                          DO UPDATE SET alerts=alerts+excluded.alerts""", (now//60000*60000, alerts))
    return inserted


def dashboard_snapshot(path, now_ms=None):
    now = time.time_ns() // 1_000_000 if now_ms is None else now_ms
    with closing(connect(path, readonly=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        flows, alerts = db.execute("SELECT flows, alerts FROM totals").fetchone()
        recent = db.execute("SELECT COUNT(*) FROM predictions WHERE processed_ms > ? AND processed_ms <= ?",
                            (now-60000, now)).fetchone()[0]
        trend = [dict(r) for r in db.execute("SELECT * FROM minutes WHERE minute_ms >= ? ORDER BY minute_ms",
                                           (now//60000*60000-59*60000,))]
        latest = [dict(r) for r in db.execute("""SELECT event_id, event_ms, source_day, score, top_feat, processed_ms
                FROM predictions WHERE alert=1 ORDER BY processed_ms DESC, rowid DESC LIMIT 25""")]
    return dict(flows=flows, alerts=alerts, flows_per_second=recent/60, trend=trend, latest=latest)
