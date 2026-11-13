import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / ".cleaning_check/test100_20261109_233550_850520/outputs/models/model.joblib"
SNAPSHOT = ROOT / "data/curated/test_snapshots/c5af9b674f2f1eb8d6aacecc"
STATE = ROOT / "streaming/runtime"
TOPIC = "traffic-replay"


def local_broker(address):
    match = re.fullmatch(r"127\.0\.0\.1:([0-9]{1,5})", address)
    if not match or not 1 <= int(match[1]) <= 65535:
        raise ValueError("Only a single 127.0.0.1:<port> Kafka broker is allowed")
    return address


def check_topic(client, topic):
    metadata = client.list_topics(timeout=10)
    if not metadata.brokers or any(b.host not in ("127.0.0.1", "localhost")
                                   for b in metadata.brokers.values()):
        raise ValueError("Kafka advertised a non-loopback endpoint")
    if topic not in metadata.topics or metadata.topics[topic].error:
        raise ValueError(f"Create the local Kafka topic first: {topic}")
