"""One source of truth for paths, train/val/test and hyperparameters."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT / "data" / "raw"))).expanduser().resolve()
OUT_DIR = Path(os.getenv("OUTPUT_DIR", str(ROOT / "outputs"))).expanduser().resolve()
CURATED_DIR = ROOT / "data" / "curated" / "cse2018_curated"
TEST_SNAPSHOT_DIR = ROOT / "data" / "curated" / "test_snapshots"
TMP_DIR = ROOT / "data" / "spark_tmp"
SUFFIX = "_TrafficForML_CICFlowMeter.parquet"

def file_of(key):
    return DATA_DIR / (key + SUFFIX)

TRAIN_FILES = [file_of("Bruteforce-Wednesday-14-02-2018"), file_of("DoS1-Thursday-15-02-2018")]
VAL_FILE = file_of("DoS2-Friday-16-02-2018")
TEST_FILES = [file_of(k) for k in [
    "DDoS1-Tuesday-20-02-2018", "DDoS2-Wednesday-21-02-2018",
    "Web1-Thursday-22-02-2018", "Web2-Friday-23-02-2018",
    "Infil1-Wednesday-28-02-2018", "Infil2-Thursday-01-03-2018",
    "Botnet-Friday-02-03-2018",
]]
SAMPLE = int(os.getenv("SAMPLE", "300000"))
TEST_FRAC = float(os.getenv("TEST_FRAC", "0.25"))
FPR_TARGET = float(os.getenv("FPR_TARGET", "0.01"))
WITH_SUPERVISED = os.getenv("WITH_SUPERVISED", "1") == "1"
SKIP_INGEST = os.getenv("SKIP_INGEST", "0") == "1"
SEED = 42
EXCLUDE_FEATURES = {"label", "label_raw", "y", "day"}
ALLOW_NEG = {"init_fwd_win_bytes", "init_bwd_win_bytes"}
MIN_NONZERO = 1e-3
CORR_THR = 0.98
STD_FLOOR = 0.1
Z_CLIP = 10.0
SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "4g")
SPARK_SHUFFLE_PARTITIONS = os.getenv("SPARK_SHUFFLE_PARTITIONS", "8")
