"""Local Spark session for VS Code (not a distributed cluster)."""
import os
import sys

from . import config as cfg
from pyspark.sql import SparkSession


def _worker_probe(batches):
    """Exercise the same Python/Arrow path used by model scoring."""
    for batch in batches:
        yield batch


def create_spark():
    # Local workers must use the notebook/CLI interpreter, not python from PATH
    # (which may be a different environment or a Windows Store alias).
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    # Notebook sys.path changes belong only to the driver. Workers also need
    # the project root to unpickle functions and import src.preprocessing.
    worker_path = os.pathsep.join(filter(None, (
        str(cfg.ROOT), os.environ.get("PYTHONPATH", "")
    )))
    os.environ["PYTHONPATH"] = worker_path
    cfg.TMP_DIR.mkdir(parents=True, exist_ok=True)
    spark = (SparkSession.builder.appName("network-anomaly-vscode")
             .master("local[*]")
             .config("spark.pyspark.python", sys.executable)
             .config("spark.pyspark.driver.python", sys.executable)
             .config("spark.executorEnv.PYTHONPATH", worker_path)
             .config("spark.driver.memory", cfg.SPARK_DRIVER_MEMORY)
             .config("spark.sql.shuffle.partitions", cfg.SPARK_SHUFFLE_PARTITIONS)
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .config("spark.sql.execution.arrow.maxRecordsPerBatch", "20000")
             .config("spark.sql.sources.partitionColumnTypeInference.enabled", "false")
             .config("spark.local.dir", str(cfg.TMP_DIR))
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        spark.range(1, numPartitions=1).mapInPandas(
            _worker_probe, "id long"
        ).collect()
    except Exception as exc:
        spark.stop()
        compatibility_hint = ""
        if sys.platform == "win32" and sys.version_info >= (3, 12):
            compatibility_hint = (
                "Windows with Python 3.12+ can hit SPARK-53759 on older Spark "
                "releases (including 3.5.8). Use the project's Python 3.11 "
                "environment and restart the notebook kernel. "
            )
        raise RuntimeError(
            "Spark Python/Arrow worker startup check failed before training. "
            f"Python: {sys.executable} ({sys.version.split()[0]}). "
            + compatibility_hint +
            "Restart the notebook kernel (or Python process) after changing the "
            "environment; an existing SparkContext retains its worker settings. "
            "Check the worker stderr below for missing packages/startup errors "
            "and whether firewall/antivirus blocks local Python-Java connections. "
            f"Worker error: {exc}"
        ) from exc
    return spark
