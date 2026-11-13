"""Schema checks and small-table transformations for the dashboard."""
from dataclasses import dataclass
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "test100_20261109_233550_850520"
MODELS = {"Isolation Forest": "IF", "Autoencoder": "AE", "Ensemble": "Ensemble"}
RATE_METRICS = ("recall", "fpr", "precision", "f1", "roc_auc", "pr_auc")
COUNTS = ("TP", "FP", "TN", "FN")


class ArtifactError(RuntimeError):
    """An offline artifact is missing or incompatible."""


@dataclass(frozen=True)
class Artifacts:
    run: Path
    research: Path

    @classmethod
    def from_environment(cls):
        return cls(Path(os.getenv("DASHBOARD_FULL_RUN", ROOT / ".cleaning_check" / RUN_ID)).resolve(),
                   Path(os.getenv("DASHBOARD_RESEARCH_DIR", ROOT / "outputs")).resolve())

    def table(self, name):
        return {
            "overall": self.run / "independent_metrics.csv",
            "daily": self.run / "metrics_by_day_independent.csv",
            "features": self.research / "temporal_drift/feature_drift_by_day.csv",
            "scores": self.research / "temporal_drift/score_drift_by_day.csv",
            "attacks": self.research / "temporal_drift/attack_analysis.csv",
            "reversal": self.research / "ranking_reversal_audit/metrics.csv",
        }[name]


SCHEMAS = {
    "overall": {"model", "n", "threshold", *COUNTS, *RATE_METRICS},
    "daily": {"day", "model", "n", "threshold", *COUNTS, *RATE_METRICS},
    "features": {"day", "reference", "feature", "ks", "psi", "wasserstein", "recon_mse"},
    "scores": {"day", "model", "label", "reference", "threshold", "p10", "median", "p90", "p99", "ks"},
    "attacks": {"day", "attack", "model", "n", "above_threshold", "alert_rate"},
    "reversal": {"day", "model", "benign", "attack", "prevalence", "roc_auc", "roc_auc_negated", "pr_auc"},
}


def read_table(path, name):
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ArtifactError(f"Dashboard chỉ đọc bảng CSV tổng hợp: {path.name}")
    try:
        if path.stat().st_size > 5_000_000:
            raise ArtifactError(f"Bảng tổng hợp vượt giới hạn 5 MB: {path.name}")
        table = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ArtifactError(f"Không đọc được {path}: {exc}") from exc
    missing = SCHEMAS[name] - set(table.columns)
    if missing or table.empty:
        raise ArtifactError(f"Bảng {path.name} rỗng hoặc thiếu cột: {', '.join(sorted(missing))}")
    if name in ("overall", "daily"):
        numeric = ["n", "threshold", *COUNTS, *RATE_METRICS]
        try:
            table[numeric] = table[numeric].apply(pd.to_numeric, errors="raise")
        except ValueError as exc:
            raise ArtifactError(f"Giá trị metrics không hợp lệ: {path.name}") from exc
        if (table[list(COUNTS)].lt(0).any().any()
                or not table[list(COUNTS)].sum(axis=1).eq(table.n).all()):
            raise ArtifactError(f"Confusion counts không khớp số flow: {path.name}")
        key = ["day", "model"] if name == "daily" else ["model"]
        if table.duplicated(key).any():
            raise ArtifactError(f"Trùng nhóm metrics: {path.name}")
    return table


def temporal_rows(frame):
    return frame[frame.model.isin(["Isolation Forest", "Autoencoder", "Ensemble AE+IF"])].assign(
        model=lambda df: df.model.replace({"Ensemble AE+IF": "Ensemble"}))


def volume_by_day(daily, model):
    selected = temporal_rows(daily).query("model == @model").sort_values("day")
    return selected.assign(Benign=selected.TN + selected.FP,
                           Attack=selected.TP + selected.FN,
                           Alerts=selected.TP + selected.FP)


def filter_day(frame, day):
    return frame if day is None else frame[frame.day.eq(day)]
