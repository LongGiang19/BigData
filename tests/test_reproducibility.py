import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import baselines
from src.test_snapshot import fixed_test_files


def test_snapshot_rebuild_reuse_and_source_identity(tmp_path):
    source = tmp_path / "Wednesday-14-02-2018.parquet"
    original = pa.table({"Label": [None, " Label "] + ["Benign", "Attack"] * 500,
                         "ordinal": list(range(1002))})
    pq.write_table(original, source, row_group_size=37)
    first, manifest = fixed_test_files([source], tmp_path / "snapshots", .25, 42)
    second, _ = fixed_test_files([source], tmp_path / "snapshots", .25, 42)
    rebuilt, _ = fixed_test_files([source], tmp_path / "rebuilt", .25, 42)
    assert first == second
    sample = pq.read_table(first[0])
    assert sample.equals(pq.read_table(rebuilt[0]))
    indices = sample["ordinal"].to_pylist()
    assert 0 < len(indices) < 1000 and min(indices) >= 2
    assert sample.drop(["record_id"]).equals(original.take(indices))
    ids = sample["record_id"].to_pylist()
    assert len(set(ids)) == len(indices)
    assert [int(value.rsplit(":", 1)[1]) for value in ids] == indices
    counts = json.loads(manifest.read_text())["counts"][0]
    assert (counts["source_rows"], counts["eligible_rows"], counts["sample_rows"]) == (1002, 1000, len(indices))
    full, full_manifest = fixed_test_files([source], tmp_path / "snapshots", 1.0, 42)
    assert full_manifest.parent != manifest.parent
    assert pq.read_table(full[0]).drop(["record_id"]).equals(original.slice(2))
    assert pq.read_table(first[0]).equals(sample)


def test_random_split_fits_only_train_and_calibrates_only_validation(monkeypatch):
    pdf = pd.DataFrame({"row": np.arange(200, dtype=float), "y": np.arange(200) % 2})
    pdf["label_raw"] = np.where(pdf.y, "Attack", "Benign")
    calls = {}

    class Spark:
        def createDataFrame(self, train):
            return train

    def fit_spec(train, feats):
        calls["feature_fit"] = train.row.astype(int).to_numpy()
        return feats, np.zeros(1), np.ones(1), None

    class Classifier:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, y):
            calls["model_fit"] = x.ravel().astype(int)
            np.testing.assert_array_equal(y, calls["model_fit"] % 2)
            return self

        def predict_proba(self, x):
            calls.setdefault("scored", []).append(x.ravel().astype(int))
            p = x.ravel() / len(pdf)
            return np.column_stack((1 - p, p))

    monkeypatch.setattr(baselines, "fit_feature_spec", fit_spec)
    monkeypatch.setattr(baselines, "to_z", lambda frame, cols, *_: frame[cols].to_numpy())
    monkeypatch.setattr(baselines, "HistGradientBoostingClassifier", Classifier)
    result = baselines.run_random_baseline(Spark(), pdf, ["row"])
    train, (val, test) = calls["feature_fit"], calls["scored"]
    np.testing.assert_array_equal(train, calls["model_fit"])
    assert set(train).isdisjoint(val) and set(train).isdisjoint(test) and set(val).isdisjoint(test)
    assert set(np.concatenate([train, val, test])) == set(range(200))
    assert result["split_rows"] == {"train": 140, "validation": 30, "test": 30}
    assert result["threshold"] == np.quantile(val[val % 2 == 0] / 200, 1 - baselines.cfg.FPR_TARGET)
