import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (invalid_value, quality_by_day, sanitize_features,
                               slog_np, to_z, valid_cond)
from src.spark_session import create_spark


@pytest.fixture(scope="module")
def spark():
    session = create_spark()
    session.conf.set("spark.sql.shuffle.partitions", "2")
    yield session
    session.stop()


@pytest.fixture
def dirty(spark):
    schema = "id int, day string, y int, flow double, init_fwd_win_bytes double, init_bwd_win_bytes double"
    rows = [
        (0, "2018-02-14", 0, float("nan"), -1.0, -2.0),
        (1, "2018-02-14", 0, None, None, 1.0),
        (2, "2018-02-14", 0, float("inf"), float("-inf"), 2.0),
        (3, "2018-02-14", 0, float("-inf"), 3.0, float("nan")),
        (4, "2018-02-14", 1, -5.0, -6.0, -7.0),
        (5, "2018-02-14", 1, 8.0, -9.0, -10.0),
        (6, "2018-02-16", 0, 0.0, 0.0, 0.0),
        (7, "2018-02-16", 1, 4.0, 5.0, 6.0),
    ]
    return spark.createDataFrame(spark.sparkContext.parallelize(rows, 1), schema)


FEATS = ["flow", "init_fwd_win_bytes", "init_bwd_win_bytes"]


@pytest.mark.parametrize("row_id", [0, 1, 2, 3, 4],
                         ids=["nan", "null", "positive_inf", "negative_inf", "negative"])
def test_invalid_cells_become_zero(dirty, row_id):
    row = sanitize_features(dirty, FEATS).filter(f"id = {row_id}").first()
    assert row.flow == 0.0
    if row_id in (1, 2):
        assert row.init_fwd_win_bytes == 0.0
    if row_id == 3:
        assert row.init_bwd_win_bytes == 0.0


def test_allowed_negatives_and_valid_values_unchanged(dirty):
    rows = sanitize_features(dirty, FEATS).orderBy("id").collect()
    assert (rows[0].init_fwd_win_bytes, rows[0].init_bwd_win_bytes) == (-1.0, -2.0)
    assert tuple(rows[5][c] for c in FEATS) == (8.0, -9.0, -10.0)
    assert tuple(rows[7][c] for c in FEATS) == (4.0, 5.0, 6.0)
    assert rows[0].day == "2018-02-14" and rows[0].y == 0


def test_row_count_and_validity(dirty):
    clean = sanitize_features(dirty, FEATS)
    assert clean.count() == dirty.count() == 8
    assert clean.filter(~valid_cond(FEATS)).count() == 0
    assert dirty.filter(invalid_value("flow")).count() == 5


def test_spark_numpy_consistency(dirty):
    raw = dirty.orderBy("id").toPandas()
    cleaned = sanitize_features(dirty, FEATS).orderBy("id").toPandas()
    mean = np.array([0.2, -0.3, 0.4])
    std = np.array([1.2, 2.0, 0.8])
    expected = np.clip((slog_np(cleaned[FEATS].to_numpy()) - mean) / std, -10, 10).astype(np.float32)
    np.testing.assert_array_equal(to_z(raw, FEATS, mean, std), expected)
    np.testing.assert_array_equal(to_z(cleaned, FEATS, mean, std), expected)
    assert pd.isna(raw.loc[0, "flow"])  # inference does not mutate input


def test_nullable_pandas_null():
    pdf = pd.DataFrame({"flow": pd.Series([pd.NA, -1, 2], dtype="Float64")})
    np.testing.assert_allclose(to_z(pdf, ["flow"], np.zeros(1), np.ones(1)).ravel(),
                               [0, 0, np.log1p(2)], rtol=1e-6)


def test_quality_report_counts_before_cleaning(dirty):
    report = quality_by_day(dirty, FEATS)
    assert report.columns == ["day", "y", "total_rows", "invalid_rows", "invalid_cells", "invalid_row_rate"]
    rows = {(r.day, r.y): r for r in report.collect()}
    for key, expected in {
        ("2018-02-14", 0): (4, 4, 7, 1.0),
        ("2018-02-14", 1): (2, 1, 1, 0.5),
        ("2018-02-16", 0): (1, 0, 0, 0.0),
        ("2018-02-16", 1): (1, 0, 0, 0.0),
    }.items():
        assert tuple(rows[key][c] for c in report.columns[2:]) == expected
    assert quality_by_day(sanitize_features(dirty, FEATS), FEATS).filter("invalid_cells != 0").count() == 0


def test_empty_report_and_no_features(dirty):
    assert quality_by_day(dirty.limit(0), FEATS).count() == 0
    assert quality_by_day(dirty, []).filter("invalid_cells != 0 OR invalid_rows != 0").count() == 0
