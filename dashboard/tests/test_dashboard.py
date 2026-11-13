from pathlib import Path

import pandas as pd
import pytest

from dashboard.data import ArtifactError, Artifacts, read_table, volume_by_day


def test_loader_rejects_missing_and_non_csv(tmp_path):
    with pytest.raises(ArtifactError, match="Không đọc được"):
        read_table(tmp_path / "missing.csv", "daily")
    with pytest.raises(ArtifactError, match="chỉ đọc"):
        read_table(tmp_path / "predictions.parquet", "daily")


def test_loader_validates_schema_and_counts(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("day,model\n2018-02-21,Ensemble\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="thiếu cột"):
        read_table(path, "daily")
    pd.DataFrame([dict(day="2018-02-21", model="Ensemble AE+IF", n=10, threshold=.5,
                       TP=1, FP=2, TN=6, FN=1, precision=1/3, recall=.5, fpr=.25,
                       f1=.4, roc_auc=.8, pr_auc=.6)]).to_csv(path, index=False)
    selected = volume_by_day(read_table(path, "daily"), "Ensemble").iloc[0]
    assert (selected.Benign, selected.Attack, selected.Alerts) == (8, 2, 3)
    data = pd.read_csv(path)
    data.loc[0, "n"] = 11
    data.to_csv(path, index=False)
    with pytest.raises(ArtifactError, match="Confusion counts"):
        read_table(path, "daily")


def test_all_pages_and_filters_on_official_artifacts(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    store = Artifacts.from_environment()
    if not all(store.table(key).is_file() for key in ("daily", "overall", "attacks", "features", "scores", "reversal")):
        pytest.skip("Official research artifacts are not available")
    def forbidden(*args, **kwargs):
        pytest.fail("Dashboard attempted to read row-level Parquet")
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setenv("TRAFFIC_REPLAY_DB", str(tmp_path / "missing.sqlite"))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20)
    app.run()
    assert not app.exception and not app.error
    assert app.metric[0].value == "4,653,501"
    assert app.metric[3].value == "20,367"
    app.selectbox(key="day").set_value("2018-02-21").run()
    assert app.metric[0].value == "561,396"
    app.selectbox(key="model").set_value("Autoencoder").run()
    assert app.metric[3].value == "1,730"
    for page, title in (("Model Evaluation", "Confusion matrix"),
                        ("Attack Analysis", "Recall theo loại tấn công"),
                        ("Temporal Drift", "21/02/2018 · Ranking reversal")):
        app.radio(key="page").set_value(page).run()
        assert not app.exception and not app.error
        assert title in [item.value for item in app.subheader]
    app.radio(key="page").set_value("Live Detection").run()
    assert not app.exception and not app.error
    assert "Chưa kết nối Kafka" in app.info[0].value


def test_missing_artifacts_are_reported(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DASHBOARD_FULL_RUN", str(tmp_path))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    assert not app.exception
    assert "Thiếu artifact" in app.error[0].value
    app.radio(key="page").set_value("Live Detection").run()
    assert not app.exception and not app.error
