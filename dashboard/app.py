"""Run from the project root with streamlit run dashboard/app.py."""
from html import escape
import os
from pathlib import Path
import sqlite3
import sys

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard.data import (ArtifactError, Artifacts, COUNTS, MODELS, RATE_METRICS,
                            filter_day, read_table, temporal_rows, volume_by_day)
from streaming.common import STATE
from dashboard.live import alert_table, kafka_status, read_live, vietnam_time

PAGES = ("Overview", "Model Evaluation", "Attack Analysis", "Temporal Drift", "Live Detection")
COLORS = ["#2563eb", "#0f766e", "#7c3aed"]
METRIC_LABELS = dict(recall="Recall", fpr="FPR", precision="Precision", f1="F1",
                     roc_auc="ROC-AUC", pr_auc="PR-AUC")
ALL = "Tất cả 7 ngày"


@st.cache_data(show_spinner=False, max_entries=24)
def cached_table(path, name, modified):
    return read_table(path, name)


def load(store, name):
    path = store.table(name)
    try:
        modified = path.stat().st_mtime_ns
    except OSError as exc:
        raise ArtifactError(f"Thiếu artifact: {path}") from exc
    return cached_table(str(path), name, modified)


def chart(spec):
    st.altair_chart(spec.configure_view(stroke=None).configure_axis(
        labelFontSize=12, titleFontSize=12, gridOpacity=.18), width="stretch")


def media(path, caption):
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.caption(f"Ảnh chưa có tại {path.name}.")


def table(frame, rate_cols=()):
    display = frame.copy()
    for col in rate_cols:
        if col in display:
            display[col] = display[col].map(lambda v: f"{v:.4%}" if pd.notna(v) else "—")
    st.dataframe(display, hide_index=True, width="stretch")


def kpis(pairs):
    for col, (title, value) in zip(st.columns(len(pairs)), pairs):
        col.metric(title, value, border=True)


def overview(daily, day, model):
    selected = filter_day(volume_by_day(daily, model), day)
    totals = selected[["n", "Benign", "Attack", "Alerts"]].sum()
    kpis([(label, f"{totals[key]:,}") for label, key in
          [("Flow", "n"), ("Benign", "Benign"), ("Attack", "Attack"), ("Alerts", "Alerts")]])
    st.caption(f"Alerts của {model} gồm TP + FP tại ngưỡng gốc. Benign và Attack lấy từ nhãn dataset.")
    left, right = st.columns(2)
    with left:
        st.subheader("Lưu lượng theo ngày")
        volumes = selected.melt(id_vars="day", value_vars=["Benign", "Attack"],
                                var_name="Nhãn", value_name="Flow")
        chart(alt.Chart(volumes).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("day:O", title="Ngày test"), y="Flow:Q",
            color=alt.Color("Nhãn:N", scale=alt.Scale(domain=["Benign", "Attack"], range=["#2563eb", "#ea580c"])),
            tooltip=["day:O", "Nhãn:N", alt.Tooltip("Flow:Q", format=",")]).properties(height=290))
    with right:
        st.subheader("Cảnh báo theo ngày")
        alerts = selected[["day", "TP", "FP"]].melt(
            id_vars="day", value_vars=["TP", "FP"], var_name="Kết quả", value_name="Alerts")
        chart(alt.Chart(alerts).mark_bar().encode(
            x=alt.X("day:O", title="Ngày test"), y="Alerts:Q",
            color=alt.Color("Kết quả:N", scale=alt.Scale(domain=["TP", "FP"], range=["#0f766e", "#ea580c"])),
            tooltip=["day:O", "Kết quả:N", alt.Tooltip("Alerts:Q", format=",")]).properties(height=290))
    st.subheader("Thiết kế đánh giá")
    st.markdown("**Train** 14–15/02, Benign · **Calibration** 16/02, Benign · **Test** 7 ngày sau, snapshot 100%")
    table(selected[["day", "n", "Benign", "Attack", "Alerts", "TP", "FP"]])


def evaluation(store, daily, day, model):
    frame = temporal_rows(load(store, "overall") if day is None else filter_day(daily, day))
    row = frame[frame.model.eq(model)].iloc[0]
    kpis([(METRIC_LABELS[key], f"{row[key]:.4%}") for key in ("recall", "fpr", "precision", "f1")])
    st.caption(f"{model} · ROC-AUC {row.roc_auc:.5f} · PR-AUC {row.pr_auc:.5f} · Ngưỡng đã lưu {row.threshold:.8g}")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("So sánh ba mô hình")
        long = frame.melt(id_vars="model", value_vars=list(RATE_METRICS), var_name="metric", value_name="value")
        long["metric"] = long.metric.map(METRIC_LABELS)
        chart(alt.Chart(long).mark_bar().encode(
            x=alt.X("metric:N", title=None, sort=list(METRIC_LABELS.values())),
            xOffset="model:N", y=alt.Y("value:Q", title="Giá trị", axis=alt.Axis(format=".0%")),
            color=alt.Color("model:N", title="Mô hình", scale=alt.Scale(domain=list(MODELS), range=COLORS)),
            tooltip=["model:N", "metric:N", alt.Tooltip("value:Q", format=".4%")]).properties(height=290))
    with right:
        st.subheader("Confusion matrix")
        table(pd.DataFrame({"Nhãn thực tế": ["Benign", "Attack"],
                            "Dự đoán Benign": [int(row.TN), int(row.FN)],
                            "Dự đoán Attack": [int(row.FP), int(row.TP)]}))
        st.caption("Hàng là nhãn thực tế, cột là dự đoán tại ngưỡng gốc.")
    table(frame[["model", *COUNTS, *RATE_METRICS]], RATE_METRICS)
    with st.expander("Temporal test và random split"):
        st.markdown("**Temporal test** đánh giá ba mô hình trên các ngày sau train và calibration. "
                    "**HistGradientBoosting random split** là đối chứng supervised trộn ngày, có test riêng.")
        ref = load(store, "overall")
        ref = ref[ref.model.str.contains("random-split", regex=False)]
        table(ref[["model", "n", *RATE_METRICS]], RATE_METRICS)
        st.caption("Random split có thể chứa cùng loại attack ở các tập và không đại diện cho tấn công chưa từng thấy.")


def attack_analysis(store, day, model):
    data = load(store, "attacks")
    data = data[data.day.eq(day or "ALL") & data.model.eq(MODELS[model])]
    focus = r"HOIC|LOIC-HTTP|LOIC-UDP|^Bot$|Infilteration"
    focus_only = st.toggle("Chỉ hiện nhóm trọng tâm", value=True)
    if focus_only:
        data = data[data.attack.str.contains(focus, case=False, regex=True)]
    if data.empty:
        st.info("Ngày đã chọn không có nhóm trọng tâm. Tắt bộ lọc để xem các nhãn attack khác.")
        return
    n, tp = int(data.n.sum()), int(data.above_threshold.sum())
    kpis([("Attack trong nhóm đang xem", f"{n:,}"), ("Phát hiện", f"{tp:,}"),
          ("Bỏ sót", f"{n-tp:,}"), ("Recall theo số flow", f"{tp/n:.4%}")])
    st.subheader("Recall theo loại tấn công")
    chart(alt.Chart(data).mark_bar(color=COLORS[list(MODELS).index(model)]).encode(
        y=alt.Y("attack:N", title=None, sort="-x"),
        x=alt.X("alert_rate:Q", title="Recall", axis=alt.Axis(format=".1%")),
        tooltip=["attack:N", alt.Tooltip("n:Q", title="Số flow", format=","),
                 alt.Tooltip("above_threshold:Q", title="TP", format=","),
                 alt.Tooltip("alert_rate:Q", title="Recall", format=".4%")]).properties(height=290))
    table(data[["attack", "n", "above_threshold", "alert_rate"]].rename(
        columns={"above_threshold": "TP", "alert_rate": "Recall"}), ["Recall"])
    st.caption("Các dòng có Recall 0 vẫn xuất hiện trong bảng. Đọc Recall cùng số flow để phân biệt nhóm hiếm.")


def drift(store, daily, day, model):
    feature_tab, score_tab, reversal_tab = st.tabs(["Feature drift", "Score & FPR", "Ranking reversal"])
    with feature_tab:
        reference = st.radio("Tham chiếu Benign", ["Train 14–15/02", "Validation 16/02"], horizontal=True)
        metric = st.selectbox("Phép đo", ["KS", "PSI", "Wasserstein"]).lower()
        features = load(store, "features")
        ref = "train" if reference.startswith("Train") else "2018-02-16"
        data = filter_day(features[features.reference.eq(ref) & features.day.isin(daily.day.unique())], day)
        order = data.groupby("feature")[metric].median().nlargest(12).index.tolist()
        chart(alt.Chart(data[data.feature.isin(order)]).mark_rect().encode(
            x=alt.X("day:O", title="Ngày test"), y=alt.Y("feature:N", sort=order, title=None),
            color=alt.Color(f"{metric}:Q", scale=alt.Scale(scheme="viridis")),
            tooltip=["day:O", "feature:N", alt.Tooltip(f"{metric}:Q", format=".5f"),
                     alt.Tooltip("recon_mse:Q", format=".5f")]).properties(height=340))
        st.caption("46 feature đầu vào dùng chung cho ba mô hình. Heatmap hiển thị 12 feature có drift lớn nhất trong phạm vi đang xem.")
        with st.expander("Phân phối feature và phương pháp lấy mẫu"):
            media(store.research / "temporal_drift/feature_distributions.png",
                  "Biểu đồ nghiên cứu đã lưu. Giữ nguyên các ngày tham chiếu của phân tích gốc.")
    with score_tab:
        scores = load(store, "scores")
        values = filter_day(scores[scores.reference.eq("2018-02-16") & scores.model.eq(MODELS[model])], day)
        base = alt.Chart(values).encode(x=alt.X("day:O", title="Ngày test"),
                                       color=alt.Color("label:N", title="Nhãn", scale=alt.Scale(
                                           domain=["Benign", "Attack"], range=["#2563eb", "#ea580c"])))
        band = base.mark_area(opacity=.16).encode(y=alt.Y("p10:Q", title="Anomaly score"), y2="p90:Q")
        line = base.mark_line(point=True).encode(y="median:Q", tooltip=[
            "day:O", "label:N", alt.Tooltip("median:Q", format=".5f"),
            alt.Tooltip("p99:Q", format=".5f"), alt.Tooltip("ks:Q", format=".4f")])
        rule = alt.Chart(values[["threshold"]].drop_duplicates()).mark_rule(
            color="#111827", strokeDash=[5, 4]).encode(y="threshold:Q")
        st.subheader("Phân phối score · median và p10–p90")
        chart((band + line + rule).properties(height=285))
        st.caption("Đường nét đứt là ngưỡng đã lưu. Tooltip có p99 và KS so với Benign Validation.")
        selected = filter_day(temporal_rows(daily).query("model == @model"), day)
        st.subheader("FPR thực tế theo ngày")
        chart(alt.Chart(selected).mark_line(point=True, color=COLORS[list(MODELS).index(model)]).encode(
            x=alt.X("day:O", title="Ngày test"), y=alt.Y("fpr:Q", title="FPR", axis=alt.Axis(format=".1%")),
            tooltip=["day:O", alt.Tooltip("fpr:Q", format=".4%"), alt.Tooltip("FP:Q", format=",")])
              .properties(height=220))
        st.caption("Distribution shift và FPR thay đổi là kết quả mô tả. Audit chưa xác nhận concept drift.")
    with reversal_tab:
        st.subheader("21/02/2018 · Ranking reversal")
        st.caption("Audit riêng ngày 21/02. Bộ lọc ngày phía trên áp dụng cho feature, score và FPR.")
        rev = load(store, "reversal").replace({"model": {"Ensemble AE+IF": "Ensemble"}})
        row = rev[rev.model.eq(model)]
        table(row[["model", "benign", "attack", "prevalence", "roc_auc", "roc_auc_negated", "pr_auc"]],
              ["prevalence"])
        st.markdown("HOIC chiếm **99,14%** flow Attack ngày này và đóng góp gần như toàn bộ phần xếp sai có trọng số. "
                    "AUC đảo dấu là đối chiếu hậu nghiệm. Dashboard hiển thị score và ngưỡng gốc.")
        media(store.research / "ranking_reversal_audit/score_distributions.png", "CDF đã kiểm chứng cho cả ba mô hình.")


@st.cache_data(ttl=20, show_spinner=False, max_entries=4)
def cached_kafka_status(address):
    import time
    status, explanation = kafka_status(address)
    return status, explanation, time.time_ns() // 1_000_000


@st.fragment(run_every="3s")
def live_detection():
    db = Path(os.getenv("TRAFFIC_REPLAY_DB", STATE / "predictions.sqlite"))
    st.caption("Demo replay Parquet → Kafka → Spark → SQLite · cập nhật mỗi 3 giây")
    data = None
    try:
        if db.is_file():
            data = read_live(db)
    except (sqlite3.Error, OSError, TypeError, ValueError) as exc:
        st.error(f"SQLite Database: Error · Không đọc được prediction store: {exc}")
    st.subheader("System Status")
    status, explanation, checked_ms = cached_kafka_status(data["broker"] if data else None)
    with st.container(border=True):
        columns = st.columns(4)
        columns[0].markdown(f"**Kafka Broker**\n\n{status}")
        columns[1].markdown("**Spark Consumer**\n\nUnknown")
        columns[2].markdown(f"**SQLite Database**\n\n{'Connected' if data else 'Error'}")
        columns[3].markdown("**Last Processed Event**\n\n" + vietnam_time(data["last_ms"] if data else None))
        st.caption(f"{explanation} Kiểm tra lúc {vietnam_time(checked_ms)} · cache tối đa 20 giây. "
                   "Spark chưa có heartbeat độc lập nên trạng thái là Unknown, kể cả khi SQLite còn dữ liệu.")
    if data is None:
        if not db.is_file():
            st.info("Chưa kết nối Kafka qua replay consumer · đang chờ prediction store.")
        return
    descriptions = [
        ("Total Processed Flows", f'{data["flows"]:,}', "Tổng ID duy nhất đã ghi vào SQLite."),
        ("Total Alerts", f'{data["alerts"]:,}', "Flow có score vượt threshold Ensemble đã lưu."),
        ("Alert Rate", f'{data["alert_rate"]:.2%}', "Alerts / tổng flow. Đây không phải Recall hoặc FPR."),
        ("Processing Rate", f'{data["rate"]:.2f}', "Flow mới ghi trong 60 giây gần nhất / 60, đơn vị flow/giây."),
    ]
    for column, (label, value, help_text) in zip(st.columns(4), descriptions):
        column.metric(label, value, help=help_text, border=True)
    st.caption("Alerts là cảnh báo của mô hình; số tấn công thực tế cần nhãn kiểm chứng. "
               "Thời gian hiển thị theo Việt Nam (UTC+7).")
    if data["flows"] == 0:
        st.info("Chưa nhận flow. Hãy kiểm tra producer, broker và consumer trong các terminal tương ứng.")
    elif data["rate"] == 0:
        st.info("Chưa có flow mới trong 60 giây gần nhất. Consumer có thể đang chờ hoặc tạm dừng; "
                "trạng thái tiến trình hiện chưa xác định.")
    trend = data["timeline"]
    left, right = st.columns(2)
    x = alt.X("Time:O", title="Giờ Việt Nam · 60 phút gần nhất",
              sort=trend.Time.tolist(), axis=alt.Axis(labelAngle=-45, labelOverlap=True, tickCount=8))
    with left:
        st.subheader("Alerts per Minute")
        chart(alt.Chart(trend).mark_bar(color="#d97706").encode(
            x=x, y=alt.Y("Alerts:Q", title="Alerts", axis=alt.Axis(tickMinStep=1)),
            tooltip=["Time:O", "Alerts:Q"]).properties(height=240))
        if trend.Alerts.sum() == 0:
            st.caption("Không có cảnh báo trong 60 phút gần nhất. Tổng alerts phía trên là số tích lũy.")
    with right:
        st.subheader("Processed Flows vs Alerts")
        values = trend.melt(id_vars="Time", value_vars=["Processed Flows", "Alerts"],
                            var_name="Series", value_name="Flows")
        chart(alt.Chart(values).mark_line().encode(
            x=x, y=alt.Y("Flows:Q", title="Flow / phút"),
            color=alt.Color("Series:N", title=None, scale=alt.Scale(
                domain=["Processed Flows", "Alerts"], range=["#2563eb", "#d97706"])),
            tooltip=["Time:O", "Series:N", "Flows:Q"]).properties(height=240))
    st.subheader("Recent Alerts")
    latest = data["latest"]
    st.caption("25 cảnh báo mới nhất, chưa phải tấn công đã được xác nhận.")
    if not latest.empty:
        st.dataframe(alert_table(latest), hide_index=True, width="stretch", column_config={
            "Anomaly Score": st.column_config.NumberColumn(format="%.4f"),
            "Event ID": st.column_config.TextColumn(width="small"),
            "Top Feature": st.column_config.TextColumn(width="small"),
        })
        with st.expander("Xem / sao chép Event ID đầy đủ"):
            selected = st.selectbox("Chọn cảnh báo", range(len(latest)), key="live_alert_id",
                format_func=lambda i: f'{i+1}. {vietnam_time(latest.iloc[i].processed_ms)} · …{latest.iloc[i].event_id[-12:]}')
            st.code(latest.iloc[selected].event_id, language=None, wrap_lines=True)
            st.download_button("Tải 25 cảnh báo với ID đầy đủ", latest.to_csv(index=False).encode("utf-8"),
                               "recent_alerts.csv", "text/csv")
    else:
        st.caption("Chưa có cảnh báo được lưu.")


def main():
    st.set_page_config(page_title="Traffic Lab · Network Anomaly Detection", page_icon="◈", layout="wide")
    st.markdown("""<style>
    .block-container {max-width: 1420px; padding-top: 2rem;}
    .traffic-hero {background: linear-gradient(115deg,#10233e,#163e5c); color: #f8fafc;
        padding: 1.7rem 2rem; border-radius: 16px; margin-bottom: 1.2rem;}
    .traffic-hero small {color: #a5f3fc; font-weight: 700; letter-spacing: .14em;}
    .traffic-hero h1 {color: #fff; margin: .3rem 0; padding: 0; font-size: 2.3rem;}
    .traffic-hero p {color: #d6e6f2; margin: .3rem 0 0;}
    </style>""", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### ◈ Traffic Lab")
        st.caption("CSE-CIC-IDS2018 · Offline research")
        page = st.radio("Trang", PAGES, key="page")
    descriptions = {
        "Overview": "Quy mô dữ liệu và cảnh báo trên snapshot test chính thức.",
        "Model Evaluation": "Hiệu năng tại ngưỡng đã hiệu chuẩn trên Benign Validation.",
        "Attack Analysis": "Độ phủ phát hiện theo loại attack và số lượng flow.",
        "Temporal Drift": "Phân phối feature, score và hiệu năng qua từng ngày.",
        "Live Detection": "Không gian dành cho luồng phát hiện trực tuyến.",
    }
    st.markdown(f'<div class="traffic-hero"><small>NETWORK ANOMALY DETECTION</small>'
                f'<h1>{escape(page)}</h1><p>{escape(descriptions[page])}</p></div>', unsafe_allow_html=True)
    if page == "Live Detection":
        live_detection()
        return
    store = Artifacts.from_environment()
    try:
        daily = load(store, "daily")
        with st.sidebar:
            st.divider()
            model = st.selectbox("Mô hình", list(MODELS), index=2, key="model")
            choice = st.selectbox("Ngày test", [ALL, *sorted(daily.day.unique())], key="day")
            st.caption(f"Full test 100% · {store.run.name}")
        day = None if choice == ALL else choice
        if page == "Overview":
            overview(daily, day, model)
        elif page == "Model Evaluation":
            evaluation(store, daily, day, model)
        elif page == "Attack Analysis":
            attack_analysis(store, day, model)
        else:
            drift(store, daily, day, model)
    except ArtifactError as exc:
        st.error(str(exc))
        st.caption("Kiểm tra DASHBOARD_FULL_RUN và DASHBOARD_RESEARCH_DIR theo dashboard/README.md.")


if __name__ == "__main__":
    main()
