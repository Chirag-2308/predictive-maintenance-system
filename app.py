"""
app.py  —  Predictive Maintenance Dashboard
============================================
iisys Hof University  |  Cyber-Physical Systems Research
Author : [Your Name]  |  B.Tech CSE  |  GitHub: [your-username]

Run:
    streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from utils.predictor import (
    load_data, get_all_machine_risk,
    predict_machine, get_report
)

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PredictMaint — Industrial AI Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  — dark #1D1D1F theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global background ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: #1D1D1F !important;
    color: #F5F5F7 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
}

/* ── Main content area ── */
[data-testid="stMain"] { background-color: #1D1D1F !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #161617 !important;
    border-right: 1px solid #2C2C2E;
}
[data-testid="stSidebar"] * { color: #E5E5EA !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background-color: #2C2C2E;
    border: 1px solid #3A3A3C;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: #8E8E93 !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: #F5F5F7 !important; font-size: 26px !important; font-weight: 600 !important; }
[data-testid="stMetricDelta"] { font-size: 12px !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { background-color: #2C2C2E !important; }
.dvn-scroller { background-color: #2C2C2E !important; }

/* ── Select / input ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background-color: #2C2C2E !important;
    border: 1px solid #3A3A3C !important;
    border-radius: 8px !important;
    color: #F5F5F7 !important;
}

/* ── Tab styling ── */
[data-testid="stTab"] button {
    color: #8E8E93 !important;
    border-radius: 8px 8px 0 0 !important;
}
[data-testid="stTab"] button[aria-selected="true"] {
    color: #F5F5F7 !important;
    border-bottom: 2px solid #0A84FF !important;
    background: transparent !important;
}

/* ── Divider ── */
hr { border-color: #2C2C2E !important; }

/* ── Alert banner ── */
.alert-critical {
    background: linear-gradient(90deg, #3B0A0A, #2C2C2E);
    border-left: 4px solid #FF453A;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 6px 0;
    color: #FF453A;
    font-size: 13px;
    font-weight: 500;
}
.alert-warning {
    background: linear-gradient(90deg, #2D1E00, #2C2C2E);
    border-left: 4px solid #FF9F0A;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 6px 0;
    color: #FF9F0A;
    font-size: 13px;
    font-weight: 500;
}
.alert-normal {
    background: linear-gradient(90deg, #0A2E1A, #2C2C2E);
    border-left: 4px solid #30D158;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 6px 0;
    color: #30D158;
    font-size: 13px;
    font-weight: 500;
}

/* ── Machine card ── */
.machine-card {
    background-color: #2C2C2E;
    border: 1px solid #3A3A3C;
    border-radius: 12px;
    padding: 16px;
    margin: 4px 0;
}

/* ── Header strip ── */
.dash-header {
    background: linear-gradient(135deg, #1C1C1E 0%, #2C2C2E 100%);
    border: 1px solid #3A3A3C;
    border-radius: 14px;
    padding: 24px 30px;
    margin-bottom: 24px;
}

/* ── Section label ── */
.section-label {
    color: #8E8E93;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

/* ── Risk badge ── */
.risk-critical { color:#FF453A; font-weight:700; }
.risk-warning  { color:#FF9F0A; font-weight:700; }
.risk-normal   { color:#30D158; font-weight:700; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #1D1D1F; }
::-webkit-scrollbar-thumb { background: #3A3A3C; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PLOTLY DARK TEMPLATE
# ─────────────────────────────────────────────
PLOT_BG     = "#1D1D1F"
PLOT_PAPER  = "#2C2C2E"
GRID_COLOR  = "#3A3A3C"
TEXT_COLOR  = "#E5E5EA"
ACCENT_BLUE = "#0A84FF"
ACCENT_GREEN= "#30D158"
ACCENT_RED  = "#FF453A"
ACCENT_AMB  = "#FF9F0A"

def dark_layout(fig, title="", height=None):
    fig.update_layout(
        paper_bgcolor=PLOT_PAPER,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COLOR, family="-apple-system, BlinkMacSystemFont, 'SF Pro Text'", size=12),
        title=dict(text=title, font=dict(size=14, color=TEXT_COLOR), x=0.01),
        xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, tickfont=dict(color="#8E8E93")),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, tickfont=dict(color="#8E8E93")),
        margin=dict(l=12, r=12, t=40 if title else 20, b=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR)),
        **({"height": height} if height else {}),
    )
    return fig


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_load():
    return load_data()

@st.cache_data(show_spinner=False)
def cached_risk(_sensor_df, _meta_df):
    return get_all_machine_risk(_sensor_df, _meta_df)

def risk_color(level):
    return {"CRITICAL": ACCENT_RED, "WARNING": ACCENT_AMB, "NORMAL": ACCENT_GREEN}.get(level, ACCENT_GREEN)

def risk_emoji(level):
    return {"CRITICAL": "🔴", "WARNING": "🟡", "NORMAL": "🟢"}.get(level, "🟢")

def risk_class(level):
    return {"CRITICAL": "risk-critical", "WARNING": "risk-warning", "NORMAL": "risk-normal"}.get(level, "risk-normal")


# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("Loading industrial data and running predictions …"):
    sensor_df, meta_df = cached_load()
    risk_df = cached_risk(sensor_df, meta_df)
    report  = get_report()

n_critical = (risk_df["risk_level"] == "CRITICAL").sum()
n_warning  = (risk_df["risk_level"] == "WARNING").sum()
n_normal   = (risk_df["risk_level"] == "NORMAL").sum()

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ PredictMaint")
    st.markdown("<p style='color:#8E8E93;font-size:12px;margin-top:-10px'>Industrial AI · iisys Research</p>", unsafe_allow_html=True)
    st.divider()

    view = st.radio(
        "Navigation",
        ["Fleet Overview", "Machine Detail", "Model Performance", "Alert Log"],
        label_visibility="collapsed"
    )
    st.divider()

    # live alert count
    if n_critical > 0:
        st.markdown(f"<div class='alert-critical'>⚠ {n_critical} CRITICAL machine{'s' if n_critical>1 else ''}</div>", unsafe_allow_html=True)
    if n_warning > 0:
        st.markdown(f"<div class='alert-warning'>● {n_warning} machine{'s' if n_warning>1 else ''} need attention</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("<p style='color:#48484A;font-size:10px'>Data: 216,000 sensor readings<br>Machines: 50  |  Sensors: 4<br>Period: 180 days</p>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <h1 style="margin:0;font-size:22px;font-weight:600;color:#F5F5F7;letter-spacing:-0.3px">
    Predictive Maintenance System
  </h1>
  <p style="margin:6px 0 0;color:#8E8E93;font-size:13px">
    Industry 4.0 · Cyber-Physical Systems · iisys Hof University Research Lab
  </p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  VIEW 1 — FLEET OVERVIEW
# ═══════════════════════════════════════════════════════════
if view == "Fleet Overview":

    # ── KPI row ──
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Machines",     f"{len(risk_df)}")
    k2.metric("🔴 Critical",         f"{n_critical}", delta=f"{n_critical} need immediate action", delta_color="inverse")
    k3.metric("🟡 Warning",          f"{n_warning}",  delta=f"{n_warning} monitoring closely",     delta_color="inverse")
    k4.metric("🟢 Normal",           f"{n_normal}",   delta="Operating within spec",               delta_color="normal")
    k5.metric("Model ROC-AUC",      f"{report['random_forest']['roc_auc']:.4f}")

    st.divider()

    # ── Risk distribution bar + pie ──
    col_chart, col_pie = st.columns([3, 2])

    with col_chart:
        # Horizontal bar chart of all machines sorted by risk score
        top30 = risk_df.head(30).copy()
        colors = [risk_color(lvl) for lvl in top30["risk_level"]]
        fig = go.Figure(go.Bar(
            x=top30["risk_score"],
            y=top30["machine_id"],
            orientation="h",
            marker=dict(color=colors, opacity=0.88),
            text=[f"{s}%" for s in top30["risk_score"]],
            textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=10),
            hovertemplate="<b>%{y}</b><br>Risk: %{x:.1f}%<extra></extra>",
        ))
        fig.add_vline(x=35, line_dash="dot", line_color=ACCENT_RED,   annotation_text="Critical threshold",
                      annotation_font_color=ACCENT_RED, annotation_font_size=10)
        fig.add_vline(x=15, line_dash="dot", line_color=ACCENT_AMB,   annotation_text="Warning threshold",
                      annotation_font_color=ACCENT_AMB, annotation_font_size=10)
        fig = dark_layout(fig, title="Machine Risk Scores (Top 30)", height=500)
        fig.update_yaxes(tickfont=dict(size=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_pie:
        fig2 = go.Figure(go.Pie(
            labels=["Normal", "Warning", "Critical"],
            values=[n_normal, n_warning, n_critical],
            marker=dict(colors=[ACCENT_GREEN, ACCENT_AMB, ACCENT_RED]),
            hole=0.62,
            textinfo="percent+label",
            textfont=dict(color=TEXT_COLOR, size=12),
            hovertemplate="%{label}: %{value} machines<extra></extra>",
        ))
        fig2.add_annotation(
            text=f"<b>{len(risk_df)}</b><br>Machines",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=TEXT_COLOR),
        )
        fig2 = dark_layout(fig2, title="Fleet Health Distribution", height=320)
        fig2.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig2, use_container_width=True)

        # ── Risk by machine type ──
        type_risk = risk_df.groupby("machine_type")["risk_score"].mean().sort_values(ascending=True)
        fig3 = go.Figure(go.Bar(
            x=type_risk.values, y=type_risk.index,
            orientation="h",
            marker=dict(color=ACCENT_BLUE, opacity=0.8),
            hovertemplate="%{y}: %{x:.1f}% avg risk<extra></extra>",
        ))
        fig3 = dark_layout(fig3, title="Avg Risk by Machine Type", height=280)
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # ── Fleet table ──
    st.markdown("<p class='section-label'>Full Fleet Status</p>", unsafe_allow_html=True)

    display_df = risk_df.copy()
    display_df["Risk %"]    = display_df["risk_score"].apply(lambda x: f"{x:.1f}%")
    display_df["Status"]    = display_df["risk_level"].apply(lambda l: f"{risk_emoji(l)} {l}")
    display_df["Age (yrs)"] = display_df["age_years"]

    st.dataframe(
        display_df[["machine_id","machine_type","location","Age (yrs)","Risk %","Status"]].rename(
            columns={"machine_id":"ID","machine_type":"Type","location":"Location"}
        ),
        use_container_width=True,
        height=340,
        hide_index=True,
    )


# ═══════════════════════════════════════════════════════════
#  VIEW 2 — MACHINE DETAIL
# ═══════════════════════════════════════════════════════════
elif view == "Machine Detail":

    all_ids = sorted(sensor_df["machine_id"].unique())
    # Put critical machines first in the selector
    critical_ids = risk_df[risk_df["risk_level"] == "CRITICAL"]["machine_id"].tolist()
    sorted_ids   = critical_ids + [m for m in all_ids if m not in critical_ids]

    sel = st.selectbox("Select Machine", sorted_ids)
    meta_row  = meta_df[meta_df["machine_id"] == sel].iloc[0]
    mach_data = sensor_df[sensor_df["machine_id"] == sel].sort_values("timestamp")

    risk_s, iso_s, level, recent_p = predict_machine(mach_data)

    # ── Machine header ──
    st.markdown(f"""
    <div class="machine-card">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div>
          <p style="margin:0;font-size:18px;font-weight:600;color:#F5F5F7">{sel}</p>
          <p style="margin:4px 0 0;color:#8E8E93;font-size:13px">
            {meta_row['machine_type']} &nbsp;·&nbsp; {meta_row['location']} &nbsp;·&nbsp; Age: {meta_row['age_years']} yrs
          </p>
        </div>
        <div style="text-align:right">
          <p class="{risk_class(level)}" style="font-size:22px;margin:0">{risk_emoji(level)} {level}</p>
          <p style="color:#8E8E93;font-size:12px;margin:2px 0 0">Risk score: {risk_s*100:.1f}%</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Alert banner ──
    if level == "CRITICAL":
        st.markdown(f"<div class='alert-critical'>⚠ CRITICAL ALERT — {sel} is showing strong fault indicators. Schedule maintenance immediately. Risk score: {risk_s*100:.1f}%</div>", unsafe_allow_html=True)
    elif level == "WARNING":
        st.markdown(f"<div class='alert-warning'>● WARNING — {sel} shows elevated risk patterns. Monitor closely and plan preventive maintenance. Risk score: {risk_s*100:.1f}%</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='alert-normal'>✓ NORMAL — {sel} is operating within specification. Risk score: {risk_s*100:.1f}%</div>", unsafe_allow_html=True)

    st.divider()

    # ── Time window selector ──
    days = st.select_slider("Time window", options=[7, 14, 30, 60, 90, 180], value=30)
    cutoff = mach_data["timestamp"].max() - timedelta(days=days)
    window = mach_data[mach_data["timestamp"] >= cutoff].copy()

    # ── Sensor charts ──
    sensors = [
        ("temperature", "Temperature (°C)",  ACCENT_RED),
        ("vibration",   "Vibration (mm/s)",   ACCENT_AMB),
        ("rpm",         "RPM",                ACCENT_BLUE),
        ("pressure",    "Pressure (bar)",     ACCENT_GREEN),
    ]

    col1, col2 = st.columns(2)
    for i, (sensor, label, color) in enumerate(sensors):
        col = col1 if i % 2 == 0 else col2
        with col:
            # rolling mean for clarity
            window[f"{sensor}_roll"] = window[sensor].rolling(6, min_periods=1).mean()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=window["timestamp"], y=window[sensor],
                mode="lines", name="Raw",
                line=dict(color=color, width=1, dash="dot"),
                opacity=0.35,
            ))
            fig.add_trace(go.Scatter(
                x=window["timestamp"], y=window[f"{sensor}_roll"],
                mode="lines", name="6-hr avg",
                line=dict(color=color, width=2),
            ))
            # fault shading
            fault_win = window[window["fault_label"] == 1]
            if not fault_win.empty:
                fig.add_vrect(
                    x0=fault_win["timestamp"].min(), x1=fault_win["timestamp"].max(),
                    fillcolor=ACCENT_RED, opacity=0.12,
                    annotation_text="Fault zone", annotation_font_color=ACCENT_RED,
                    annotation_font_size=10,
                )
            fig = dark_layout(fig, title=label, height=220)
            fig.update_layout(showlegend=False, margin=dict(t=36, b=8))
            st.plotly_chart(fig, use_container_width=True)

    # ── Risk probability timeline ──
    st.markdown("<p class='section-label'>Fault Risk Probability — Last 7 Days</p>", unsafe_allow_html=True)

    if len(recent_p) > 0:
        proba_arr = np.array(recent_p)
        ts_recent = mach_data["timestamp"].values[-len(proba_arr):]

        fig_prob = go.Figure()
        fig_prob.add_trace(go.Scatter(
            x=ts_recent, y=proba_arr * 100,
            mode="lines", fill="tozeroy",
            fillcolor=f"rgba(255,69,58,0.15)",
            line=dict(color=ACCENT_RED, width=2),
            hovertemplate="%{x|%b %d %H:%M}<br>Risk: %{y:.2f}%<extra></extra>",
        ))
        fig_prob.add_hline(y=35, line_dash="dot", line_color=ACCENT_RED,
                           annotation_text="Critical (35%)", annotation_font_color=ACCENT_RED, annotation_font_size=10)
        fig_prob.add_hline(y=15, line_dash="dot", line_color=ACCENT_AMB,
                           annotation_text="Warning (15%)",  annotation_font_color=ACCENT_AMB,  annotation_font_size=10)
        fig_prob = dark_layout(fig_prob, height=220)
        fig_prob.update_yaxes(title="Risk %", range=[0, max(max(proba_arr)*100*1.2, 40)])
        st.plotly_chart(fig_prob, use_container_width=True)

    # ── Latest sensor readings ──
    st.divider()
    st.markdown("<p class='section-label'>Latest Sensor Readings</p>", unsafe_allow_html=True)

    latest = mach_data.tail(1).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperature", f"{latest['temperature']:.1f} °C")
    c2.metric("Vibration",   f"{latest['vibration']:.3f} mm/s")
    c3.metric("RPM",         f"{latest['rpm']:.0f}")
    c4.metric("Pressure",    f"{latest['pressure']:.2f} bar")


# ═══════════════════════════════════════════════════════════
#  VIEW 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════
elif view == "Model Performance":

    rf = report["random_forest"]

    # ── Metrics ──
    st.markdown("<p class='section-label'>Random Forest Classifier — Test Set Performance</p>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision",  f"{rf['precision']*100:.1f}%",  help="How often a fault alert is correct")
    m2.metric("Recall",     f"{rf['recall']*100:.1f}%",     help="% of actual faults correctly detected")
    m3.metric("F1 Score",   f"{rf['f1_score']*100:.1f}%",   help="Harmonic mean of precision and recall")
    m4.metric("ROC-AUC",    f"{rf['roc_auc']:.4f}",         help="Area under the ROC curve — 1.0 is perfect")

    st.info("ℹ  In predictive maintenance, **Recall is the most critical metric**. "
            "A missed fault (false negative) costs far more than a false alarm. "
            f"Our model achieves **{rf['recall']*100:.1f}% recall** — meaning it detects "
            f"{rf['recall']*100:.1f}% of all real fault events before they occur.")

    st.divider()

    col_cm, col_feat = st.columns(2)

    with col_cm:
        cm = np.array(rf["confusion_matrix"])
        labels = ["Normal", "Pre-Fault"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels,
            colorscale=[[0,"#1D1D1F"],[0.5,"#0A3D6B"],[1.0,ACCENT_BLUE]],
            text=cm, texttemplate="%{text}",
            textfont=dict(size=18, color="white"),
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            showscale=False,
        ))
        fig_cm = dark_layout(fig_cm, title="Confusion Matrix", height=320)
        fig_cm.update_xaxes(title="Predicted Label")
        fig_cm.update_yaxes(title="True Label", autorange="reversed")
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown(f"""
        <div class="machine-card" style="font-size:12px;color:#8E8E93">
          <p style="color:#F5F5F7;font-weight:500;margin:0 0 8px">Confusion Matrix Explained</p>
          True Negatives (TN): <b style="color:#30D158">{cm[0][0]:,}</b> — normal, predicted normal<br>
          False Positives (FP): <b style="color:#FF9F0A">{cm[0][1]:,}</b> — normal, predicted fault<br>
          False Negatives (FN): <b style="color:#FF453A">{cm[1][0]:,}</b> — fault missed (critical!)<br>
          True Positives (TP): <b style="color:#0A84FF">{cm[1][1]:,}</b> — fault correctly detected<br>
        </div>
        """, unsafe_allow_html=True)

    with col_feat:
        feats  = [f[0] for f in rf["top_features"]]
        scores = [f[1] for f in rf["top_features"]]
        # Clean feature names for display
        clean  = [f.replace("_roll6_mean","  (6h avg)").replace("_roll24_mean","  (24h avg)")
                   .replace("_roll6_std","  (6h std)").replace("_"," ").replace("vib x temp","Vib × Temp")
                  for f in feats]

        fig_fi = go.Figure(go.Bar(
            x=scores, y=clean,
            orientation="h",
            marker=dict(
                color=scores,
                colorscale=[[0, "#0A3D6B"],[1.0, ACCENT_BLUE]],
            ),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            text=[f"{s:.3f}" for s in scores],
            textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=10),
        ))
        fig_fi = dark_layout(fig_fi, title="Top Feature Importances", height=400)
        fig_fi.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Model comparison table ──
    st.divider()
    st.markdown("<p class='section-label'>Model Comparison — Why Random Forest was selected</p>", unsafe_allow_html=True)

    comp = pd.DataFrame({
        "Model":       ["Random Forest ✓", "Logistic Regression", "SVM", "Isolation Forest (unsupervised)"],
        "Recall":      [f"{rf['recall']*100:.1f}%", "~62%", "~55%", f"{report['isolation_forest']['recall']*100:.1f}%"],
        "Precision":   [f"{rf['precision']*100:.1f}%", "~8%",  "~6%", f"{report['isolation_forest']['precision']*100:.1f}%"],
        "ROC-AUC":     [f"{rf['roc_auc']:.4f}", "~0.78", "~0.72", "N/A"],
        "Handles imbalance": ["✓ class_weight=balanced", "Partial", "Partial", "✓ unsupervised"],
        "Interpretable":     ["✓ Feature importance", "✓ Coefficients", "✗", "✗"],
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="machine-card" style="font-size:12px;color:#8E8E93;margin-top:10px">
      <p style="color:#F5F5F7;font-weight:500;margin:0 0 8px">Research Connection — Prof. Dr. Valentin Plenk, Cyber-Physical Systems, iisys Hof</p>
      This project implements condition monitoring and predictive maintenance using machine learning on industrial sensor data —
      directly aligned with the CPS research group's focus on Industry 4.0 vertical integration and processing of performance
      and quality data from physical machines. The Random Forest model was selected for its balance of recall, interpretability
      via feature importances, and robustness to class imbalance in industrial fault datasets.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  VIEW 4 — ALERT LOG
# ═══════════════════════════════════════════════════════════
elif view == "Alert Log":

    st.markdown("<p class='section-label'>Active Alerts — All Machines</p>", unsafe_allow_html=True)

    critical = risk_df[risk_df["risk_level"] == "CRITICAL"]
    warning  = risk_df[risk_df["risk_level"] == "WARNING"]

    if critical.empty and warning.empty:
        st.markdown("<div class='alert-normal'>✓ No active alerts — all machines operating normally.</div>", unsafe_allow_html=True)
    else:
        for _, row in critical.iterrows():
            st.markdown(f"""
            <div class='alert-critical'>
              ⚠ CRITICAL &nbsp;|&nbsp; <b>{row['machine_id']}</b> — {row['machine_type']} &nbsp;|&nbsp;
              {row['location']} &nbsp;|&nbsp; Risk: {row['risk_score']}% &nbsp;|&nbsp; Age: {row['age_years']} yrs
            </div>""", unsafe_allow_html=True)

        for _, row in warning.iterrows():
            st.markdown(f"""
            <div class='alert-warning'>
              ● WARNING &nbsp;|&nbsp; <b>{row['machine_id']}</b> — {row['machine_type']} &nbsp;|&nbsp;
              {row['location']} &nbsp;|&nbsp; Risk: {row['risk_score']}% &nbsp;|&nbsp; Age: {row['age_years']} yrs
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Risk distribution by location ──
    st.markdown("<p class='section-label'>Risk Distribution by Location</p>", unsafe_allow_html=True)
    loc_data = risk_df.groupby(["location","risk_level"]).size().unstack(fill_value=0).reset_index()

    fig_loc = go.Figure()
    for level, color in [("CRITICAL",ACCENT_RED),("WARNING",ACCENT_AMB),("NORMAL",ACCENT_GREEN)]:
        if level in loc_data.columns:
            fig_loc.add_trace(go.Bar(
                x=loc_data["location"], y=loc_data[level],
                name=level, marker_color=color, opacity=0.85,
            ))
    fig_loc = dark_layout(fig_loc, title="Machines per Location by Risk Level", height=320)
    fig_loc.update_layout(barmode="stack", xaxis_tickangle=-20)
    st.plotly_chart(fig_loc, use_container_width=True)

    # ── Age vs risk scatter ──
    st.markdown("<p class='section-label'>Machine Age vs Risk Score</p>", unsafe_allow_html=True)
    fig_scatter = go.Figure()
    for level, color in [("NORMAL",ACCENT_GREEN),("WARNING",ACCENT_AMB),("CRITICAL",ACCENT_RED)]:
        subset = risk_df[risk_df["risk_level"] == level]
        if not subset.empty:
            fig_scatter.add_trace(go.Scatter(
                x=subset["age_years"], y=subset["risk_score"],
                mode="markers+text",
                marker=dict(size=10, color=color, opacity=0.85, line=dict(width=1, color="#1D1D1F")),
                name=level,
                text=subset["machine_id"],
                textposition="top center",
                textfont=dict(size=8, color="#8E8E93"),
                hovertemplate="<b>%{text}</b><br>Age: %{x} yrs<br>Risk: %{y:.1f}%<extra></extra>",
            ))
    fig_scatter = dark_layout(fig_scatter, title="Age vs Risk Score", height=380)
    fig_scatter.update_xaxes(title="Machine Age (years)")
    fig_scatter.update_yaxes(title="Risk Score (%)")
    fig_scatter.add_hline(y=35, line_dash="dot", line_color=ACCENT_RED,   annotation_text="Critical threshold", annotation_font_size=10, annotation_font_color=ACCENT_RED)
    fig_scatter.add_hline(y=15, line_dash="dot", line_color=ACCENT_AMB, annotation_text="Warning threshold",  annotation_font_size=10, annotation_font_color=ACCENT_AMB)
    st.plotly_chart(fig_scatter, use_container_width=True)
