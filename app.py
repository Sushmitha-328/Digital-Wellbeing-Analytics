"""
Digital Wellbeing Analytics — Streamlit Dashboard
==================================================
A production-oriented behavioural risk prediction application.

IMPORTANT DISCLAIMER
--------------------
This application provides **behavioural risk indicators** based on self-reported
smartphone usage patterns.  It is NOT a medical or psychological diagnosis tool.
All predictions are probabilistic and should not replace professional advice.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Make src/ importable when run from repo root ─────────────────── #
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import FEATURE_LABELS, RAW_FEATURES
from src.predict import DigitalWellbeingPredictor
from src.validate import validate_inputs

# ================================================================== #
#  Page configuration                                                 #
# ================================================================== #
st.set_page_config(
    page_title="Digital Wellbeing Analytics",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================== #
#  CSS — Premium dark theme                                           #
# ================================================================== #
st.markdown(
    """
    <style>
    /* ── Base ──────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 100%); }

    /* ── Sidebar ────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16213e 0%, #0f3460 100%);
        border-right: 1px solid rgba(100,180,255,0.15);
    }
    section[data-testid="stSidebar"] .stRadio > label {
        color: #b0c4de !important;
        font-size: 0.95rem;
    }

    /* ── Cards ──────────────────────────────────────────────────── */
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(100,180,255,0.15);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }
    .metric-card h3 { margin: 0 0 0.2rem; font-size: 0.8rem;
                      color: #7fa8d0; text-transform: uppercase;
                      letter-spacing: 0.1em; }
    .metric-card p  { margin: 0; font-size: 1.8rem;
                      font-weight: 700; color: #e8f4fd; }

    /* ── Risk banners ───────────────────────────────────────────── */
    .risk-high     { background: linear-gradient(135deg,#4a1a1a,#6b2323);
                     border: 1px solid #e74c3c; border-radius:14px;
                     padding:1.2rem 1.6rem; margin-bottom:1rem; }
    .risk-moderate { background: linear-gradient(135deg,#3d2a00,#5a3d00);
                     border: 1px solid #f39c12; border-radius:14px;
                     padding:1.2rem 1.6rem; margin-bottom:1rem; }
    .risk-low      { background: linear-gradient(135deg,#0d2a1a,#1a4028);
                     border: 1px solid #27ae60; border-radius:14px;
                     padding:1.2rem 1.6rem; margin-bottom:1rem; }

    .risk-high h2, .risk-moderate h2, .risk-low h2 {
        margin: 0 0 0.3rem; font-size: 1.3rem; }
    .risk-high h2     { color: #e74c3c; }
    .risk-moderate h2 { color: #f39c12; }
    .risk-low h2      { color: #27ae60; }
    .risk-high p, .risk-moderate p, .risk-low p {
        margin: 0; color: #d0d0d0; font-size: 0.9rem; }

    /* ── Suggestion cards ───────────────────────────────────────── */
    .suggestion-card {
        background: rgba(100,180,255,0.06);
        border-left: 3px solid #3498db;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        color: #d0e8fa;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* ── Factor bars ────────────────────────────────────────────── */
    .factor-row {
        display: flex; align-items: center;
        margin-bottom: 0.7rem; gap: 0.8rem;
    }
    .factor-label { width: 230px; font-size: 0.85rem; color: #a0b4c8; flex-shrink: 0; }
    .factor-bar-bg { flex: 1; background: rgba(255,255,255,0.07);
                     border-radius: 4px; height: 10px; overflow: hidden; }
    .factor-bar    { height: 100%; border-radius: 4px;
                     background: linear-gradient(90deg,#3498db,#9b59b6); }
    .factor-pct    { width: 45px; text-align: right; font-size: 0.8rem;
                     color: #7fa8d0; flex-shrink: 0; }

    /* ── Section headers ────────────────────────────────────────── */
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #64b4ff;
        margin: 1.5rem 0 0.8rem;
        border-bottom: 1px solid rgba(100,180,255,0.2);
        padding-bottom: 0.4rem;
    }

    /* ── Disclaimer ─────────────────────────────────────────────── */
    .disclaimer {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,200,0,0.2);
        border-radius: 10px;
        padding: 0.8rem 1.1rem;
        font-size: 0.78rem;
        color: #a0a0b0;
        line-height: 1.5;
        margin-top: 1.5rem;
    }

    /* ── Streamlit overrides ────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg,#3498db,#9b59b6);
        color: white; border: none; border-radius: 10px;
        padding: 0.65rem 2rem; font-weight: 600;
        font-size: 1rem; transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    h1 { color: #e8f4fd !important; }
    h2 { color: #c0d8f0 !important; }
    h3 { color: #a0c4e0 !important; }
    label, .stSelectbox label, .stNumberInput label { color: #90aec8 !important; }

    /* Hide Streamlit watermark */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ================================================================== #
#  Model loading (cached)                                             #
# ================================================================== #
@st.cache_resource(show_spinner="Loading model…")
def load_predictor() -> DigitalWellbeingPredictor | None:
    candidates = [
        ROOT / "random_forest_final_model.pkl",
        ROOT / "models" / "random_forest_final.pkl",
        ROOT / "random_forest_model.pkl",
    ]
    for path in candidates:
        if path.exists():
            return DigitalWellbeingPredictor(path)
    return None


predictor = load_predictor()


# ================================================================== #
#  Helper: Gauge chart                                                #
# ================================================================== #
def _gauge(prob: float, category: str) -> go.Figure:
    colour_map = {"LOW": "#27ae60", "MODERATE": "#f39c12", "HIGH": "#e74c3c"}
    colour = colour_map.get(category, "#3498db")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"size": 38, "color": colour}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#506070",
                    "tickfont": {"color": "#7090a0", "size": 11},
                },
                "bar": {"color": colour, "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35],  "color": "rgba(39,174,96,0.15)"},
                    {"range": [35, 65], "color": "rgba(243,156,18,0.15)"},
                    {"range": [65, 100],"color": "rgba(231,76,60,0.15)"},
                ],
                "threshold": {
                    "line": {"color": colour, "width": 3},
                    "thickness": 0.8,
                    "value": prob * 100,
                },
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


# ================================================================== #
#  Helper: Factor importance bars (HTML)                              #
# ================================================================== #
def _render_factors(top_factors: list[dict]) -> str:
    max_imp = max(f["importance"] for f in top_factors) or 1.0
    rows = ""
    for f in top_factors:
        pct = f["importance"] / max_imp * 100
        rows += (
            f'<div class="factor-row">'
            f'  <div class="factor-label">{f["label"]}</div>'
            f'  <div class="factor-bar-bg">'
            f'    <div class="factor-bar" style="width:{pct:.1f}%"></div>'
            f'  </div>'
            f'  <div class="factor-pct">{f["importance"]*100:.1f}%</div>'
            f'</div>'
        )
    return rows


# ================================================================== #
#  Sidebar navigation                                                 #
# ================================================================== #
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:1.2rem 0 0.5rem;">
          <div style="font-size:2.5rem">📱</div>
          <div style="font-size:1.1rem;font-weight:700;color:#e8f4fd;">Digital Wellbeing</div>
          <div style="font-size:0.75rem;color:#7090a0;margin-top:0.2rem;">Analytics · ML Dashboard</div>
        </div>
        <hr style="border-color:rgba(100,180,255,0.15);margin:0.8rem 0;">
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        options=[
            "🏠  Overview",
            "📊  Risk Assessment",
            "📂  Batch Prediction",
            "📈  Model Performance",
            "ℹ️  About",
        ],
        label_visibility="collapsed",
    )


# ================================================================== #
#  Page 1 — Overview                                                  #
# ================================================================== #
if page == "🏠  Overview":
    st.markdown(
        """
        <h1 style="margin-bottom:0.2rem;">📱 Digital Wellbeing Analytics</h1>
        <p style="color:#7090a0;font-size:1.05rem;margin-top:0;">
        Behavioural risk prediction powered by Machine Learning
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h3>Dataset</h3><p>7,500</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>Features</h3><p>17</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>ROC-AUC</h3><p>0.965</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h3>F1-Score</h3><p>0.930</p></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown("### 🎯 What This Application Does")
    st.markdown(
        """
        This dashboard analyses **self-reported smartphone usage patterns** and uses a
        trained **Random Forest classifier** to estimate an individual's behavioural
        risk profile for smartphone over-reliance.

        Rather than giving a simple yes/no label, the system provides:

        | Output | Description |
        |--------|-------------|
        | **Risk Probability** | A continuous score (0–100%) |
        | **Risk Category** | LOW / MODERATE / HIGH with transparent thresholds |
        | **Top Risk Factors** | The behavioural patterns driving the prediction |
        | **Personalised Suggestions** | Evidence-informed digital wellbeing tips |
        | **Batch Prediction** | Upload a CSV to analyse multiple users at once |
        """
    )

    st.markdown("### 🏗️ System Architecture")
    st.markdown(
        """
        ```
        User Input
            │
            ▼
        Input Validation  (src/validate.py)
            │
            ▼
        Feature Engineering  (src/features.py)
            │  Raw features → + 5 engineered behavioural features
            ▼
        sklearn Pipeline
            │  ColumnTransformer (OHE + StandardScaler)
            │  Random Forest Classifier
            ▼
        Prediction Service  (src/predict.py)
            │  Probability · Risk Category · Top Factors · Suggestions
            ▼
        Streamlit Dashboard  (app.py)
        ```
        """
    )

    st.markdown(
        '<div class="disclaimer">'
        "⚠️  <strong>Disclaimer</strong>: This tool provides <em>behavioural risk indicators</em> "
        "based on self-reported data.  It is <strong>not</strong> a medical or psychological "
        "diagnosis.  All results are probabilistic estimates and should not replace "
        "professional advice."
        "</div>",
        unsafe_allow_html=True,
    )


# ================================================================== #
#  Page 2 — Risk Assessment                                           #
# ================================================================== #
elif page == "📊  Risk Assessment":
    st.markdown(
        '<h1 style="margin-bottom:0.2rem;">📊 Personal Risk Assessment</h1>'
        '<p style="color:#7090a0;">Enter your daily smartphone usage details below.</p>',
        unsafe_allow_html=True,
    )

    if predictor is None:
        st.error(
            "⚠️  Model file not found. Please ensure `random_forest_final_model.pkl` "
            "is present in the repository root."
        )
        st.stop()

    # ── Input form ─────────────────────────────────────────────── #
    with st.form("risk_form"):
        st.markdown('<div class="section-header">👤 Demographics</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age    = st.number_input("Age", min_value=10, max_value=80, value=21, step=1)
        with c2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])

        st.markdown('<div class="section-header">📱 Screen Time</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            daily_screen = st.number_input("Daily Screen Time (hrs)", 0.0, 18.0, 6.0, 0.5)
        with c2:
            weekend_screen = st.number_input("Weekend Screen Time (hrs)", 0.0, 18.0, 7.0, 0.5)
        with c3:
            sleep_hrs = st.number_input("Sleep Hours", 3.0, 14.0, 7.0, 0.5)

        st.markdown('<div class="section-header">🎮 App Usage Breakdown</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            social = st.number_input("Social Media (hrs)", 0.0, 16.0, 2.0, 0.5)
        with c2:
            gaming = st.number_input("Gaming (hrs)", 0.0, 16.0, 1.0, 0.5)
        with c3:
            work   = st.number_input("Work / Study (hrs)", 0.0, 16.0, 4.0, 0.5)

        st.markdown('<div class="section-header">🔔 Notifications & Engagement</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            notifs    = st.number_input("Notifications per Day", 0, 3000, 80, 10)
        with c2:
            app_opens = st.number_input("App Opens per Day", 1, 2000, 50, 5)

        st.markdown('<div class="section-header">🧠 Wellbeing Indicators</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
        with c2:
            impact = st.selectbox(
                "Does phone use impact your academic / work performance?",
                ["No", "Yes"],
            )

        submitted = st.form_submit_button("🔍  Analyse My Risk")

    # ── Process submission ──────────────────────────────────────── #
    if submitted:
        raw_inputs = {
            "age":                     age,
            "gender":                  gender,
            "daily_screen_time_hours": daily_screen,
            "social_media_hours":      social,
            "gaming_hours":            gaming,
            "work_study_hours":        work,
            "sleep_hours":             sleep_hrs,
            "notifications_per_day":   notifs,
            "app_opens_per_day":       app_opens,
            "weekend_screen_time":     weekend_screen,
            "stress_level":            stress,
            "academic_work_impact":    impact,
        }

        # Validate
        is_valid, errors = validate_inputs(raw_inputs)
        if not is_valid:
            st.markdown("#### ⚠️ Please correct the following inputs:")
            for err in errors:
                st.markdown(f"- {err}")
            st.stop()

        # Predict
        with st.spinner("Analysing your usage patterns…"):
            result = predictor.predict(raw_inputs)

        prob     = result.probability
        category = result.risk_category

        st.markdown("---")
        st.markdown("## 📋 Your Results")

        # ── Risk banner + gauge ────────────────────────────────── #
        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            css_class = f"risk-{category.lower()}"
            descriptions = {
                "LOW": (
                    "Your smartphone usage patterns are <strong>within a healthy range</strong>. "
                    "Keep up the good habits!"
                ),
                "MODERATE": (
                    "Some aspects of your usage show <strong>borderline patterns</strong>. "
                    "A few small adjustments could help."
                ),
                "HIGH": (
                    "Your usage patterns are <strong>associated with higher risk</strong> of "
                    "smartphone over-reliance. Consider the suggestions below."
                ),
            }
            st.markdown(
                f'<div class="{css_class}">'
                f'<h2>Risk Level: {category}</h2>'
                f'<p>{descriptions[category]}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Probability metrics
            mc1, mc2 = st.columns(2)
            mc1.metric("Risk Probability", f"{prob*100:.1f}%")
            mc2.metric(
                "Risk Threshold",
                "p < 35% = LOW · 35–65% = MODERATE · ≥ 65% = HIGH",
            )

        with col_right:
            st.plotly_chart(
                _gauge(prob, category),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        # ── Top risk factors ───────────────────────────────────── #
        st.markdown('<div class="section-header">🔍 Top Behavioural Risk Factors</div>', unsafe_allow_html=True)
        st.caption(
            "These are the features most associated with the model's prediction — "
            "based on Random Forest feature importance.  Association ≠ causation."
        )
        st.markdown(
            _render_factors(result.top_risk_factors),
            unsafe_allow_html=True,
        )

        # ── Personalised suggestions ───────────────────────────── #
        st.markdown('<div class="section-header">💡 Personalised Suggestions</div>', unsafe_allow_html=True)
        for tip in result.suggestions:
            st.markdown(f'<div class="suggestion-card">{tip}</div>', unsafe_allow_html=True)

        # ── Summary table ──────────────────────────────────────── #
        with st.expander("📄 View full feature profile"):
            st.dataframe(result.engineered_df.T.rename(columns={0: "Your Value"}))

        # Disclaimer
        st.markdown(
            '<div class="disclaimer">'
            "⚠️  <strong>Disclaimer</strong>: This is a <em>behavioural risk indicator</em> based on "
            "self-reported data and a machine learning model trained on survey data.  It is "
            "<strong>not</strong> a medical diagnosis.  The model has known limitations — "
            "see the <em>About</em> section for details."
            "</div>",
            unsafe_allow_html=True,
        )


# ================================================================== #
#  Page 3 — Batch Prediction                                          #
# ================================================================== #
elif page == "📂  Batch Prediction":
    st.markdown(
        '<h1>📂 Batch Prediction</h1>'
        '<p style="color:#7090a0;">Upload a CSV file to analyse multiple users at once.</p>',
        unsafe_allow_html=True,
    )

    if predictor is None:
        st.error("⚠️  Model file not found.")
        st.stop()

    # ── Template download ───────────────────────────────────────── #
    st.markdown("#### Step 1 — Download the CSV template")
    template_cols = RAW_FEATURES
    template_df = pd.DataFrame(
        [
            [21, "Female", 8.0, 4.0, 2.0, 4.0, 7.0, 150, 100, 10.0, "High",   "Yes"],
            [25, "Male",   3.0, 1.5, 0.5, 6.0, 8.0,  40,  30,  4.0, "Low",    "No"],
            [19, "Other",  5.0, 2.5, 1.0, 3.0, 6.5,  90,  70,  7.0, "Medium", "Yes"],
        ],
        columns=template_cols,
    )
    csv_template = template_df.to_csv(index=False)
    st.download_button(
        "⬇️  Download Template CSV",
        data=csv_template,
        file_name="dwa_batch_template.csv",
        mime="text/csv",
    )

    # ── Upload ──────────────────────────────────────────────────── #
    st.markdown("#### Step 2 — Upload your file")
    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read file: {exc}")
            st.stop()

        # Check required columns
        missing_cols = [c for c in RAW_FEATURES if c not in batch_df.columns]
        if missing_cols:
            st.error(
                f"The uploaded file is missing required columns: {missing_cols}\n\n"
                "Please use the template above."
            )
            st.stop()

        st.success(f"✅  Loaded {len(batch_df):,} row(s).")
        st.dataframe(batch_df.head(), use_container_width=True)

        with st.spinner(f"Predicting for {len(batch_df):,} users…"):
            try:
                results_df = predictor.predict_batch(batch_df)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()

        # ── Results ──────────────────────────────────────────────── #
        st.markdown("#### Step 3 — Results")

        # Summary metrics
        mc1, mc2, mc3 = st.columns(3)
        counts = results_df["risk_category"].value_counts()
        mc1.metric("🔴 HIGH Risk",     counts.get("HIGH",     0))
        mc2.metric("🟡 MODERATE Risk", counts.get("MODERATE", 0))
        mc3.metric("🟢 LOW Risk",      counts.get("LOW",      0))

        # Distribution bar chart
        fig = go.Figure(
            go.Bar(
                x=list(counts.index),
                y=list(counts.values),
                marker_color=["#e74c3c" if k=="HIGH" else "#f39c12" if k=="MODERATE" else "#27ae60"
                              for k in counts.index],
                text=list(counts.values),
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Risk Category Distribution",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#90aec8", "family": "Inter"},
            xaxis={"title": "Risk Category"},
            yaxis={"title": "Count"},
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Full results table
        st.dataframe(
            results_df[
                RAW_FEATURES + ["addicted_probability", "risk_category"]
            ].sort_values("addicted_probability", ascending=False),
            use_container_width=True,
        )

        # Download
        csv_out = results_df.to_csv(index=False)
        st.download_button(
            "⬇️  Download Predictions CSV",
            data=csv_out,
            file_name="dwa_predictions.csv",
            mime="text/csv",
        )

        st.markdown(
            '<div class="disclaimer">'
            "⚠️  Batch results are for analytical purposes only. "
            "Never share or publish results containing personal identifiable information."
            "</div>",
            unsafe_allow_html=True,
        )


# ================================================================== #
#  Page 4 — Model Performance                                         #
# ================================================================== #
elif page == "📈  Model Performance":
    st.markdown(
        '<h1>📈 Model Performance</h1>'
        '<p style="color:#7090a0;">Evaluation metrics from the training pipeline.</p>',
        unsafe_allow_html=True,
    )

    st.info(
        "Metrics below are from the held-out test set (20% of 7,500 records, "
        "stratified split, random_state=42).  See notebook `05_machine_learning.ipynb` "
        "for full training details."
    )

    # ── Model comparison table ──────────────────────────────────── #
    st.markdown("### Model Comparison")
    metrics_df = pd.DataFrame(
        {
            "Model":      ["Logistic Regression", "Random Forest (Final)"],
            "Accuracy":   [0.8987,  0.9213],
            "Precision":  [0.9136,  0.9340],
            "Recall":     [0.9463,  0.9580],
            "F1-Score":   [0.9297,  0.9458],
            "ROC-AUC":    [0.9575,  0.9648],
        }
    ).set_index("Model")
    st.dataframe(metrics_df.style.highlight_max(axis=0, color="#1a3a1a"), use_container_width=True)

    st.caption(
        "⚠️  Random Forest metrics shown are from the saved model. "
        "For full cross-validation results, run notebook `05_machine_learning.ipynb`."
    )

    # ── Metric bar chart ────────────────────────────────────────── #
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    fig = go.Figure()
    colours = ["#3498db", "#9b59b6"]
    for i, model in enumerate(metrics_df.index):
        fig.add_trace(
            go.Bar(
                name=model,
                x=metric_cols,
                y=[metrics_df.loc[model, m] for m in metric_cols],
                marker_color=colours[i],
                text=[f"{metrics_df.loc[model, m]:.4f}" for m in metric_cols],
                textposition="outside",
            )
        )
    fig.update_layout(
        barmode="group",
        title="Model Comparison — Test Set Metrics",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#90aec8", "family": "Inter"},
        yaxis={"range": [0.80, 1.00], "title": "Score"},
        legend={"bgcolor": "rgba(0,0,0,0)"},
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Class imbalance note ────────────────────────────────────── #
    st.markdown("### ⚖️ Class Imbalance")
    st.markdown(
        """
        The dataset has a **~71% / 29%** class distribution (addicted / not addicted).

        | Class | Count (test set) | Fraction |
        |-------|-----------------|----------|
        | 0 — Not Addicted | 438 | 29.2% |
        | 1 — Addicted | 1,062 | 70.8% |

        Because the dataset is imbalanced:
        - **Accuracy alone is insufficient** — a naive classifier predicting "addicted" for all users would achieve 70.8% accuracy.
        - **F1-Score** (harmonic mean of precision and recall) is the primary metric.
        - **ROC-AUC** (0.965) confirms the model discriminates well across all thresholds.
        """
    )

    # ── Feature importance ──────────────────────────────────────── #
    if predictor is not None:
        st.markdown("### 🔍 Global Feature Importance")
        st.caption(
            "Random Forest feature importances (mean impurity decrease). "
            "OHE-expanded categorical columns are shown individually."
        )
        names = predictor._transformed_feature_names
        imps  = predictor._feature_importances
        fi_df = (
            pd.DataFrame({"Feature": names, "Importance": imps})
            .sort_values("Importance", ascending=True)
            .tail(15)
        )
        fig2 = go.Figure(
            go.Bar(
                x=fi_df["Importance"],
                y=fi_df["Feature"],
                orientation="h",
                marker_color="#3498db",
            )
        )
        fig2.update_layout(
            title="Top 15 Feature Importances",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#90aec8", "family": "Inter"},
            xaxis={"title": "Importance"},
            height=460,
        )
        st.plotly_chart(fig2, use_container_width=True)


# ================================================================== #
#  Page 5 — About                                                     #
# ================================================================== #
elif page == "ℹ️  About":
    st.markdown(
        '<h1>ℹ️ About This Project</h1>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["📖 Project", "⚠️ Limitations & Ethics", "🔗 Links"])

    with tab1:
        st.markdown(
            """
            ### Digital Wellbeing Analytics

            This project demonstrates an end-to-end machine learning pipeline for
            **behavioural risk prediction** from self-reported smartphone usage data.

            **Dataset**
            - 7,500 smartphone usage records
            - 12 raw features (demographics, usage patterns, wellbeing indicators)
            - 5 engineered behavioural features
            - Binary target: smartphone addiction label

            **Methodology**
            1. Exploratory data analysis and statistical testing
            2. Feature engineering with documented rationale
            3. Preprocessing pipeline (OHE + StandardScaler via sklearn)
            4. Model comparison (Logistic Regression vs Random Forest)
            5. Risk categorisation with transparent thresholds
            6. Personalised suggestion engine (rule-based)
            7. Batch prediction with validation
            8. Deployed on Streamlit Cloud

            **Tech Stack**
            `Python` · `scikit-learn` · `pandas` · `numpy` · `plotly` · `streamlit` · `pytest`

            **Repository Structure**
            ```
            ├── app.py               # This dashboard
            ├── src/
            │   ├── features.py      # Feature engineering (single source of truth)
            │   ├── validate.py      # Input validation
            │   └── predict.py       # Inference service
            ├── tests/               # pytest unit tests
            └── notebooks/           # Exploratory analysis & model training
            ```
            """
        )

    with tab2:
        st.markdown(
            """
            ### ⚠️ Limitations

            1. **Self-report bias** — All features are self-reported; users may under- or over-estimate usage.
            2. **Dataset provenance** — The training dataset is not from a published clinical study.  It should be treated as a proxy, not ground truth.
            3. **Class imbalance** — ~71% of samples are labelled "addicted", which may bias probability estimates.
            4. **Demographic generalisability** — The model may not generalise equally across all age groups, cultures, or socioeconomic backgrounds.
            5. **False positives** — A HIGH risk label does not mean a user *is* addicted.  The cost of false positives is low (unnecessary concern); the cost of false negatives (missing a real problem) could be higher.
            6. **No longitudinal data** — Snapshots of behaviour are used; temporal patterns (e.g., worsening over time) are not captured.

            ### 🤝 Ethical Considerations

            - **Not a diagnosis** — This tool is for awareness only.  Smartphone addiction is a clinically complex phenomenon.
            - **No data stored** — Individual inputs are not logged or stored anywhere.
            - **Association ≠ causation** — High feature importance does not imply that a feature *causes* addiction.
            - **Privacy** — Batch prediction results should never contain personally identifiable information.
            """
        )

    with tab3:
        st.markdown(
            """
            ### 🔗 Links

            - 📦 **GitHub Repository**: [github.com/Sushmitha-328/Digital-Wellbeing-Analytics](https://github.com/Sushmitha-328/Digital-Wellbeing-Analytics)
            - 🌐 **Live Application**: [Streamlit Cloud](https://digital-wellbeing-analytics-9osycluwd9ypkzb6evdwqh.streamlit.app/)
            - 📓 **Notebooks**: Available in the `notebooks/` directory of the repository

            ---
            *Built as a portfolio demonstration of end-to-end ML engineering.*
            """
        )
