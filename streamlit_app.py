import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_breast_cancer

try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError as error:
    if "application control policy has blocked" not in str(error).lower():
        raise
    from sklearn.tree import DecisionTreeClassifier as RandomForestClassifier
    RANDOM_FOREST_AVAILABLE = False
else:
    RANDOM_FOREST_AVAILABLE = True

MODEL_NAME = "Random Forest" if RANDOM_FOREST_AVAILABLE else "Decision Tree (Policy Fallback)"
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Page Config
st.set_page_config(
    page_title="OncoAI · Breast Cancer Diagnostic Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Contrast CSS Styling to Ensure ALL Text & Words are 100% Visible
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    /* Main App Background */
    .stApp {
        background-color: #0b0f19 !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #f8fafc !important;
    }
    
    /* Global Text Color Overrides for Maximum Readability */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
    }
    
    p, span, label, li, td, th, div {
        color: #e2e8f0 !important;
    }
    
    /* Streamlit Markdown Container Text */
    div[data-testid="stMarkdownContainer"] p, 
    div[data-testid="stMarkdownContainer"] span, 
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3 {
        color: #f8fafc !important;
    }
    
    /* Form & Input Labels */
    div[data-testid="stWidgetLabel"] p, 
    div[data-testid="stWidgetLabel"] label, 
    label[data-testid="stMetricLabel"],
    .stRadio label,
    .stSelectbox label,
    .stNumberInput label,
    .stMultiSelect label {
        color: #ffffff !important;
        font-size: 0.92rem !important;
        font-weight: 700 !important;
        margin-bottom: 4px !important;
    }
    
    /* Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #111827 0%, #1e1b4b 60%, #0f172a 100%);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        color: #ffffff !important;
        margin-bottom: 1.8rem;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4);
        border: 1px solid #374151;
        position: relative;
        overflow: hidden;
    }
    
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.25);
        border: 1px solid rgba(165, 180, 252, 0.4);
        color: #e0e7ff !important;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    
    .status-pulse {
        width: 8px;
        height: 8px;
        background-color: #34d399;
        border-radius: 50%;
        box-shadow: 0 0 10px #34d399;
    }
    
    .hero-title {
        font-size: 2.3rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        margin: 0 0 0.5rem 0 !important;
        letter-spacing: -0.02em !important;
        line-height: 1.2 !important;
    }
    
    .hero-subtitle {
        color: #cbd5e1 !important;
        font-size: 1.05rem;
        max-width: 780px;
        margin: 0;
        font-weight: 400;
        line-height: 1.55;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1f2937 !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0 0.5rem 1.25rem 0.5rem;
        border-bottom: 1px solid #1f2937;
        margin-bottom: 1.25rem;
    }
    
    .brand-icon {
        width: 44px;
        height: 44px;
        background: linear-gradient(135deg, #6366f1 0%, #06b6d4 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        color: #ffffff;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
    }
    
    .brand-title {
        font-weight: 800;
        font-size: 1.2rem;
        color: #ffffff !important;
        line-height: 1.1;
    }
    
    .brand-tag {
        font-size: 0.75rem;
        color: #9ca3af !important;
        font-weight: 600;
    }
    
    /* Sidebar Navigation Options */
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: #1f2937 !important;
        border: 1px solid #374151 !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        margin-bottom: 8px !important;
        color: #f9fafb !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        border-color: #6366f1 !important;
        background-color: #374151 !important;
    }
    
    /* Input Controls (Selectbox, Number Input, Text Inputs) */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] > div,
    input, 
    textarea {
        background-color: #1f2937 !important;
        color: #ffffff !important;
        border-color: #374151 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    
    div[data-baseweb="select"] span {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Dropdown Menus */
    div[data-baseweb="popover"] div, 
    div[data-baseweb="menu"] div,
    ul[role="listbox"] li {
        background-color: #1f2937 !important;
        color: #ffffff !important;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-top: 3px solid #6366f1 !important;
        border-radius: 12px !important;
        padding: 1.2rem 1.4rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
    }
    
    div[data-testid="stMetricLabel"] p {
        color: #9ca3af !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    
    div[data-testid="stMetricValue"] div {
        color: #38bdf8 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.85rem !important;
        font-weight: 700 !important;
    }
    
    /* Form & Buttons */
    div[data-testid="stForm"] {
        background-color: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 14px !important;
        padding: 1.5rem !important;
    }
    
    .stButton button {
        background-color: #1f2937 !important;
        color: #ffffff !important;
        border: 1px solid #374151 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.25rem !important;
    }
    
    .stButton button:hover {
        background-color: #374151 !important;
        border-color: #6366f1 !important;
        color: #ffffff !important;
    }
    
    .stFormSubmitButton button {
        background: linear-gradient(135deg, #6366f1 0%, #3b82f6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 0.75rem 2rem !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
    }
    
    .stFormSubmitButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
    }

    /* Assessment Result Badges */
    .result-pill-malignant {
        background-color: rgba(225, 29, 72, 0.18);
        border: 1px solid rgba(225, 29, 72, 0.5);
        color: #ffe4e6 !important;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        font-weight: 800;
        font-size: 1.25rem;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1rem;
    }
    
    .result-pill-benign {
        background-color: rgba(16, 185, 129, 0.18);
        border: 1px solid rgba(16, 185, 129, 0.5);
        color: #d1fae5 !important;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        font-weight: 800;
        font-size: 1.25rem;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1rem;
    }
    
    /* DataFrame Visibility */
    div[data-testid="stDataFrame"] {
        background-color: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 12px !important;
    }
    
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load Dataset & Train Models
@st.cache_resource
def load_data_and_models():
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    target = dataset.target
    
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    
    if RANDOM_FOREST_AVAILABLE:
        rf = RandomForestClassifier(
            criterion="gini",
            max_features="log2",
            max_depth=4,
            n_estimators=300,
            random_state=42,
        )
    else:
        rf = RandomForestClassifier(
            criterion="gini", max_depth=4, random_state=42
        )
        
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            max_iter=1200,
            random_state=42,
        ),
    )
    
    models = {
        MODEL_NAME: rf,
        "K-Nearest Neighbors": knn,
        "Neural Network (MLP)": mlp,
    }
    
    for model in models.values():
        model.fit(x_train, y_train)
        
    return dataset, features, target, x_test, y_test, models

dataset, features, target, x_test, y_test, models = load_data_and_models()
class_names = {0: "Malignant", 1: "Benign"}
class_colors = {"Malignant": "#f43f5e", "Benign": "#10b981"}
feature_options = list(features.columns)

# Sidebar Navigation
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-icon">🩺</div>
            <div>
                <div class="brand-title">OncoAI Studio</div>
                <div class="brand-tag">v2.4 Diagnostic Suite</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    page = st.radio(
        "Navigation",
        ["📊 Executive Dashboard", "🔬 Feature Analytics", "🎯 AI Risk Predictor", "📈 Model Leaderboard"],
        label_visibility="collapsed",
    )
    
    st.divider()
    
    st.markdown("### 📋 Dataset Overview")
    st.markdown("""
    - **Cohort Size**: 569 patient samples
    - **Biomarkers**: 30 cell-nucleus features
    - **Classes**: Benign (357) vs Malignant (212)
    - **Source**: UCI Wisconsin Diagnostic (WDBC)
    """)
    
    st.divider()
    st.caption("🔒 HIPAA Compliant Architecture Demo. Intended for research and educational visualization only.")

# Helper Header Generator
def render_hero(title, subtitle, tag="Wisconsin Diagnostic Intelligence"):
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="badge-pill">
                <span class="status-pulse"></span>
                {tag}
            </div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Page 1: Executive Dashboard
if page == "📊 Executive Dashboard":
    render_hero(
        "Breast Cancer Diagnostic Intelligence",
        "Comprehensive machine learning telemetry comparing multi-model classification, ROC performance, and confusion matrices on clinical test cohorts.",
    )
    
    results = {}
    for name, model in models.items():
        probs = model.predict_proba(x_test)[:, 0]  # Prob of Malignant (class 0)
        preds = model.predict(x_test)
        results[name] = {
            "predictions": preds,
            "malignant_probability": probs,
            "accuracy": accuracy_score(y_test, preds),
            "auc": roc_auc_score(y_test == 0, probs),
            "sensitivity": recall_score(y_test, preds, pos_label=0),
            "precision": precision_score(y_test, preds, pos_label=0),
            "f1": f1_score(y_test, preds, pos_label=0),
        }
        
    col_sel, _ = st.columns([1, 2])
    with col_sel:
        selected_model = st.selectbox("Primary Model Inspection", list(results.keys()), index=0)
        
    m = results[selected_model]
    
    # Top KPI Bar
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Test Cohort Size", f"{len(y_test)} patients")
    kpi2.metric(f"{selected_model} Accuracy", f"{m['accuracy']:.1%}")
    kpi3.metric("Malignant Recall (Sensitivity)", f"{m['sensitivity']:.1%}")
    kpi4.metric("ROC AUC Score", f"{m['auc']:.3f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualization Grid
    c1, c2 = st.columns([1.3, 1], gap="large")
    
    with c1:
        st.markdown('### 📉 Multi-Model ROC Curves')
        roc_fig = go.Figure()
        
        colors = {"Random Forest": "#818cf8", "Decision Tree (Policy Fallback)": "#818cf8", "K-Nearest Neighbors": "#2dd4bf", "Neural Network (MLP)": "#c084fc"}
        for name, res in results.items():
            fpr, tpr, _ = roc_curve(y_test == 0, res["malignant_probability"])
            roc_fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{name} (AUC = {res['auc']:.3f})",
                    line=dict(width=3, color=colors.get(name, "#38bdf8")),
                )
            )
        roc_fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                name="Baseline Chance",
                line=dict(dash="dash", color="#64748b", width=1.5),
            )
        )
        roc_fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#111827",
            font=dict(color="#f8fafc", family="Plus Jakarta Sans"),
            xaxis=dict(title="False Positive Rate (1 - Specificity)", gridcolor="#1f2937", zeroline=False, tickfont=dict(color="#cbd5e1"), title_font=dict(color="#ffffff")),
            yaxis=dict(title="True Positive Rate (Sensitivity)", gridcolor="#1f2937", zeroline=False, tickfont=dict(color="#cbd5e1"), title_font=dict(color="#ffffff")),

            legend=dict(orientation="h", y=-0.22, x=0, font=dict(color="#f8fafc")),
            hovermode="x unified",
        )
        st.plotly_chart(roc_fig, use_container_width=True)
        
    with c2:
        st.markdown(f'### 🎯 Confusion Matrix · {selected_model}')
        cm = confusion_matrix(y_test, m["predictions"], labels=[0, 1])
        cm_fig = px.imshow(
            cm,
            text_auto=True,
            x=["Pred. Malignant", "Pred. Benign"],
            y=["Actual Malignant", "Actual Benign"],
            color_continuous_scale=[[0, "#111827"], [0.5, "#6366f1"], [1, "#818cf8"]],
            aspect="auto",
        )
        cm_fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=20, b=20),
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#111827",
            font=dict(color="#ffffff", family="Plus Jakarta Sans"),
            xaxis=dict(tickfont=dict(color="#ffffff", size=12)),
            yaxis=dict(tickfont=dict(color="#ffffff", size=12)),
        )
        st.plotly_chart(cm_fig, use_container_width=True)

# Page 2: Feature Analytics
elif page == "🔬 Feature Analytics":
    render_hero(
        "Cell Biomarker Exploratory Suite",
        "Analyze morphological features extracted from digitized fine needle aspirate (FNA) images of breast mass.",
        tag="Exploratory Data Analysis",
    )
    
    plot_df = features.copy()
    plot_df["Diagnosis"] = target.map(class_names)
    
    col_feat1, col_feat2, col_feat3 = st.columns(3)
    with col_feat1:
        x_col = st.selectbox("X-Axis Feature", feature_options, index=feature_options.index("mean radius"))
    with col_feat2:
        y_col = st.selectbox("Y-Axis Feature", feature_options, index=feature_options.index("mean texture"))
    with col_feat3:
        z_col = st.selectbox("3D Z-Axis Feature (Optional)", ["None"] + feature_options, index=feature_options.index("worst area") + 1)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    if z_col == "None":
        scatter_fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color="Diagnosis",
            color_discrete_map=class_colors,
            opacity=0.85,
            marginal_x="histogram",
            marginal_y="box",
            hover_data=["mean area", "worst perimeter"],
            title=f"Biomarker Interaction: {x_col.title()} vs {y_col.title()}",
        )
        scatter_fig.update_layout(
            height=540,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#111827",
            font=dict(color="#f8fafc", family="Plus Jakarta Sans"),
            xaxis=dict(gridcolor="#1f2937", tickfont=dict(color="#cbd5e1"), title_font=dict(color="#ffffff")),
            yaxis=dict(gridcolor="#1f2937", tickfont=dict(color="#cbd5e1"), title_font=dict(color="#ffffff")),

            legend=dict(title="Diagnosis", orientation="h", y=1.05, font=dict(color="#ffffff")),
        )
        st.plotly_chart(scatter_fig, use_container_width=True)
    else:
        scatter_3d = px.scatter_3d(
            plot_df,
            x=x_col,
            y=y_col,
            z=z_col,
            color="Diagnosis",
            color_discrete_map=class_colors,
            opacity=0.85,
            title=f"3D Morphological Space ({x_col}, {y_col}, {z_col})",
        )
        scatter_3d.update_layout(
            height=600,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc", family="Plus Jakarta Sans"),
        )
        st.plotly_chart(scatter_3d, use_container_width=True)
        
    st.divider()
    
    # Feature Importance Ranking
    rf_model = models[MODEL_NAME]
    if hasattr(rf_model, "feature_importances_"):
        st.markdown(f"### 🏆 Key Predictive Features ({MODEL_NAME})")
        imp_df = pd.DataFrame({
            "Feature": feature_options,
            "Importance": rf_model.feature_importances_
        }).sort_values(by="Importance", ascending=True).tail(15)
        
        bar_fig = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=[[0, "#818cf8"], [1, "#38bdf8"]],
            labels={"Importance": "Gini Impurity Decrease", "Feature": ""},
        )
        bar_fig.update_layout(
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#111827",
            font=dict(color="#f8fafc", family="Plus Jakarta Sans"),
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#1f2937", tickfont=dict(color="#cbd5e1"), title_font=dict(color="#ffffff")),

            yaxis=dict(tickfont=dict(color="#ffffff", size=12)),
        )
        st.plotly_chart(bar_fig, use_container_width=True)

# Page 3: AI Risk Predictor
elif page == "🎯 AI Risk Predictor":
    render_hero(
        "Patient Case AI Diagnostic Assistant",
        "Input cell morphological characteristics to calculate malignant risk probabilities using trained clinical models.",
        tag="Clinical Risk Inference",
    )
    
    # Patient Case Presets
    st.markdown("### 📋 Quick Sample Presets")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    
    benign_sample = features[target == 1].iloc[0]
    malignant_sample = features[target == 0].iloc[0]
    median_sample = features.median()
    
    if "input_values" not in st.session_state:
        st.session_state.input_values = median_sample.to_dict()
        
    if p_col1.button("🟢 Load Typical Benign Case", use_container_width=True):
        st.session_state.input_values = benign_sample.to_dict()
        st.rerun()
    if p_col2.button("🔴 Load Malignant Case", use_container_width=True):
        st.session_state.input_values = malignant_sample.to_dict()
        st.rerun()
    if p_col3.button("⚖️ Load Dataset Median Case", use_container_width=True):
        st.session_state.input_values = median_sample.to_dict()
        st.rerun()
    if p_col4.button("🔄 Reset Inputs", use_container_width=True):
        st.session_state.input_values = median_sample.to_dict()
        st.rerun()
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.form("clinical_form"):
        st.markdown("### 🩺 Biomarker Measurement Inputs")
        
        f_cols = st.columns(3, gap="medium")
        groups = [
            ("Mean Measurements", [f for f in feature_options if f.startswith("mean ")]),
            ("Standard Errors", [f for f in feature_options if f.endswith(" error")]),
            ("Worst (Largest) Measurements", [f for f in feature_options if f.startswith("worst ")]),
        ]
        
        current_vals = {}
        for col, (g_title, g_features) in zip(f_cols, groups):
            with col:
                st.markdown(f"#### {g_title}")
                for f in g_features:
                    min_val = float(features[f].min())
                    max_val = float(features[f].max())
                    default_val = float(st.session_state.input_values.get(f, features[f].median()))
                    current_vals[f] = st.number_input(
                        label=f.title(),
                        min_value=min_val * 0.5,
                        max_value=max_val * 1.5,
                        value=default_val,
                        format="%.4f",
                        key=f"input_{f}"
                    )
                    
        st.markdown("<br>", unsafe_allow_html=True)
        model_choice = st.selectbox("Select Classifier for Inference", list(models.keys()), index=0)
        submit_btn = st.form_submit_button("⚡ Run Clinical Inference Analysis", use_container_width=True)
        
    if submit_btn:
        input_data = pd.DataFrame([current_vals])
        chosen_model = models[model_choice]
        
        probs = chosen_model.predict_proba(input_data)[0]
        pred_class = chosen_model.predict(input_data)[0]
        
        prob_malignant = probs[0]
        prob_benign = probs[1]
        
        st.divider()
        st.markdown("## 📊 Diagnostic Inference Output")
        
        res_col1, res_col2 = st.columns([1, 1.2], gap="large")
        
        with res_col1:
            if pred_class == 0:
                st.markdown(
                    f"""
                    <div class="result-pill-malignant">
                        <span>🚨 Assessment: <strong>MALIGNANT</strong></span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-pill-benign">
                        <span>✅ Assessment: <strong>BENIGN</strong></span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
            p1, p2 = st.columns(2)
            p1.metric("Malignant Risk Probability", f"{prob_malignant:.1%}")
            p2.metric("Benign Probability", f"{prob_benign:.1%}")
            
            st.info("💡 **Clinical Note**: Malignant risk calculation evaluates high-dimensional nuclear perimeter, concavity, and texture features.")
            
        with res_col2:
            # Gauge Chart for Probability
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_malignant * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Malignant Probability Gauge (%)", 'font': {'size': 16, 'color': '#ffffff', 'family': 'Plus Jakarta Sans'}},
                number={'suffix': "%", 'font': {'size': 38, 'color': '#38bdf8', 'family': 'JetBrains Mono'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8", 'tickfont': {'color': '#ffffff'}},
                    'bar': {'color': "#f43f5e" if prob_malignant > 0.5 else "#10b981"},
                    'bgcolor': "#111827",
                    'borderwidth': 1,
                    'bordercolor': "#374151",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.2)'},
                        {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                        {'range': [70, 100], 'color': 'rgba(244, 63, 94, 0.2)'}
                    ],
                    'threshold': {
                        'line': {'color': "#ffffff", 'width': 3},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            gauge_fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(gauge_fig, use_container_width=True)

# Page 4: Model Leaderboard
else:
    render_hero(
        "Classifier Benchmark Leaderboard",
        "Rigorous performance breakdown across accuracy, recall, precision, F1 score, and area under curve (AUC).",
        tag="Model Performance Benchmarks",
    )
    
    benchmark_data = []
    for name, model in models.items():
        preds = model.predict(x_test)
        probs = model.predict_proba(x_test)[:, 0]
        benchmark_data.append({
            "Model Architecture": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Sensitivity (Recall)": recall_score(y_test, preds, pos_label=0),
            "Precision": precision_score(y_test, preds, pos_label=0),
            "F1 Score": f1_score(y_test, preds, pos_label=0),
            "ROC AUC Score": roc_auc_score(y_test == 0, probs),
        })
        
    bench_df = pd.DataFrame(benchmark_data).sort_values(by="Accuracy", ascending=False)
    
    st.dataframe(
        bench_df.style.format({
            "Accuracy": "{:.1%}",
            "Sensitivity (Recall)": "{:.1%}",
            "Precision": "{:.1%}",
            "F1 Score": "{:.3f}",
            "ROC AUC Score": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Comparison Bar Chart
    melted_df = bench_df.melt(id_vars=["Model Architecture"], value_vars=["Accuracy", "Sensitivity (Recall)", "Precision"], var_name="Metric", value_name="Score")
    comp_fig = px.bar(
        melted_df,
        x="Model Architecture",
        y="Score",
        color="Metric",
        barmode="group",
        color_discrete_sequence=["#818cf8", "#2dd4bf", "#c084fc"],
        title="Comparative Metric Profile Across Architectures",
    )
    comp_fig.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111827",
        font=dict(color="#f8fafc", family="Plus Jakarta Sans"),
        xaxis=dict(gridcolor="#1f2937", tickfont=dict(color="#ffffff", size=12)),
        yaxis=dict(range=[0.85, 1.0], gridcolor="#1f2937", tickfont=dict(color="#cbd5e1"), title_font=dict(color="#ffffff")),

        legend=dict(font=dict(color="#ffffff")),
    )
    st.plotly_chart(comp_fig, use_container_width=True)