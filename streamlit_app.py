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

MODEL_NAME = "Random forest" if RANDOM_FOREST_AVAILABLE else "Decision tree (policy fallback)"
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Breast Cancer Analysis",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap');
    :root { --ink: #202c2a; --muted: #71807c; --paper: #f4f7f2; --line: #dce2dc; --green: #176b57; --coral: #d45b49; }
    .stApp { background: linear-gradient(145deg, #f4f7f2 0%, #edf3ef 58%, #f8f5ef 100%); color: var(--ink); font-family: 'DM Sans', sans-serif; }
    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; animation: reveal 420ms ease-out both; }
    [data-testid="stSidebar"] { background: #e9efe9; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div { padding-top: 1.5rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0 !important; }
    h1 { font-size: 2.15rem !important; font-weight: 650 !important; }
    .eyebrow { color: var(--green); font: 500 0.72rem 'DM Mono', monospace; letter-spacing: 0; text-transform: uppercase; }
    .lede { color: var(--muted); font-size: 1rem; margin-top: -0.5rem; }
    div[data-testid="stMetric"] { background: rgba(255,255,255,0.9); border: 1px solid var(--line); border-top: 3px solid var(--green); border-radius: 5px; padding: 14px 16px; }
    div[data-testid="stMetricLabel"] p { color: var(--muted); font-size: 0.78rem; }
    div[data-testid="stMetricValue"] { color: var(--ink); font-size: 1.7rem; }
    div[data-testid="stForm"] { border: 1px solid var(--line); border-radius: 5px; background: #fff; padding: 1rem; }
    .stButton button, .stFormSubmitButton button { background: var(--green); color: white; border: 0; border-radius: 4px; }
    .stButton button:hover, .stFormSubmitButton button:hover { background: #105440; color: white; }
    hr { border-color: var(--line); }
    @keyframes reveal { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
    @media (prefers-reduced-motion: reduce) { .block-container { animation: none; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models_and_data():
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    target = dataset.target
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    if RANDOM_FOREST_AVAILABLE:
        forest_model = RandomForestClassifier(
            criterion="gini",
            max_features="log2",
            max_depth=3,
            n_estimators=400,
            random_state=42,
        )
    else:
        forest_model = RandomForestClassifier(
            criterion="gini", max_depth=3, random_state=42
        )
    models = {
        MODEL_NAME: forest_model,
        "K-nearest neighbors": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=3)
        ),
        "Neural network": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(12,),
                activation="tanh",
                learning_rate_init=0.05,
                max_iter=1500,
                random_state=42,
            ),
        ),
    }
    for model in models.values():
        model.fit(x_train, y_train)
    return dataset, features, target, x_test, y_test, models


dataset, features, target, x_test, y_test, models = load_models_and_data()
class_names = {0: "Malignant", 1: "Benign"}
class_colors = {"Malignant": "#d45b49", "Benign": "#176b57"}
feature_options = list(features.columns)

with st.sidebar:
    st.markdown('<div class="eyebrow">Wisconsin · Diagnostic study</div>', unsafe_allow_html=True)
    st.markdown("## Breast Cancer Analysis")
    page = st.radio("Workspace", ["Overview", "Explore data", "Make a prediction"], label_visibility="collapsed")
    st.divider()
    st.caption("569 samples · 30 cell-nucleus measurements · 2 classes")
    st.caption("Research demonstration only. Not for clinical use.")


def page_heading(eyebrow, title, description):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="lede">{description}</div>', unsafe_allow_html=True)


if page == "Overview":
    page_heading(
        "Wisconsin diagnostic dataset",
        "Breast Cancer Analysis",
        "Explore diagnostic measurements, compare model performance, and inspect individual predictions.",
    )
    results = {}
    for name, model in models.items():
        probabilities = model.predict_proba(x_test)[:, 0]
        predictions = model.predict(x_test)
        results[name] = {
            "predictions": predictions,
            "malignant_probability": probabilities,
            "accuracy": accuracy_score(y_test, predictions),
            "auc": roc_auc_score(y_test == 0, probabilities),
            "sensitivity": recall_score(y_test, predictions, pos_label=0),
        }

    selected_model_name = st.selectbox(
        "Model to inspect", list(results), index=list(results).index(MODEL_NAME)
    )
    preferred = results[selected_model_name]
    metrics = st.columns(4)
    metrics[0].metric("Test samples", f"{len(y_test)}")
    metrics[1].metric(f"{selected_model_name} accuracy", f"{preferred['accuracy']:.1%}")
    metrics[2].metric("Malignant sensitivity", f"{preferred['sensitivity']:.1%}")
    metrics[3].metric(f"{selected_model_name} AUC", f"{preferred['auc']:.3f}")

    roc_column, matrix_column = st.columns([1.25, 0.85], gap="large")
    with roc_column:
        st.subheader("ROC curves")
        roc_figure = go.Figure()
        for name, result in results.items():
            false_positive, true_positive, _ = roc_curve(
                y_test == 0, result["malignant_probability"]
            )
            roc_figure.add_trace(
                go.Scatter(
                    x=false_positive,
                    y=true_positive,
                    mode="lines",
                    name=f"{name} · {result['auc']:.3f}",
                    line={"width": 2.5},
                )
            )
        roc_figure.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line={"dash": "dot", "color": "#a8b2ad"})
        )
        roc_figure.update_layout(
            height=350,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend={"orientation": "h", "y": -0.24},
        )
        st.plotly_chart(roc_figure)
    with matrix_column:
        st.subheader(f"{selected_model_name} · test set")
        matrix = confusion_matrix(y_test, preferred["predictions"], labels=[0, 1])
        matrix_figure = px.imshow(
            matrix,
            text_auto=True,
            x=["Pred. malignant", "Pred. benign"],
            y=["Actual malignant", "Actual benign"],
            color_continuous_scale=[[0, "#e9eeea"], [1, "#176b57"]],
            aspect="auto",
        )
        matrix_figure.update_layout(
            height=350,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(matrix_figure)
    st.caption("The dataset uses class 0 for malignant and class 1 for benign. Sensitivity is reported for the malignant class.")

elif page == "Explore data":
    page_heading(
        "Dataset explorer",
        "Explore diagnostic measurements.",
        "Compare cell measurements across malignant and benign diagnoses.",
    )
    plot_data = features.copy()
    plot_data["Diagnosis"] = target.map(class_names)
    plot_data["Diagnosis"] = plot_data["Diagnosis"].astype(str)
    feature_column, value_column, diagnosis_column = st.columns(3)
    with feature_column:
        x_feature = st.selectbox("Horizontal axis", feature_options, index=0)
    with value_column:
        y_feature = st.selectbox("Vertical axis", feature_options, index=27)
    with diagnosis_column:
        diagnosis_filter = st.multiselect(
            "Diagnoses", list(class_colors), default=list(class_colors)
        )
    if diagnosis_filter:
        plot_data = plot_data[plot_data["Diagnosis"].isin(diagnosis_filter)]
        scatter = px.scatter(
            plot_data,
            x=x_feature,
            y=y_feature,
            color="Diagnosis",
            color_discrete_map=class_colors,
            opacity=0.72,
            hover_data={"Diagnosis": True},
        )
        scatter.update_layout(
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="white",
            legend_title_text="Diagnosis",
            margin={"l": 10, "r": 10, "t": 15, "b": 10},
        )
        st.plotly_chart(scatter)
    else:
        st.info("Select at least one diagnosis to display the measurements.")

    importance_model = models[MODEL_NAME]
    importance = (
        importance_model.feature_importances_
        if hasattr(importance_model, "feature_importances_")
        else None
    )
    if importance is not None:
        st.subheader(f"{MODEL_NAME} · most informative features")
        ranking = px.bar(
            x=importance,
            y=feature_options,
            orientation="h",
            labels={"x": "Feature importance", "y": ""},
            color=importance,
            color_continuous_scale=[[0, "#a9c9bc"], [1, "#176b57"]],
        )
        ranking.update_layout(
            height=620,
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
        )
        st.plotly_chart(ranking)

else:
    page_heading(
        "Interactive prediction",
        "Enter a set of cell measurements.",
        "Adjust the feature values and compare the model's estimated class probabilities.",
    )
    model_name = st.selectbox("Model", list(models.keys()))
    feature_groups = [
        ("Mean measurements", [name for name in feature_options if name.startswith("mean ")]),
        ("Measurement error", [name for name in feature_options if name.endswith(" error")]),
        ("Worst measurements", [name for name in feature_options if name.startswith("worst ")]),
    ]
    with st.form("prediction_form"):
        values = {}
        group_columns = st.columns(3, gap="medium")
        for column, (group_name, group_features) in zip(group_columns, feature_groups):
            with column:
                st.markdown(f"**{group_name}**")
                for feature in group_features:
                    minimum = float(features[feature].min())
                    maximum = float(features[feature].max())
                    values[feature] = st.number_input(
                        feature,
                        min_value=minimum,
                        max_value=maximum,
                        value=float(features[feature].median()),
                        format="%.5f",
                        key=f"feature_{feature}",
                    )
        submitted = st.form_submit_button("Run prediction", type="primary")

    if submitted:
        input_row = features.iloc[[0]].copy()
        for feature, value in values.items():
            input_row.loc[input_row.index[0], feature] = value
        selected_model = models[model_name]
        probabilities = selected_model.predict_proba(input_row)[0]
        predicted_class = int(selected_model.predict(input_row)[0])
        malignant_probability = float(probabilities[0])
        benign_probability = float(probabilities[1])
        st.divider()
        if predicted_class == 0:
            st.error(f"Model estimate: {class_names[predicted_class]}")
        else:
            st.success(f"Model estimate: {class_names[predicted_class]}")
        probability_columns = st.columns(2)
        probability_columns[0].metric("Estimated malignant probability", f"{malignant_probability:.1%}")
        probability_columns[1].metric("Estimated benign probability", f"{benign_probability:.1%}")
        st.warning("This is an educational machine-learning demonstration, not a medical diagnosis or a substitute for professional care.")
    else:
        st.caption("Values start at the dataset medians. Submit the form to calculate a prediction.")