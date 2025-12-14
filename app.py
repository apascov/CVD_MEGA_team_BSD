"""
Streamlit Visual Analytics Tool for Cardiovascular Disease Dataset
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib

# Import visualization functions
from visualizations.vis1 import plot_vis1
from visualizations.vis2 import plot_vis2
from visualizations.vis3 import plot_vis3

# Page configuration
st.set_page_config(
    page_title="CVD Analytics Tool",
    page_icon="❤️",
    layout="wide"
)

# Data loading with caching
@st.cache_data
def load_data():
    """Load the cardiovascular disease dataset"""
    return pd.read_csv("data/CVD_cleaned.csv")

# Model + metrics loading with caching
@st.cache_resource
def load_models():
    """Load trained ML models and metrics if available."""
    models = {}
    metrics = {}

    # Attempt to load models
    model_paths = {
        "Logistic Regression": "models/logreg.pkl",
        "Random Forest": "models/rf.pkl",
        "XGBoost": "models/xgb.pkl",
    }
    for name, path in model_paths.items():
        try:
            if Path(path).exists():
                models[name] = joblib.load(path)
        except Exception:
            pass

    # Attempt to load metrics
    metrics_path = Path("models/metrics.json")
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            metrics = {}

    return models, metrics

# Load data
df = load_data()

# Main title
st.title("❤️ Cardiovascular Disease Visual Analytics Tool")
st.markdown("---")

# Create two-column layout for dataset overview and ML model
left_col, right_col = st.columns([1, 1])

# ==================== LEFT COLUMN ====================
with left_col:
    # Introduction and Dataset Overview
    st.header("📊 Introduction and Data Overview")
    
    st.markdown("""
    **Welcome to the CVD Visual Analytics Tool!**
    
    This interactive application helps you explore cardiovascular disease (CVD) risk factors through:
    
    - **Visual Analytics**: Three interactive visualizations to explore relationships between health variables and CVD:
        - Categorical variable distributions by heart disease status
        - Heart disease rates across two categorical variables
        - Numerical variable distributions comparing CVD and non-CVD patients
    
    - **Personal Risk Prediction**: Enter your own health parameters to estimate your CVD risk using machine learning
    
    **About the Dataset:**  
    The dataset contains health and lifestyle information from patients, including demographic data, 
    behavioral factors, and medical history to assess cardiovascular disease risk.
    """)
    
    # Dataset statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", f"{len(df):,}")
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        if 'Heart_Disease' in df.columns:
            # Convert to binary if needed
            if df['Heart_Disease'].dtype == 'object':
                cvd_rate = (df['Heart_Disease'] == 'Yes').mean()
            else:
                cvd_rate = df['Heart_Disease'].mean()
            st.metric("CVD Rate", f"{cvd_rate*100:.1f}%")

# ==================== RIGHT COLUMN ====================
with right_col:
    # ML Model Summary
    st.header("🤖 Machine Learning Model")

    st.markdown("""
    **Models Used & Performance**
    
    We trained three machine learning models to predict CVD risk:
    
    1. **Logistic Regression**: Fast, interpretable baseline model
    2. **Random Forest**: Ensemble method, captures non-linear patterns
    3. **XGBoost**: Advanced boosting algorithm, often highest performance
    
    **Why these models?** They balance interpretability with predictive power and handle imbalanced data well.
    
    ---
    
    **Select a model below and choose a decision threshold to explore predictions.**
    """)

    models, metrics = load_models()
    if not models:
        st.info(
            "No saved models found in `models/`. Train in the notebook and save `*.pkl` + `metrics.json`."
        )
    else:
        # Model selector stored in session state so prediction section can reuse it
        model_name = st.selectbox("Select model", list(models.keys()), key="model_select")

        # Map display name to metrics key (if using short keys in JSON)
        name_to_key = {
            "Logistic Regression": "logreg",
            "Random Forest": "rf",
            "XGBoost": "xgb",
        }
        m = metrics.get(name_to_key.get(model_name, model_name), {})

        st.markdown(f"**Performance on Test Set** ({model_name})")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{m.get('accuracy', float('nan')):.3f}" if 'accuracy' in m else "N/A")
        c2.metric("Precision", f"{m.get('precision', float('nan')):.3f}" if 'precision' in m else "N/A")
        c3.metric("Recall", f"{m.get('recall', float('nan')):.3f}" if 'recall' in m else "N/A")
        c4.metric("F1", f"{m.get('f1', float('nan')):.3f}" if 'f1' in m else "N/A")
        c5.metric("ROC AUC", f"{m.get('roc_auc', float('nan')):.3f}" if 'roc_auc' in m else "N/A")

st.markdown("---")

# ==================== VISUALIZATION SECTION ====================
st.header("📈 Interactive Visualizations")

# Visualization selector
vis_choice = st.radio(
    "Choose Visualization",
    ["Visualization 1: Categorical Distribution", 
     "Visualization 2: Heart Disease Rate by Two Variables",
     "Visualization 3: Numerical Distribution"],
    horizontal=True
)

st.markdown("---")

# Visualization 1: Categorical variable distribution
if "Visualization 1" in vis_choice:
    st.subheader("Categorical Variable Distribution by Heart Disease")
    
    # Get categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'Heart_Disease' in cat_cols:
        cat_cols.remove('Heart_Disease')
    
    selected_var = st.selectbox(
        "Choose a categorical variable:",
        cat_cols,
        index=0 if cat_cols else 0
    )
    
    if selected_var:
        fig = plot_vis1(df, selected_var)
        st.pyplot(fig, use_container_width=True)
        st.caption(f"Distribution of {selected_var} split by Heart Disease status with percentages.")

# Visualization 2: Heart disease rate by two categorical variables
elif "Visualization 2" in vis_choice:
    st.subheader("Heart Disease Rate by Two Categorical Variables")
    
    # Get categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'Heart_Disease' in cat_cols:
        cat_cols.remove('Heart_Disease')
    
    col1, col2 = st.columns(2)
    with col1:
        choice1 = st.selectbox(
            "First variable (X-axis):",
            cat_cols,
            index=cat_cols.index('Age_Category') if 'Age_Category' in cat_cols else 0
        )
    with col2:
        choice2 = st.selectbox(
            "Second variable (Groups):",
            cat_cols,
            index=cat_cols.index('Sex') if 'Sex' in cat_cols else 1
        )
    
    if choice1 and choice2:
        fig = plot_vis2(df, choice1, choice2)
        st.pyplot(fig, use_container_width=True)
        st.caption(f"Heart disease rate grouped by {choice1} and {choice2}.")

# Visualization 3: Numerical variable distribution
elif "Visualization 3" in vis_choice:
    st.subheader("Numerical Variable Distribution by Heart Disease")
    
    # Get numerical columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'Heart_Disease' in num_cols:
        num_cols.remove('Heart_Disease')
    
    selected_num = st.selectbox(
        "Choose a numerical variable:",
        num_cols,
        index=num_cols.index('BMI') if 'BMI' in num_cols else 0
    )
    
    if selected_num:
        fig = plot_vis3(df, selected_num)
        st.pyplot(fig, use_container_width=True)
        st.caption(f"Distribution and boxplot of {selected_num} by Heart Disease status.")

st.markdown("---")

# ==================== PREDICTION SECTION ====================
st.header("🔮 CVD Risk Prediction")

st.markdown("""
Enter your health parameters to estimate cardiovascular disease risk.
""")

with st.form("pred_form"):
    st.subheader("Personal Information")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=45)
        gender = st.selectbox("Sex", ["Male", "Female"])  # matches feature name "Sex"
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    
    with col2:
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        # Auto-calculate BMI from height and weight
        height_m = height / 100.0  # convert cm to meters
        bmi = weight / (height_m ** 2)
        st.metric("BMI (auto-calculated)", f"{bmi:.1f}")
    
    st.subheader("Health Metrics")
    col3, col4 = st.columns(2)
    with col3:
        general_health = st.selectbox(
            "General_Health",
            ["Excellent", "Very Good", "Good", "Fair", "Poor"]
        )
        exercise = st.selectbox("Exercise", ["Yes", "No"])
        diabetes = st.selectbox("Diabetes", ["Yes", "No"])
        depression = st.selectbox("Depression", ["Yes", "No"])
    
    with col4:
        smoking = st.selectbox("Smoking_History", ["Yes", "No"])
        # If dataset present, pull choices; otherwise provide sensible defaults
        checkup_options = (
            sorted(df["Checkup"].dropna().unique().tolist())
            if "Checkup" in df.columns else
            [
                "Within the past year",
                "Within the past 2 years",
                "Within the past 5 years",
                "5 or more years ago",
                "Never",
            ]
        )
        checkup = st.selectbox("Checkup", checkup_options)
        age_cat_options = (
            sorted(df["Age_Category"].dropna().unique().tolist())
            if "Age_Category" in df.columns else
            ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]
        )
        age_category = st.selectbox("Age_Category", age_cat_options)

    st.subheader("Diet & Consumption")
    st.markdown("""**Ordinal Scale (1-5):**
    - **1**: Never
    - **2**: Rarely (few times per year)
    - **3**: Sometimes (few times per month)
    - **4**: Frequently (several times per week)
    - **5**: Very Frequently (daily or almost daily)
    """)
    
    ordinal_labels = {
        1: "Never",
        2: "Rarely (few times/year)",
        3: "Sometimes (few times/month)",
        4: "Frequently (several times/week)",
        5: "Very Frequently (daily)"
    }
    
    c5, c6, c7 = st.columns(3)
    with c5:
        alcohol = st.selectbox(
            "Alcohol_Consumption",
            [1, 2, 3, 4, 5],
            format_func=lambda x: ordinal_labels[x],
            index=1
        )
    with c6:
        fruit = st.selectbox(
            "Fruit_Consumption",
            [1, 2, 3, 4, 5],
            format_func=lambda x: ordinal_labels[x],
            index=3
        )
    with c7:
        greens = st.selectbox(
            "Green_Vegetables_Consumption",
            [1, 2, 3, 4, 5],
            format_func=lambda x: ordinal_labels[x],
            index=3
        )

    fried_potato = st.selectbox(
        "FriedPotato_Consumption",
        [1, 2, 3, 4, 5],
        format_func=lambda x: ordinal_labels[x],
        index=0
    )

    arthritis = st.selectbox("Arthritis", ["Yes", "No"])

    # Reuse selected model from summary; fallback to first available
    models, metrics = load_models()
    model_name = st.session_state.get("model_select", next(iter(models.keys())) if models else None)
    threshold_default = 0.5
    if model_name and metrics:
        name_to_key = {"Logistic Regression": "logreg", "Random Forest": "rf", "XGBoost": "xgb"}
        m = metrics.get(name_to_key.get(model_name, model_name), {})
        threshold_default = float(m.get("threshold", 0.5))

    threshold = st.slider("Decision threshold", 0.0, 1.0, threshold_default, 0.01)
    
    submit = st.form_submit_button("🔍 Predict CVD Risk", use_container_width=True)
    
    if submit:
        if not models or not model_name:
            st.warning("No model available. Please train and save models to the `models/` folder.")
        else:
            pipeline = models[model_name]
            # Build input in the exact feature schema used for training
            feature_row = {
                "General_Health": general_health,
                "Checkup": checkup,
                "Exercise": exercise,
                "Smoking_History": smoking,
                "Alcohol_Consumption": alcohol,
                "Fruit_Consumption": fruit,
                "Green_Vegetables_Consumption": greens,
                "FriedPotato_Consumption": fried_potato,
                "Sex": gender,
                "Age_Category": age_category,
                "BMI": float(bmi),
                "Diabetes": diabetes,
                "Depression": depression,
                "Arthritis": arthritis,
            }
            X_input = pd.DataFrame([feature_row])
            try:
                prob = float(pipeline.predict_proba(X_input)[:, 1][0])
                st.metric("Estimated CVD Risk", f"{prob*100:.1f}%")
                label = "CVD" if prob >= threshold else "No CVD"
                st.success(f"Classification (threshold {threshold:.2f}): {label}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# Footer
st.markdown("---")
st.caption("💡 Cardiovascular Disease Analytics Tool | Built with Streamlit")
