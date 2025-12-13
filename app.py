"""
Streamlit Visual Analytics Tool for Cardiovascular Disease Dataset
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

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
    # ML Model Summary (Placeholder)
    st.header("🤖 Machine Learning Model")
    
    st.info("""
    **Machine Learning Model Summary** (Placeholder)
    
    Once the ML model is trained, this section will display:
    - Algorithm used
    - Performance metrics (Accuracy, Precision, Recall, ROC-AUC)
    - Feature importance
    - Preprocessing steps
    """)

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
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    
    with col2:
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0)
    
    st.subheader("Health Metrics")
    col3, col4 = st.columns(2)
    with col3:
        general_health = st.selectbox("General Health", 
            ["Excellent", "Very Good", "Good", "Fair", "Poor"])
        exercise = st.selectbox("Regular Exercise", ["Yes", "No"])
    
    with col4:
        diabetes = st.selectbox("Diabetes", ["Yes", "No"])
        smoking = st.selectbox("Smoking History", ["Yes", "No"])
    
    submit = st.form_submit_button("🔍 Predict CVD Risk", use_container_width=True)
    
    if submit:
        st.warning("""
        ⚠️ **Model Not Yet Available**
        
        The prediction model is still being trained. Once ready, your CVD risk will be displayed here.
        """)
        
        # Placeholder for future prediction
        # risk = model.predict_proba(features)[0, 1]
        # st.metric("Estimated CVD Risk", f"{risk*100:.1f}%")

# Footer
st.markdown("---")
st.caption("💡 Cardiovascular Disease Analytics Tool | Built with Streamlit")
