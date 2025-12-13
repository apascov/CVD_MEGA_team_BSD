Visual Analytics Tool for Cardiovascular Disease Dataset

This project implements an interactive Streamlit web application for visualizing and predicting cardiovascular disease (CVD) risk. It is designed as a horizontally structured analytics dashboard that can be deployed on Streamlit Community Cloud or run locally.

The application integrates:

Exploratory visual analytics based on functions from the provided UE6_Bea.ipynb notebook.

Dataset description and summary statistics.

Machine learning model overview (placeholder until ML results are finalized).

User input → model prediction: users can enter their health parameters to estimate CVD risk using a trained model.

project/
│
├── app.py                      # Streamlit main file
├── data/
│     └── CVD_cleaned.csv       # cleaned cardiovascular dataset
├── models/
│     └── model.pkl             # trained ML model (placeholder for now)
├── visualizations/
│     ├── vis1.py               # extracted VISUALIZATION 1 from UE6_Bea.ipynb
│     ├── vis2.py               # extracted VISUALIZATION 2
│     └── vis3.py               # extracted VISUALIZATION 3
└── README.md

2. Application Layout Overview

The Streamlit app divides the page into two vertical columns, each splitting into upper and lower parts.

------------------------------------------------------------
|                        LEFT COLUMN                       |
| -------------------------------------------------------- |
| (Upper Left, 15%) Dataset overview, explanation          |
| -------------------------------------------------------- |
| (Lower Left, 85%) Interactive visualizations             |
------------------------------------------------------------
|                        RIGHT COLUMN                      |
| -------------------------------------------------------- |
| (Upper Right) ML model summary (placeholder)             |
| -------------------------------------------------------- |
| (Lower Right) User-input form + CVD risk prediction      |
------------------------------------------------------------


This layout is implemented using:

left_col, right_col = st.columns([1, 1])
with left_col:
    top_left = st.container()
    bottom_left = st.container()
with right_col:
    top_right = st.container()
    bottom_right = st.container()


No custom CSS is required.

3. Dataset Description

The application loads the cleaned cardiovascular dataset:

data/CVD_cleaned.csv


A caching layer accelerates interactions:

@st.cache_data
def load_data():
    return pd.read_csv("data/CVD_cleaned.csv")


The upper-left panel contains:

A short general description of the dataset

Number of samples and attributes

Basic statistics (mean age, proportion of CVD cases, etc.)

Context needed for interpretation of the visualizations

4. Visualizations (Lower Left Panel)

Three visualization functions must be extracted from the notebook UE6_Bea.ipynb.
They should be rewritten as simple Python functions inside the visualizations/ folder:

vis1.py — VISUALIZATION 1

vis2.py — VISUALIZATION 2

vis3.py — VISUALIZATION 3

The app provides:

Visualization Selector
vis_choice = st.radio(
    "Choose Visualization",
    ["Visualization 1", "Visualization 2", "Visualization 3"]
)

User-controlled Parameters

Each visualization receives arguments through Streamlit widgets:

var = st.selectbox("Choose variable", df.columns)
fig = plot_vis1(df, var)
st.pyplot(fig)


The functions may use matplotlib, seaborn, or plotly — Streamlit supports all.

5. Machine Learning Summary (Upper Right Panel)

This section will eventually contain a concise description of:

Which algorithm was trained

Performance metrics (accuracy, recall, ROC AUC)

Any preprocessing steps (scaling, encoding, etc.)

A textual explanation of why the model was selected

For now, it is a static placeholder:

top_right.info("Machine Learning Model Summary will appear here.")


After ML development is complete, replace this with real metrics and plots.

6. User Input Prediction (Lower Right Panel)

A form allows users to enter their health parameters:

with st.form("pred_form"):
    age = st.number_input("Age", 18, 100)
    gender = st.selectbox("Gender", ["Male", "Female"])
    cholesterol = st.number_input("Cholesterol", 100, 400)
    # ...
    submit = st.form_submit_button("Predict")


The model should be saved beforehand as:

models/model.pkl


and loaded in the app:

import joblib
model = joblib.load("models/model.pkl")


Prediction is displayed using:

risk = model.predict_proba(features)[0, 1]
st.metric("Estimated CVD Risk", f"{risk*100:.1f}%")

7. Running the Application Locally
Prerequisites

Python ≥ 3.9
Install dependencies:

pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib

Run Streamlit
streamlit run app.py


Open the provided URL in your browser.

8. Deployment Instructions (Vercel or Streamlit Cloud)
Option A — Streamlit Community Cloud (simplest)

Push repo to GitHub

Go to https://share.streamlit.io

Select your repo → choose app.py

Deploy

Option B — Vercel

Vercel requires:

A Python runtime

A vercel.json config

The app entrypoint exposed as a web server

A minimal vercel.json:

{
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ]
}


Streamlit will start as a server inside Vercel’s container; make sure all files are inside the repo.