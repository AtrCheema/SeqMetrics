
from typing import Union

import numpy as np
import streamlit as st
import pandas as pd
from io import StringIO
from SeqMetrics import RegressionMetrics
from SeqMetrics import ClassificationMetrics


PRED_ALIASES = ['Predicted', 'PREDICTED', 'predicted', 'pred', 'Pred', 'PRED',
                'Calculated', 'Simulated', 'Simulation', 'calculated', 'response value'
                ]

TRUE_ALIASES = ['true', 'observed', 'True', 'TRUE', 'Observed', 'OBSERVED'
                ]

def calculate_metrics(
        true:Union[list, np.ndarray, pd.Series],
        predicted:Union[list, np.ndarray, pd.Series],
        metric_name:str,
        task_type:str):
    if task_type == "regression":
        # Call the appropriate function from SeqMetrics based on the selected metric
        return getattr(RegressionMetrics(true, predicted), metric_name)()

    elif task_type == "classification":
        # Call the appropriate function from SeqMetrics for classification metrics
        return getattr(ClassificationMetrics(true, predicted), metric_name)()


def process_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.xls') or uploaded_file.name.endswith('.xlsx'):
                # Directly reading from the buffer for Excel files
                return pd.read_excel(uploaded_file)
            else:
                # Using StringIO for CSV as it's text-based
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                return pd.read_csv(stringio)
        except Exception as e:
            print(f"Error processing file: {e}")
            return None
    return None


# Set page configuration - this should be the first command
st.set_page_config(page_title="Metrics Calculator", layout="wide")

# Custom CSS for styling
background_url = "back.png"
st.markdown(f"""
    <style>
    body {{
        background-image: url("back.png");
        background-size: cover;
    }}
    .main {{
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 5px;
        padding: 10px;
    }}
    .heading {{
        text-align: center;
    }}
    .stButton>button {{
        width: 25%;
        border-radius: 20px;
        font-size: 16px;
        font-weight: bold;
    }}
    .stButton {{
        display: flex;
        justify-content: center;
    }}
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="heading">Metrics Calculator</h1>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    true_values = st.text_area("True Values", height=100, key="true")
    uploaded_true = st.file_uploader("Upload CSV for True Values", key="upload_true")
    regression_metrics = [None]+[func for func in dir(RegressionMetrics) if not func.startswith("_")]
    regression_metric = st.selectbox("Select Regression Metric", options=regression_metrics, index= 0, key="reg_metric")

with col2:
    predicted_values = st.text_area("Predicted Values", height=100, key="predicted")
    uploaded_pred = st.file_uploader("Upload CSV for Predicted Values", key="upload_pred")
    classification_metrics = [None]+[func for func in dir(ClassificationMetrics) if not func.startswith("_")]
    classification_metric = st.selectbox("Select Classification Metric", options=classification_metrics, index= 0, key="cls_metric")

if st.button("Calculate"):
    if uploaded_true:
        df_true = process_uploaded_file(uploaded_true)

        # Convert all column names to strings
        df_true.columns = df_true.columns.map(str)
        df_true.column = [str(col) for col in df_true.columns]
        colname = None
        for colname_ in TRUE_ALIASES:
            if colname_ in df_true.column:
                colname = colname_
                break
        if colname is None:
            raise ValueError(f"No column found with label True")
        true_values = df_true.loc[:, colname].tolist()
    else:
        if "," in true_values:
            true_values = [float(val) for val in true_values.split(',')]
        elif "\n" in true_values:
            true_values = [float(val) for val in true_values.split('\n')]
        else:
            true_values = [float(val) for val in true_values.split(' ')]

    if uploaded_pred:
        df_pred = process_uploaded_file(uploaded_pred)
        colname = None
        for colname in PRED_ALIASES:
            if colname in df_pred.columns:
                break
        if colname is None:
            raise ValueError(f"No column found with label Predicted ")
        predicted_values = df_pred.loc[:, colname].tolist()
    else:
        if ',' in predicted_values:
            predicted_values = [float(val) for val in predicted_values.split(',')]
        elif "\n" in predicted_values:
            predicted_values = [float(val) for val in predicted_values.split('\n')]
        else:
            predicted_values = [float(val) for val in predicted_values.split(' ')]

    if regression_metric:
        # Calculate Regression Metrics
        calc_value = calculate_metrics(true_values, predicted_values,
                                       regression_metric, "regression")
        st.subheader(f"{regression_metric}")
        st.write(calc_value)
    else:
        # Calculate Classification Metrics
        calc_value = calculate_metrics(true_values, predicted_values,
                                       classification_metric,
                                       "classification")
        st.subheader(f"{classification_metric}")
        st.write(calc_value)
