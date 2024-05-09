
import importlib
from io import StringIO
from typing import Union

import numpy as np
import streamlit as st
import pandas as pd

from SeqMetrics import RegressionMetrics
from SeqMetrics import ClassificationMetrics


SM = importlib.import_module('SeqMetrics')
PRED_ALIASES = ['Predicted', 'PREDICTED', 'predicted', 'pred', 'Pred', 'PRED',
                'Calculated', 'Simulated', 'Simulation', 'calculated', 'response value'
                ]

TRUE_ALIASES = ['true', 'observed', 'True', 'TRUE', 'Observed', 'OBSERVED'
                ]

NAME_REG = {'calculate_all' : 'Calculate All Regression Metrics',
            'calculate_hydro_metrics' : 'Calculates All Metrics for Hydrological Data.',
            'adjusted_r2' : 'Adjusted R2',
            'agreement_index': 'Agreement Index',
            'abs_pbias': 'Absolute Percent Bias',
            'amemiya_pred_criterion': 'Amemiya’s Prediction Criterion',
            'amemiya_adj_r2': 'Amemiya’s Adjusted R-squared',
            'aitchison': 'Aitchison Distance',
            'aic': 'Akaike Information Criterion',
            'acc': 'Anomaly Correction Coefficient',
            'bias': 'Bias',
            'brier_score': 'Brier Score (BS)',
            'bic': 'Bayesian Information Criterion',
            'corr_coeff': 'Pearson Correlation Coefficient',
            'covariance': 'Covariance',
            'cronbach_alpha': 'Cronbach Alpha',
            'cosine_similarity': 'Cosine Similarity',
            'centered_rms_dev': 'Centered Root-Mean-Square (RMS) Deviation',
            'decomposed_mse': 'Decomposed MSE',
            'euclid_distance': 'Euclidian Distance',
            'exp_var_score': 'Explained Variance Score',
            'expanded_uncertainty': 'Expanded Uncertainty',
            'fdc_fhv': 'fdc_fhv',
            'fdc_flv': 'fdc_flv',
            'gmean_diff': 'Geometric Mean Difference',
            'gmrae': 'Geometric Mean Relative Absolute Error',
            'gmae': 'Geometric Mean Absolute Error',
            'inrse': 'Integral Normalized Root Squared Error',
            'irmse': 'Inertial RMSE',
            'JS': 'Jensen-Shannon Divergence',
            'kge' : 'Kling-Gupta Efficiency',
            'kge_bound': 'Bounded Kling-Gupta Efficiency',
            'kge_mod': 'Modified Kling-Gupta Efficiency',
            'kge_np': 'Non parametric Kling-Gupta Efficiency',
            'kendaull_tau': 'Kendall’s tau',
            'kgeprime_bound': 'Bounded Version of the Modified Kling-Gupta Efficiency',
            'kgenp_bound': 'Bounded Version of the Non-Parametric Kling-Gupta Efficiency',
            'kl_sym': 'Symmetric Kullback-leibler Divergence',
            'lm_index': 'Legate-McCabe Efficiency Index',
            'log_prob': 'Logarithmic Probability Distribution',
            'log_nse': 'Log Nash-Sutcliffe model Efficiency',
            'maape': 'Mean Arctangent Absolute Percentage Error',
            'mbrae': 'Mean Bounded Relative Absolute Error',
            'max_error': 'Maximum Absolute Error',
            'mb_r': 'Mielke-Berry R value',
            'mda': 'Mean Directional Accuracy',
            'mde': 'Median Error',
            'mdape': 'Median Absolute Percentage Error',
            'mdrae': 'Median Relative Absolute Error',
            'me': 'Mean Error',
            'mean_bias_error': 'Mean Bias Error',
            'mean_var': 'Mean Variance',
            'mean_poisson_deviance': 'Mean Poisson Deviance',
            'mean_gamma_deviance': 'Mean Gamma Deviance',
            'median_abs_error': 'Median Absolute Error',
            'mle': 'Mean Log Error',
            'mod_agreement_index': 'Modified Agreement of Index',
            'mpe': 'Mean Percentage Error',
            'mrae': 'Mean Relative Absolute Error',
            'mape': 'Mean Absolute Percentage Error',
            'med_seq_error': 'Median Squared Error',
            'mae': 'Mean Absolute Error',
            'mase': 'Mean Absolute Scaled Error',
            'mare': 'Mean Absolute Relative Error',
            'msle': 'Mean Square Logrithmic Error',
            'mapd': 'Mean Absolute Percentage Deviation',
            'nrmse': 'Normalized Root Mean Squared Error',
            'nse': 'Nash-Sutcliff Efficiency',
            'nse_alpha': 'NSE Alpha',
            'nse_beta': 'NSE Beta',
            'nse_mod': 'NSE Mod',
            'nse_rel': 'Relative NSE',
            'nse_bound': 'NSE Bound',
            'norm_euclid_distance': 'Normalized Euclidian Distance',
            'nrmse_range': 'Range Normalized Root Mean Squared Error',
            'nrmse_ipercentile': 'NRMSE Ipercentile',
            'nrmse_mean': 'Mean Normalized RMSE',
            'norm_ae': 'Normalized Absolute Error',
            'norm_ape': 'Normalized Absolute Percentage Error',
            'pbias': 'Percent Bias',
            'r2' : 'R-Squared',
            'r2_score' : 'R-Squared Score',
            'rmse' : 'Root Mean Squared Error',
            'rmsle' : 'Root Mean Square Log Error',
            'rmdspe': 'Root Median Squared Percentage Error',
            'rse': 'Relative Squared Error',
            'rrse': 'Root Relative Squared Error',
            'rae': 'Relative Absolute Error',
            'ref_agreement_index': 'Refined Index of Agreement',
            'rel_agreement_index': 'Relative Index of Agreement',
            'relative_rmse': 'Relative Root Mean Squared Error',
            'rmspe': 'Root Mean Square Percentage Error',
            'rsr': 'RSR',
            'rmsse': 'Root Mean Squared Scaled Error',
            'sse' : 'Sum of Squared Errors',
            'sa' : 'Spectral Angle',
            'sc' : 'Spectral Correlation',
            'smape' : 'Symmetric Mean Absolute Percentage Error',
            'smdape' : 'Symmetric Median Absolute Percentage Error',
            'sid' : 'Spectral Information Divergence',
            'skill_score_murphy' : 'Skill Score Murphy',
            'std_ratio' : 'Ratio of Standard Deviations',
            'spearmann_corr': 'Spearmann Correlation Coefficient',
            'sga': 'Spectral Gradient Angle',
            'umbrae' : 'Unscaled Mean Bounded Relative Absolute Error',
            've' : 'Volumetric Efficiency',
            'volume_error' : 'Volume Error',
            'wape' : 'Weighted Absolute Percentage Error',
            'watt_m' : 'Watterson’s M. Refrence',
            'wmape' : 'Weighted Mean Absolute Percent Error',
            'mre': "Mean Relative Error",
            }

NAME_CLS = {'accuracy' : 'Accuracy',
            'balanced_accuracy': 'Balanced Accuracy',
            'confusion_matrix': 'Confusion Matrix',
            'cross_entropy': 'Cross Entropy',
            'calculate_all' : 'Calculate All Classification Metrics',
            'error_rate' : 'Error Rate',
            'false_positive_rate' : 'False Positive Rate',
            'false_negative_rate' : 'False Negative Rate',
            'false_discovery_rate' : 'false Discovery Rate',
            'false_omission_rate' : 'False Omission Rate',
            'f1_score' : 'f1 Score',
            'f2_score': 'f2 Score',
            'fowlkes_mallows_index': 'Fowlkes–Mallows Index',
            'mathews_corr_coeff' : 'Methews Correlation Coefficient',
            'negative_likelihood_ratio': 'Negative Likelihood Ratio',
            'negative_predictive_value' : 'Negative Predictive Value',
            'precision' : 'Precision',
            'positive_likelihood_ratio' : 'Positive Likelihood Ratio',
            'prevalence_threshold' : 'Prevalence Threshold',
            'recall' : 'Recall',
            'specificity' : 'Specificity',
            'youden_index' : 'Youden Index'
            }

def calculate_metrics(
        true:Union[list, np.ndarray, pd.Series],
        predicted:Union[list, np.ndarray, pd.Series],
        metric_name:str,
        task_type:str):
    doc_string = getattr(SM, metric_name).__doc__.split('Parameters')[0]
    if task_type == "regression":
        metric_function = getattr(RegressionMetrics(true, predicted), metric_name)
        calc_value = metric_function()

        return calc_value, doc_string
    else:
        metric_function = getattr(ClassificationMetrics(true, predicted), metric_name)
        calc_value = metric_function()

        return calc_value, doc_string


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
st.markdown(f"""
    <style>
    .main {{
        background-color: #F6D1DA;
        border-radius: 5px;
        padding: 10px;
    }}
    .heading {{
        text-align: center;
        font-family: 'Arial', sans-serif;
        font-size: 30px;
        font-weight: bold;
        color: #4a4a4a;
    }}
    .subtitle {{
        text-align: center;
        font-style: italic;
        font-size: 18px;
        margin-bottom: 20px;
        color: #4a4a4a;
    }}
    .stLabel {{
        font-size: 16px;
        font-weight: bold;
        color: #333;
    }}
    .stTextInput>div>div>input {{
        font-size: 14px;
    }}
    .stButton>button {{
        width: 25%;
        border-radius: 20px;
        font-size: 16px;
        font-weight: bold;
        background-color: #A28089;
    }}
    .stButton {{
        display: flex;
        justify-content: center;
    }}
    .upload-label {{
        font-size: 14px;
        color: #333;
        margin-bottom: 5px;
    }}
    .or-separator {{
        text-align: center;
        margin-top: 10px;
        margin-bottom: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="heading">Metrics Calculator</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Effortlessly calculate a variety of metrics for both regression and classification.</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="upload-label">Provide the observed/true data by either typing/pasting or by uploading a file:</div>', unsafe_allow_html=True)
    true_values = st.text_area("True Values", placeholder="Type the observed/true data as comma or space separated. You can copy from Excel, text or any other file", height=100, key="true")
    st.markdown('<div class="or-separator">OR</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-label">Upload CSV or Excel file with one column as true:</div>', unsafe_allow_html=True)
    uploaded_true = st.file_uploader("", key="upload_true")
    # Map function names to friendly names for regression metrics
    reg_metrics_available = [func for func in dir(RegressionMetrics) if func in NAME_REG
                             and not func.startswith("_")]
    reg_options = {None: "None"}  # Add None as the first option
    reg_options.update({func: NAME_REG[func] for func in reg_metrics_available})
    regression_metric = st.selectbox("Select Regression Metric", options=list(reg_options.values()), index=0,
                                     key="reg_metric")

with col2:
    st.markdown('<div class="upload-label">Provide the predicted/simulated data by either typing/pasting or by uploading a file:</div>', unsafe_allow_html=True)
    predicted_values = st.text_area("Predicted data", placeholder="Type the predicted/simulated data as comma or space separated. You can copy from Excel, text or any other file", height=100, key="predicted")
    st.markdown('<div class="or-separator">OR</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-label">Upload CSV or Excel file with one column as predicted:</div>', unsafe_allow_html=True)
    uploaded_pred = st.file_uploader("", key="upload_pred")
    # Map function names to friendly names for classification metrics
    cls_metrics_available = [func for func in dir(ClassificationMetrics) if
                             func in NAME_CLS and not func.startswith("_")]
    cls_options = {None: "None"}  # Add None as the first option
    cls_options.update({func: NAME_CLS[func] for func in cls_metrics_available})
    classification_metric = st.selectbox("Select Classification Metric", options=list(cls_options.values()),
                                         index=0,
                                         key="cls_metric")

if st.button("Calculate"):
    error = False  # Flag to indicate if there's an error
    if uploaded_true:
        #process true values
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
    if len(true_values) != len(predicted_values):
        st.error(f"The number of true values are {len(true_values)} and the number of predicted "
                 f"values are {len(predicted_values)}. These must be equal. Please re-enter the inputs.")
        error = True

    if not error:
        # Find the actual function name for the selected metric
        reg_func_name = next((k for k, v in reg_options.items() if v == regression_metric), None)
        cls_func_name = next((k for k, v in cls_options.items() if v == classification_metric), None)

        if reg_func_name:
            calc_value, doc_string = calculate_metrics(true_values, predicted_values, reg_func_name, "regression")
            st.subheader(f"{regression_metric}")
            st.markdown(f"**Documentation:** {doc_string}")
            st.write(calc_value)
        elif cls_func_name:
            calc_value, doc_string = calculate_metrics(true_values, predicted_values, cls_func_name, "classification")
            st.subheader(f"{classification_metric}")
            st.markdown(f"**Documentation:** {doc_string}")
            st.write(calc_value)
