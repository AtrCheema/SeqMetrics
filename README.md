
# SeqMetrics: a unified library for performance metrics calculation in Python

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/d11c4520bf514a3094dc2c13659d0bc5)](https://www.codacy.com/gh/AtrCheema/SeqMetrics/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=AtrCheema/SeqMetrics&amp;utm_campaign=Badge_Grade)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![HitCount](http://hits.dwyl.com/AtrCheema/SeqMetrics.svg)](http://hits.dwyl.com/AtrCheema/SeqMetrics)
[![Downloads](https://pepy.tech/badge/SeqMetrics)](https://pepy.tech/project/SeqMetrics)
[![Documentation Status](https://readthedocs.org/projects/seqmetrics/badge/?version=latest)](https://seqmetrics.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/SeqMetrics.svg)](https://badge.fury.io/py/SeqMetrics)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/SeqMetrics)](https://pypi.org/project/SeqMetrics/)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/AtrCheema/SeqMetrics)
![GitHub contributors](https://img.shields.io/github/contributors/AtrCheema/SeqMetrics)
![GitHub last commit (branch)](https://img.shields.io/github/last-commit/AtrCheema/SeqMetrics/master)
[![Zenodo](https://zenodo.org/badge/251072512.svg)](https://zenodo.org/doi/10.5281/zenodo.12958901)

The purpose of this repository to collect various classification and regression 
performance metrics or errors which can be calculated for time-series/sequential/tabular data, 
at one place. Currently only 1-dimensional data is supported.

## How to Install

You can install SeqMetrics using `pip`

    pip install SeqMetrics

or using GitHub link for the latest code

	python -m pip install git+https://github.com/AtrCheema/SeqMetrics.git

or using setup file, go to folder where repo is downloaded

    python setup.py install

You can also install SeqMetrics with all of its dependencies by making use of ``all`` option

    pip install SeqMetrics[all]

This will install [scipy](https://scipy.org) and [easy_mpl](https://github/com/Sara-Iftikhar/easy_mpl) libraries. 
The scipy library is used to calculate some
additional metrics such as [kendall_tau](https://seqmetrics.readthedocs.io/en/latest/rgr.html#SeqMetrics.RegressionMetrics.kendall_tau) 
or [mape_for_peaks]() while easy_mpl is used for plotting purpose.

## How to Use
SeqMetrics provides a uniform API for calculation of both regression and classification metrics.
It has a functional API and a class based API.

### Regression Metrics

The use of the functional API is as straightforward as calling the required function 
and providing it with true and predicted arrays or array-like objects (lists, tuples, DataFrames, Series, tensors).

```python
import numpy as np
from SeqMetrics import nse

true = np.random.random((20, 1))
pred = np.random.random((20, 1))

nse(true, pred)   # calculate Nash Sutcliff efficiency

```

The method for calling functions is consistent across all 100+ metrics. 

Alternatively, the same outcome can be achieved using a class-based API.

```python
import numpy as np
from SeqMetrics import RegressionMetrics

true = np.random.random((20, 1))
pred = np.random.random((20, 1))

er = RegressionMetrics(true, pred)

for m in er.all_methods: print("{:20}".format(m)) # get names of all availabe methods

er.nse()   # calculate Nash Sutcliff efficiency

er.calculate_all(verbose=True)  # or calculate errors using all available methods 
```

We can visualize the calcuated performance metrics if we have [easy_mpl](https://easy-mpl.readthedocs.io) package installed.
```python
import numpy as np
from SeqMetrics import RegressionMetrics, plot_metrics

np.random.seed(313)
true = np.random.random((20, 1))
pred = np.random.random((20, 1))

er = RegressionMetrics(true, pred)

plot_metrics(er.calculate_all(),  color="Blues")
```

<p float="left">
  <img src="/docs/source/imgs/reg1.png" width="500" />
  <img src="/docs/source/imgs/reg2.png" width="500" />
</p>

<p float="left">
  <img src="/docs/source/imgs/reg3.png" width="500" />
  <img src="/docs/source/imgs/reg4.png" width="500" />
</p>


`RegressionMetrics` currently, calculates following performane metrics for regression.

| Name                          | Name in this repository  |
| -------------------------- | ------------- |
| Absolute Percent Bias | `abs_pbias` |
| Agreement Index | `agreement_index` |
| Aitchison Distance | `aitchison` |
| Alpha decomposition of the NSE | `nse_alpha` |
| Anomaly correction coefficient | `acc` |
| Bias | `bias` |
| Beta decomposition of NSE | `nse_beta` |
| Bounded NSE | `nse_bound` |
| Bounded KGE | `kge_bound` |
| Brier Score | `brier_score` |
| Correlation Coefficient | `corr_coeff` |
| Coefficient of Determination | `r2` |
| Centered Root Mean Square Deviation | `centered_rms_dev` |
| Covariances | `covariance` |
| Decomposed Mean Square Error | `decomposed_mse` |
| Explained variance score | `exp_var_score` |
| Euclid Distance | `euclid_distance` |
| Geometric Mean Difference | `gmaen_diff` |
| Geometric Mean Absolute Error | `gmae` |
| Geometric Mean Relative Absolute Error | `gmrae` |
| Inertial Root Squared Error | `irmse` |
| Integral Normalized Root Squared Error | `inrse` |
| Inter-percentile Normalized Root Mean Squared Error | `nrmse_ipercentile` |
| Jensen-shannon divergence | `JS` |
| Kling-Gupta Efficiency | `kge` |
| Legate-McCabe Efficiency Index | `lm_index` |
| Logrithmic Nash Sutcliff Efficiency | `log_nse` |
| Logrithmic probability distribution | `log_prob` |
| maximum error | `max_error` |
| Mean Absolute Error | `mae` |
| Mean Absolute Percentage Deviation | `mapd` |
| Mean Absolute Percentage Error | `mape` |
| Mean Absolute Relative Error | `mare` |
| Mean Absolute Scaled Error | `mase` |
| Mean Arctangle Absolute Percentage Error | `maape` |
| Mean Bias Error | `mean_bias_error` |
| Mean Bounded relative Absolute Error | `mbrae` |
| Mean Errors | `me` |
| Mean Gamma Deviances | `mean_gamma_deviance` |
| Mean Log Error | `mle` |
| Mean Normalized Root Mean Square Error | `nrmse_mean` |
| Mean Percentage Error | `mpe` |
| Mean Poisson Deviance | `mean_poisson_deviance` |
| Mean Relative Absolute Error | `mrae` |
| Mean Square Error | `mse` |
| Mean Square Logrithmic Errors | `mean_square_log_error` |
| Mean Variance | `mean_var` |
| Median Absolute Error | `median_abs_error` |
| Median Absolute Percentage Error | `mdape` |
| Median Dictionary Accuracy | |
| Median Error | `mde` |
| Median Relative Absolute Error | `mdrae` |
| Median Squared Error | `med_seq_error` |
| Mielke-Berry R | `mb_r` |
| Modified Agreement of Index | `mod_agreement_index` |
| Modified Kling-Gupta Efficiency | `kge_mod` |
| Modified Nash-Sutcliff Efficiency | `nse_mod` |
| Nash-Sutcliff Efficiency | `nse` |
| Non parametric Kling-Gupta Efficiency | `kge_np` |
| Normalized Absolute Error | `norm_ae` |
| Normalized Absolute Percentage Error | `norm_ape` |
| Normalized Euclid Distance | `norm_euclid_distance` |
| Normalized Root Mean Square Error | `nrmse` |
| Peak flow bias of the flow duration curve | `fdc_fhv` |
| Pearson correlation coefficient | `person_r` |
| Percent Bias | `pbias` |
| Range Normalized root mean square | `nrmse_range` |
| Refined Agreement of Index | `ref_agreement_index` |
| Relative Agreement of Index | `rel_agreement_index` |
| Relative Absolute Error | `rae` |
| Relative Root Mean Squared Error | `relative_rmse` |
| Relative Nash-Sutcliff Efficiency | `nse_rel` |
| Root Mean Square Errors | `rmse` |
| Root Mean Square Log Error | `rmsle` |
| Root Mean Square Percentage Error | `rmspe` |
| Root Mean Squared Scaled Error | `rmsse` |
| Root Median Squared Scaled Error | `rmsse` |
| Root Relative Squared Error | `rrse` |
| RSR | `rsr` |
| Separmann correlation coefficient | `spearmann_corr` |
| Skill Score of Murphy | `skill_score_murphy` |
| Spectral Angle | `sa` |
| Spectral Correlation | `sc` |
| Spectral Gradient Angle | `sga` |
| Spectral Information Divergence | `sid` |
| Symmetric kullback-leibler divergence | `KLsym` |
| Symmetric Mean Absolute Percentage Error | `smape` |
| Symmetric Median Absolute Percentage Error | `smdape` |
| sum of squared errors | `sse` | 
| Volume Errors | `volume_error` |
| Volumetric Efficiency | `ve` |
| Unscaled Mean Bounded Relative Absolute Error | `umbrae` |
| Watterson's M | `watt_m` |
| Weighted Mean Absolute Percent Errors | `wmape` |
| Weighted Absolute Percentage Error | `wape` |

### Classification Metrics

The API is same for performance metrics of classification problem.

```python
import numpy as np
from SeqMetrics import ClassificationMetrics

# boolean array

t = np.array([True, False, False, False])
p = np.array([True, True, True, True])
metrics = ClassificationMetrics(t, p)
print(metrics.calculate_all())

# binary classification with numerical labels

true = np.array([1, 0, 0, 0])
pred = np.array([1, 1, 1, 1])
metrics = ClassificationMetrics(true, pred)
print(metrics.calculate_all())

# multiclass classification with numerical labels

true = np.random.randint(1, 4, 100)
pred = np.random.randint(1, 4, 100)
metrics = ClassificationMetrics(true, pred)
print(metrics.calculate_all())

# You can also provide logits instead of labels.

predictions = np.array([[0.25, 0.25, 0.25, 0.25],
                       [0.01, 0.01, 0.01, 0.96]])
targets = np.array([[0, 0, 0, 1],
                    [0, 0, 0, 1]])
metrics = ClassificationMetrics(targets, predictions, multiclass=True)
print(metrics.calculate_all())

# Working with categorical values is seamless

true = np.array(['a', 'b', 'b', 'b']) 
pred = np.array(['a', 'a', 'a', 'a'])
metrics = ClassificationMetrics(true, pred)
print(metrics.calculate_all())

# same goes for multiclass categorical labels

t = np.array(['car', 'truck', 'truck', 'car', 'bike', 'truck'])
p = np.array(['car', 'car',   'bike',  'car', 'bike', 'truck'])
metrics = ClassificationMetrics(targets, predictions, multiclass=True)
print(metrics.calculate_all())
```

SeqMetrics library currently calculates following performance metrics
of classification.

| Name                          | Name in this repository  |
| -------------------------- | ------------- |
| Accuracy | `accuracy` |
| Balanced Accuracy | `balanced_accuracy` |
| Error Rate | `error_rate` |
| Recall | `recall` |
| Precision | `precision` |
| F1 score | `f1_score` |
| F2 score | `f2_score` |
| Specificity | `specificity` |
| Cross Entropy | `cross_entropy` |
| False Positive Rate | `false_positive_rate` |
| False Negative Rate | `false_negative_rate` |
| False Discovery Rate | `false_discovery_rate` |
| False Omission Rate | `false_omission_rate` |
| Negative Predictive Value | `negative_predictive_value` |
| Positive Likelihood Ratio | `positive_likelihood_ratio` |
| Negative Likelihood Ratio | `negative_likelihood_ratio` |
| Prevalence Threshold | `prevalence_threshold` |
| Youden Index | `youden_index` |
| Confusion Matrix | `confusion_matrix` |
| Fowlkes Mallows Index | `fowlkes_mallows_index` |
| Mathews correlation Coefficient | `mathews_corr_coeff` |

## Web App
The SeqMetrics library is available from the webapp which is deployed
used stream [https://seqmetrics.streamlit.app/](https://seqmetrics.streamlit.app/)

You can also launch the app locally if you do not wish to use the web-based app. 
Make sure you follow the below steps

    git clone https://github.com/AtrCheema/SeqMetrics.git
    cd SeqMetrics
    pip install requirements.txt
    pip install streamlit
    streamlit run app.py

Usage of streamlit based application app involves, 1) providing the true and predicted arrays
either by pasting the data in the boxes or by uploading a file, 2) Selecting the 
relevant performance metric and 3) calculating the performance metric. These steps are further
illustrated below.

<p float="left">
  <img src="/paper/fig2.jpg"/>
</p>

The method to provide data from a (excel/csv) file is described in below image

<p float="left">
  <img src="/paper/fig3.jpg"/>
</p>

## Related

[forecasting_metrics](https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9)

[hydroeval](https://github.com/ThibHlln/hydroeval)

[SkillMetrics](https://github.com/PeterRochford/SkillMetrics)

[HydroErr](https://github.com/BYU-Hydroinformatics/HydroErr)

[Keras](https://github.com/keras-team/keras.git)

[SickitLearn](https://github.com/scikit-learn/scikit-learn.git)

[Torchmetrics](https://github.com/Lightning-AI/torchmetrics.git)