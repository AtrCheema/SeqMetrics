
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/d11c4520bf514a3094dc2c13659d0bc5)](https://www.codacy.com/gh/AtrCheema/SeqMetrics/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=AtrCheema/SeqMetrics&amp;utm_campaign=Badge_Grade)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![HitCount](http://hits.dwyl.com/AtrCheema/SeqMetrics.svg)](http://hits.dwyl.com/AtrCheema/SeqMetrics)
[![Downloads](https://pepy.tech/badge/SeqMetrics)](https://pepy.tech/project/SeqMetrics)
[![Documentation Status](https://readthedocs.org/projects/seqmetrics/badge/?version=latest)](https://seqmetrics.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/SeqMetrics.svg)](https://badge.fury.io/py/SeqMetrics)

The purpose of this repository to collect various classification and regression 
performance metrics or errors which can be calculated for time-series/sequential/tabular data, 
at one place. Currently only 1-dimensional data is supported.

## How to Install

using `pip`

    pip install SeqMetrics

or using github link for the latest code

	python -m pip install git+https://github.com/AtrCheema/SeqMetrics.git

or using setup file, go to folder where repo is downloaded

    python setup.py install


## How to Use

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

The API is same for classification performance metrics.
```python
import numpy as np
from SeqMetrics import ClassificationMetrics

# boolean array

t = np.array([True, False, False, False])
p = np.array([True, True, True, True])
metrics = ClassificationMetrics(t, p)
accuracy = metrics.accuracy()

# binary classification with numerical labels

true = np.array([1, 0, 0, 0])
pred = np.array([1, 1, 1, 1])
metrics = ClassificationMetrics(true, pred)
accuracy = metrics.accuracy()

# multiclass classification with numerical labels

true = np.random.randint(1, 4, 100)
pred = np.random.randint(1, 4, 100)
metrics = ClassificationMetrics(true, pred)
accuracy = metrics.accuracy()
```


## RegressionMetrics

Currently following regression performance metrics are being calculated.

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

## Related

[forecasting_metrics](https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9)

[hydroeval](https://github.com/ThibHlln/hydroeval)

[SkillMetrics](https://github.com/PeterRochford/SkillMetrics)

[HydroErr](https://github.com/BYU-Hydroinformatics/HydroErr)