
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/72cbd07c421b42ffa4ecddeec71d103f)](https://app.codacy.com/gh/AtrCheema/TSErrors?utm_source=github.com&utm_medium=referral&utm_content=AtrCheema/TSErrors&utm_campaign=Badge_Grade)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![HitCount](http://hits.dwyl.com/AtrCheema/TSErrors.svg)](http://hits.dwyl.com/AtrCheema/TSErrors)

The purpose of this repository to collect various performance metrics or errors which can be
calculated for time-series/sequential data, at one place. Currently only 1d data is supported.

## How to Install

using `pip`

    pip install TSErrors

using github link for the latest code

	python -m pip install git+https://github.com/AtrCheema/TSErrors.git

using setup file, go to folder where repo is downloaded

    python setup.py install

## Errors

Currently following errors are being calculated.

| Name                          | Name in this repository  |
| -------------------------- | ------------- |
| Absolute Percent Bias | `abs_pbias` |
| Agreement Index | `agreement_index` |
| Aitchison Distance | `aitchison` |
| Alpha decomposition of the NSE | `nse_alpha` |
| Bias | `bias` |
| Beta decomposition of NSE | `nse_beta` |
| Bounded NSE | `nse_bound` |
| Bounded KGE | `kge_bound` |
| Correlation Coefficient | `corr_coeff` |
| Coefficient of Determination | `r2` |
| Covariances | `covariance` |
| Decomposed Mean Square Error | `decomposed_mse` |
| Explained variance score | `exp_var_score` |
| Geometric Mean Absolute Error | `gmae` |
| Geometric Mean Relative Absolute Error | `gmrae` |
| Integral Normalized Root Squared Error | `inrse` |
| Jensen-shannon divergence | `JS` |
| Kling-Gupta Efficiency | `kge` |
| Logrithmic Nash Sutcliff Efficiency | `log_nse` |
| Logrithmic probability distribution | `log_prob` |
| maximum error | `max_error` |
| Mean Absolute Error | `mae` |
| Mean Absolute Percentage Error | `mape` |
| Mean Absolute Relative Error | `mean_abs_rel_error` |
| Mean Absolute Scaled Error | `mase` |
| Mean Arctangle Absolute Percentage Error | `maape` |
| Mean Bias Error | `mean_bias_error` |
| Mean Bounded relative Absolute Error | `mbrae` |
| Mean Errors | `me` |
| Mean Gamma Deviances | `mean_gamma_deviance` |
| Mean Percentage Error | `mpe` |
| Mean Poisson Deviance | `mean_poisson_deviance` |
| Mean Relative Absolute Error | `mrae` |
| Mean Square Error | `mse` |
| Mean Square Logrithmic Errors | `mean_square_log_error` |
| Median Absolute Error | `median_abs_error` |
| Median Absolute Percentage Error | `mdape` |
| Median Dictionary Accuracy | |
| Median Relative Absolute Error | `mdrae` |
| Modified Kling-Gupta Efficienc | `kge_mod` |
| Nash-Sutcliff Efficiency | `nse` |
| Non parametric Kling-Gupta Efficiency | `kge_np` |
| Normalized Absolute Error | `norm_ae` |
| Normalized Absolute Percentage Error | `norm_ape` |
| Normalized Root Mean Square Error | `nrmse` |
| Peak flow bias of the flow duration curve | `fdc_fhv` |
| Percent Bias | `pbias` |
| Relative Absolute Error | `rae` |
| Relative Root Mean Squared Error | `relative_rmse` |
| Root Mean Square Errors | `rmse` |
| Root Mean Square Percentage Error | `rmspe` |
| Root Mean Squared Scaled Error | `rmsse` |
| Root Median Squared Scaled Error | `rmsse` |
| Root Relative Squared Error | `rrse` |
| RSR | `rsr` |
| Separmann correlation coefficient | `spearmann_corr` |
| Symmetric kullback-leibler divergence | `KLsym` |
| Symmetric Mean Absolute Percentage Error | `smape` |
| Symmetric Median Absolute Percentage Error | `smdape` |
| sum of squared errors | `sse` | 
| Volume Errors | `volume_error` |
| Unscaled Mean Bounded Relative Absolute Error | `umbrae` |
| Weighted Mean Absolute Percent Errors | `wmape` |
| Weighted Absolute Percentage Error | `wape` |

## How to Use

```python
import numpy as np
from TSErrors import FindErrors

true = np.random.random((20, 1))
pred = np.random.random((20, 1))

er = FindErrors(true, pred)

for m in er.all_methods: print("{:20}".format(m)) # get names of all availabe methods

er.nse()   # calculate Nash Sutcliff efficiency

er.calculate_all(verbose=True)  # or calculate errors using all available methods
er.stats(verbose=True)  # get some important stats
```

## Related

[forecasting_metrics](https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9)

[hydroeval](https://github.com/ThibHlln/hydroeval)