
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

The purpose of this repository to collect all performance metrics or errors which can be
calculated for time-series data, at one place. Currently only 1d data is supported.

## How to Install

using github link

	python -m pip install git+https://github.com/AtrCheema/TSErrors.git

using setup file, go to folder where repo is downloaded

    python setup.py install

## Errors

Currently following errors are being calculated.

* Absolute Percent Bias
* Agreement Index
* Alpha decomposition of the NSE
* Bias
* Beta decomposition of NSE
* Bounded NSE
* Bounded KGE
* Correlation Coefficient
* Coefficient of Determination
* Covariance
* Decomposed Mean Square Error
* Explained variance score
* Geometric Mean Absolute Error
* Geometric Mean Relative Absolute Error
* Integral Normalized Root Squared Error
* Kling-Gupta Efficiency
* Logrithmic Nash Sutcliff Efficiency
* Logrithmic probability distribution
* maximum error
* Mean Absolute Error
* Mean Absolute Percentage Error
* Mean Absolute Relative Error
* Mean Absolute Scaled Error
* Mean Arctangle Absolute Percentage Error
* Mean Bias Error
* Mean Bounded relative Absolute Error
* Mean Error
* Mean Gamma Deviance
* Mean Percentage Error
* Mean Poisson Deviance
* Mean Relative Absolute Error
* Mean Square Error
* Mean Square Logrithmic Error
* Median Absolute Error
* Median Absoltue Percentage Error
* Median Dictionary Accuracy
* Median Relative Absolute Error
* Modified Kling-Gupta Efficiency
* Nash Sutcliff Efficiency
* Non parametric Kling-Gupta Efficiency
* Normalized Absolute Error
* Normalized Absolute Percentage Error
* Normalized Root Mean Square Error
* Peak flow bias of the flow duration curve
* Percent Bias
* Relative Absolute Error
* Relative Root Mean Squared Error
* Root Mean Square Error
* Root Mean Square Percentage Error
* Root Mean Squared Scaled Error
* Root Median Squared Scaled Error
* Root Relative Squared Error
* RSR
* Symmetric Mean Absolute Percentage Error
* Symmetric Median Absolute Percentage Error
* Volume Error
* Unscaled Mean Bounded Relative Absolute Error
* Weighted Mean Absolute Percent Error

## How to Use

```python
import numpy as np
from TSErrors import FindErrors

true = np.random.random((20, 1))
pred = np.random.random((20, 1))

er = FindErrors(true, pred)

for m in er.all_methods: print("{:20}".format(m)) # get names of all availabe methods

er.nse()   # calculate Nash Sutcliff efficiency

er.calculate_all()  # or calculate errors using all available methods
```

## Related

[forecasting_metrics](https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9)

[hydroeval](https://github.com/ThibHlln/hydroeval)