
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


## How to Install

using github link

	python -m pip install git+https://github.com/AtrCheema/TSErrors.git

using setup file, go to folder where repo is downloaded

    python setup.py install

## Errors

Currently following errors are being calculated.

* Mean Absolute Error
* Mean Square Error
* Root Mean Square Error
* Coefficient of Determination
* RSR
* Nash Sutcliff Efficiency
* Absolute Percent Bias
* Normalized Root Mean Square Error
* Mean Absolute Relative Error
* Bias
* Logrithmic Nash Sutcliff Efficiency
* Logarithmic probability distribution
* Correlation Coefficient
* Relative Root Mean Squared Error
* Agreement Index
* Covariance
* Decomposed Mean Square Error
* Kling-Gupta Efficiency 
* Non parametric Kling-Gupta Efficiency
* Modified Kling-Gupta Efficiency
* Volume Error
* Mean Poisson Deviance
* Mean Gamma Deviance
* Median Absolute Error
* Mean Square Logrithmic Error
* maximum error
* Explained variance score
* Peak flow bias of the flow duration curve
*  Alpha decomposition of the NSE
* Beta decomposition of NSE

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