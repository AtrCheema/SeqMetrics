
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


## How to Install

using github link

	python -m pip install git+https://github.com/AtrCheema/TSErrors.git

using setup file, go to folder where repo is downloaded

    python setup.py install

## How to Use

```python
import numpy as np
from TSErrors import FindErrors

true = np.random.random((20, 1))
pred = np.random.random((20, 1))

er = FindErrors(true, pred)

er.all_methods # get names of all availabe methods

er.calculate_all()  # calculate errors using all available methods
```