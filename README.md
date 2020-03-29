
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


## How to Install

using github link

	python -m pip install git+https://github.com/AtrCheema/TSErrors.git

using setup file, go to folder where repo is downloaded

    python setup.py install

## How to Use

```python
from TSErrors import 

true = np.random.random((20, 1))
pred = np.random.random((20, 1))

er = FindErrors(true, pred)

er_methods = [method for method in dir(er) if callable(getattr(er, method)) if
						   not method.startswith('_')]

for m in er_methods:
	print('{0:15} :  {1:<12.3f}'.format(m, float(getattr(er, m)())))
```