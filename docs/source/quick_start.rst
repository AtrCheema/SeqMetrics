Quick Start
************

RegressionMetrics
==================

.. code-block:: python

    >>> import numpy as np
    >>> from SeqMetrics import RegressionMetrics

    >>> true = np.random.random((20, 1))
    >>> pred = np.random.random((20, 1))

    >>> er = RegressionMetrics(true, pred)

    >>> for m in er.all_methods: print("{:20}".format(m)) # get names of all availabe methods

    >>> er.nse()   # calculate Nash Sutcliff efficiency

    >>> er.calculate_all(verbose=True)  # or calculate errors using all available methods 


ClassificationMetrics
=====================

.. code-block:: python

    >>> import numpy as np
    >>> from SeqMetrics import ClassificationMetrics

    using boolean array

    >>> t = np.array([True, False, False, False])
    >>> p = np.array([True, True, True, True])
    >>> metrics = ClassificationMetrics(t, p)
    >>> accuracy = metrics.accuracy()

    binary classification with numerical labels

    >>> true = np.array([1, 0, 0, 0])
    >>> pred = np.array([1, 1, 1, 1])
    >>> metrics = ClassificationMetrics(true, pred)
    >>> accuracy = metrics.accuracy()

    multiclass classification with numerical labels

    >>> true = np.random.randint(1, 4, 100)
    >>> pred = np.random.randint(1, 4, 100)
    >>> metrics = ClassificationMetrics(true, pred)
    >>> accuracy = metrics.accuracy()

    You can also provide logits instead of labels.

    >>> predictions = np.array([[0.25, 0.25, 0.25, 0.25],
    >>>                        [0.01, 0.01, 0.01, 0.96]])
    >>> targets = np.array([[0, 0, 0, 1],
    >>>                     [0, 0, 0, 1]])
    >>> metrics = ClassificationMetrics(targets, predictions, multiclass=True)
    >>> metrics.cross_entropy()
    ...  0.71355817782