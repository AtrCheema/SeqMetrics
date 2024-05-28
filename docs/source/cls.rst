Classification Metrics
**********************


Functional API
===============
SeqMetrics also provides a functional API for all the performance
metrics.

.. autofunction:: SeqMetrics.f1_score
.. autofunction:: SeqMetrics.accuracy
.. autofunction:: SeqMetrics.precision
.. autofunction:: SeqMetrics.recall
.. autofunction:: SeqMetrics.balanced_accuracy
.. autofunction:: SeqMetrics.confusion_matrix
.. autofunction:: SeqMetrics.cross_entropy
.. autofunction:: SeqMetrics.specificity
.. autofunction:: SeqMetrics.f2_score
.. autofunction:: SeqMetrics.false_positive_rate
.. autofunction:: SeqMetrics.false_discovery_rate
.. autofunction:: SeqMetrics.negative_predictive_value
.. autofunction:: SeqMetrics.error_rate
.. autofunction:: SeqMetrics.mathews_corr_coeff
.. autofunction:: SeqMetrics.positive_likelihood_ratio
.. autofunction:: SeqMetrics.negative_likelihood_ratio
.. autofunction:: SeqMetrics.youden_index
.. autofunction:: SeqMetrics.fowlkes_mallows_index
.. autofunction:: SeqMetrics.prevalence_threshold
.. autofunction:: SeqMetrics.false_omission_rate

.. autofunction:: SeqMetrics.ClassificationMetrics

Class-Based API
=====================
.. autoclass:: SeqMetrics.ClassificationMetrics
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

