
import numbers
import warnings
from typing import Union

import numpy as np

from ._main import Metrics
from .utils import one_hot_encode
from .utils import confusion_matrix
from .utils import list_subclass_methods


# confusion index

class ClassificationMetrics(Metrics):
    """Calculates classification metrics.

    Parameters
    ----------
    true : array-like of shape = [n_samples] or [n_samples, n_classes]
        True class labels.
    predicted : array-like of shape = [n_samples] or [n_samples, n_classes]
        Predicted class labels.
    multiclass : boolean, optional
        If true, it is assumed that the true labels are multiclass.
    **kwargs : optional
        Additional arguments to be passed to the :py:class:`Metrics` class.


    Examples
    --------
    >>> import numpy as np
    >>> from SeqMetrics import ClassificationMetrics

    boolean array

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

    Working with categorical values is seamless

    >>> true = np.array(['a', 'b', 'b', 'b'])
    >>> pred = np.array(['a', 'a', 'a', 'a'])
    >>> metrics = ClassificationMetrics(true, pred)
    >>> accuracy = metrics.accuracy()

    same goes for multiclass categorical labels

    >>> t = np.array(['car', 'truck', 'truck', 'car', 'bike', 'truck'])
    >>> p = np.array(['car', 'car',   'bike',  'car', 'bike', 'truck'])
    >>> metrics = ClassificationMetrics(targets, predictions, multiclass=True)
    >>> print(metrics.calculate_all())
    """

    # todo add very major erro and major error

    def __init__(
            self,
            true,
            predicted,
            multiclass: bool = False,
            *args,
            **kwargs
    ):

        self.multiclass = multiclass

        super().__init__(true, predicted, metric_type='classification', *args, **kwargs)

        self.is_categorical = False
        if self.true.dtype.kind in ['S', 'U']:
            self.is_categorical = True
            assert self.predicted.dtype.kind in ['S', 'U']

            self.true_cls, self.true_encoded = self._encode(self.true)
            self.pred_cls, self.pred_encoded = self._encode(self.predicted)

        self.true_labels = self._true_labels()
        self.true_logits = self._true_logits()
        self.pred_labels = self._pred_labels()
        self.pred_logits = self._pred_logits()

        self.all_methods = list_subclass_methods(ClassificationMetrics, True)

        self.n_samples = len(self.true_labels)
        self.labels = np.unique(np.stack((self.true_labels, self.pred_labels)))
        self.n_labels = self.labels.size

        self.cm = confusion_matrix(self.true_labels, self.pred_labels)

    @staticmethod
    def _minimal() -> list:
        """some minimal and basic metrics"""
        return list_subclass_methods(ClassificationMetrics, True)

    @staticmethod
    def _hydro_metrics() -> list:
        """some minimal and basic metrics"""
        return list_subclass_methods(ClassificationMetrics, True)

    def _num_classes(self):
        return len(self._classes())

    def _classes(self):
        if self.is_categorical:
            # can't do np.isnan on categorical
            return np.unique(self.true_labels)

        array = self.true_labels
        return np.unique(array[~np.isnan(array)])

    def _true_labels(self):
        """retuned array is 1d"""

        if self.multiclass:

            if self.true.size == len(self.true):
                return self.true.reshape(-1, 1)

            # supposing this to be logits
            return np.argmax(self.true, axis=1)

        true = self.true
        # it should be 1 dimensional
        if true.size != len(true):
            true = np.argmax(true, 1)
        return true.reshape(-1, )

    def _true_logits(self):
        """returned array is 2d"""
        if self.multiclass:
            return self.true

        # it is a one-D array for binary classification
        if len(self.true) == self.true.size:
            return binarize(self.true)

        # for binary if the array is 2-d, consider it to be logits
        return self.true

    def _pred_labels(self):
        """returns 1d"""

        if self.multiclass:

            if self.predicted.size == len(self.predicted):
                return self.predicted.reshape(-1, 1)

            # supposing this to be logits
            return np.argmax(self.predicted, axis=1)

        # for binary if the array is 2-d, consider it to be logits
        if len(self.predicted) != self.predicted.size:
            return np.argmax(self.predicted, 1)

        if self.is_categorical:
            return np.array(self.predicted)

        return np.array(self.predicted, dtype=int)

    def _pred_logits(self):
        """returned array is 2d"""
        if self.multiclass:
            return self.true
        # we can't do it
        return None

    def cross_entropy(self, epsilon=1e-12) -> float:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from SeqMetrics import ClassificationMetrics
        >>> true = np.array([1, 0, 0, 0])
        >>> pred = np.array([1, 1, 1, 1])
        >>> metrics = ClassificationMetrics(true, pred)
        >>> print(metrics.cross_entropy())
        Computes cross entropy between targets (encoded as one-hot vectors)
        and predictions.
        """
        return cross_entropy(true=self.true, predicted=self.predicted, epsilon=epsilon)

    # def hinge_loss(self):
    #     """hinge loss using sklearn"""
    #     if self.pred_logits is not None:
    #         return hinge_loss(self.true_labels, self.pred_logits)
    #     return None

    def accuracy(self, normalize: bool = True) -> float:
        """
        calculates accuracy

        .. math::
            \\text{Accuracy} = \\frac{\\sum_{i=1}^{N} \\mathbb{1}(true_i = predicted_i)}{N}

        Examples
        --------
        >>> import numpy as np
        >>> from SeqMetrics import ClassificationMetrics
        >>> true = np.array([1, 0, 0, 0])
        >>> pred = np.array([1, 1, 1, 1])
        >>> metrics = ClassificationMetrics(true, pred)
        >>> print(metrics.accuracy())
        """

        return accuracy(true=self.true, predicted=self.predicted, normalize=normalize)

    def confusion_matrix(self, normalize=False):
        """
        calculates confusion matrix

        Examples
        --------
        >>> import numpy as np
        >>> from SeqMetrics import ClassificationMetrics
        >>> true = np.array([1, 0, 0, 0])
        >>> pred = np.array([1, 1, 1, 1])
        >>> metrics = ClassificationMetrics(true, pred)
        >>> metrics.confusion_matrix()

        multiclass classification

        >>> true = np.random.randint(1, 4, 100)
        >>> pred = np.random.randint(1, 4, 100)
        >>> metrics = ClassificationMetrics(true, pred)
        >>> metrics.confusion_matrix()
        """

        if self.multiclass:
            return confusion_matrix(
                self.true_labels,
                self.pred_labels,
                normalize=normalize)

        return confusion_matrix(
            self.true_labels,
            self.pred_labels,
            num_classes=self._num_classes(),
            normalize=normalize)


    def _tp(self):
        return np.diag(self.cm)

    def _fp(self):
        return np.sum(self.cm, axis=0) - self._tp()

    def _fn(self):
        return np.sum(self.cm, axis=1) - self._tp()

    def _tn(self):

        TN = []
        for i in range(self.n_labels):
            temp = np.delete(self.cm, i, 0)  # delete ith row
            temp = np.delete(temp, i, 1)  # delete ith column
            TN.append(sum(sum(temp)))

        return TN

    @staticmethod
    def _is_scalar_nan(x):
        # same as sklearn function
        return bool(isinstance(x, numbers.Real) and np.isnan(x))

    def _encode(self, x: np.ndarray) -> tuple:
        """encodes a categorical array into numerical values"""
        classes, encoded = np.unique(x, return_inverse=True)

        # following lines are taken from sklearn
        # np.unique will have duplicate missing values at the end of `uniques`
        # here we clip the nans and remove it from uniques
        if classes.size and self._is_scalar_nan(classes[-1]):
            nan_idx = np.searchsorted(classes, np.nan)
            classes = classes[:nan_idx + 1]

            encoded[encoded > nan_idx] = nan_idx

        return classes, encoded

    def _decode_true(self):
        raise NotImplementedError

    def _decode_prediction(self):
        raise NotImplementedError

    def precision(self, average=None):
        """
        Returns precision score, also called positive predictive value.
        It is number of correct positive predictions divided by the total
        number of positive predictions.
        .. math::
            \\text{Precision}_{\\text{micro}} = \\frac{\\sum TP}{\\sum (TP + FP)}

        .. math::
            \\text{Precision}_{\\text{macro}} = \\frac{1}{N} \\sum_{i=1}^{N} \\frac{TP_i}{TP_i + FP_i}

        .. math::
            \\text{Precision}_{\\text{weighted}} = \\frac{\\sum_{i=1}^{N} (TP_i + FN_i) \\cdot \\frac{TP_i}{TP_i + FP_i}}{\\sum_{i=1}^{N} (TP_i + FN_i)}

        Examples
        --------
        >>> import numpy as np
        >>> from SeqMetrics import ClassificationMetrics
        >>> true = np.array([1, 0, 0, 0])
        >>> pred = np.array([1, 1, 1, 1])
        >>> metrics = ClassificationMetrics(true, pred)
        >>> print(metrics.precision())
        ...
        >>> print(metrics.precision(average="macro"))
        >>> print(metrics.precision(average="weighted"))
        """
        return precision(true=self.true, predicted=self.predicted, average=average)

    def recall(self, average=None):
        """
        It is also called sensitivity or true positive rate. It is
        number of correct positive predictions divided by the total number of positives

        .. math::
            \\text{Recall} = \\frac{\\text{True Positive}}{\\text{True Positive} + \\text{False Negative}}

        .. math::
            \\text{Recall}_{\\text{micro}} = \\frac{\\sum_{i=1}^{n} \\text{TP}_i}{\\sum_{i=1}^{n} (\\text{TP}_i + \\text{FN}_i)}

        .. math::
            \\text{Recall}_{\\text{macro}} = \\frac{1}{n} \\sum_{i=1}^{n} \\frac{\\text{TP}_i}{\\text{TP}_i + \\text{FN}_i}

        .. math::
            \\text{Recall}_{\\text{weighted}} = \\sum_{i=1}^{n} w_i \\cdot \\frac{\\text{TP}_i}{\\text{TP}_i + \\text{FN}_i}

        Examples
        --------
        >>> import numpy as np
        >>> from SeqMetrics import ClassificationMetrics
        >>> true = np.array([1, 0, 0, 0])
        >>> pred = np.array([1, 1, 1, 1])
        >>> metrics = ClassificationMetrics(true, pred)
        >>> metrics.recall()

        """
        return recall(true=self.true, predicted=self.predicted, average=average)

    def specificity(self, average=None):
        """
        It is also called true negative rate or selectivity. It is the probability that
        the predictions are negative when the true labels are also negative.
        It is number of correct negative predictions divided by the total number of negatives.

        .. math::
            \\text{Specificity} = \\frac{TN}{TN + FP}

        Examples
        --------
        >>> import numpy as np
        >>> from SeqMetrics import ClassificationMetrics
        >>> true = np.array([1, 0, 0, 0])
        >>> pred = np.array([1, 1, 1, 1])
        >>> metrics = ClassificationMetrics(true, pred)
        >>> print(metrics.specificity())
        ...
        >>> print(metrics.specificity(average="macro"))
        >>> print(metrics.specificity(average="weighted"))
        """
        return specificity(true=self.true, predicted=self.predicted, average=average)

    def balanced_accuracy(self, average=None) -> float:
        """
        balanced accuracy.
        It performs better on imbalanced datasets.

        .. math::
            \\text{Balanced Accuracy} = \\frac{1}{C} \\sum_{i=1}^{C} \\frac{TP_i}{TP_i + FN_i}

        Examples
        --------
        >>> import numpy as np
        >>> from SeqMetrics import ClassificationMetrics
        >>> true = np.array([1, 0, 0, 0])
        >>> pred = np.array([1, 1, 1, 1])
        >>> metrics = ClassificationMetrics(true, pred)
        >>> metrics.balanced_accuracy()
        """
        return balanced_accuracy(true=self.true, predicted=self.predicted, average=average)

    def _f_score(self, average=None, beta=1.0):
        """calculates baseic f score"""

        precision = self.precision()
        recall = self.recall()

        if average == "micro":
            return ((1 + beta ** 2) * (self.precision("micro") * self.recall("micro"))) / (
                    beta ** 2 * (self.precision("micro") + self.recall("micro")))

        _f_score = ((1 + beta ** 2) * (precision * recall)) / (beta ** 2 * (precision + recall))

        _f_score = np.nan_to_num(_f_score)

        if average:
            assert average in ['macro', 'weighted']

            if average == 'macro':
                return _f_score.mean()

            if average == 'weighted':
                return np.average(_f_score, weights=self._tp() + self._fn())

        return _f_score

    def f1_score(self, average=None) -> Union[np.ndarray, float]:
        """
           Calculates f1 score according to following formula
           f1_score = 2 * (precision * recall)  / (precision + recall)

            .. math::
                F1 = 2 \\cdot \\frac{\\text{precision} \\cdot \\text{recall}}{\\text{precision} + \\text{recall}}

           Examples
           --------
           >>> import numpy as np
           >>> from SeqMetrics import ClassificationMetrics
           >>> true = np.array([1, 0, 0, 0])
           >>> pred = np.array([1, 1, 1, 1])
           >>> metrics = ClassificationMetrics(true, pred)
           >>> calc_f1_score = metrics.f1_score()
           ...
           >>> print(metrics.f1_score(average="macro"))
           >>> print(metrics.f1_score(average="weighted"))

               """
        return f1_score(true=self.true, predicted=self.predicted, average=average)

    def f2_score(self, average=None):
        """
        f2 score

        .. math::
            F2 = \\left(1 + 2^2\\right) \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{(2^2 \\cdot \\text{Precision}) + \\text{Recall}}

        Examples
       --------
       >>> import numpy as np
       >>> from SeqMetrics import ClassificationMetrics
       >>> true = np.array([1, 0, 0, 0])
       >>> pred = np.array([1, 1, 1, 1])
       >>> metrics = ClassificationMetrics(true, pred)
       >>> calc_f2_score = metrics.f2_score()
       ...
       >>> print(metrics.f2_score(average="macro"))
       >>> print(metrics.f2_score(average="weighted"))
        """
        return f2_score(true=self.true, predicted=self.predicted, average=average)

    def false_positive_rate(self):
        """
        False positive rate is the number of incorrect positive predictions divided
        by the total number of negatives. Its best value is 0.0 and worst value is 1.0.
        It is also called probability of false alarm or fall-out.

        .. math::
            \\text{FPR} = \\frac{\\text{FP}}{\\text{FP} + \\text{TN}}S

        Examples
       --------
       >>> import numpy as np
       >>> from SeqMetrics import ClassificationMetrics
       >>> true = np.array([1, 0, 0, 0])
       >>> pred = np.array([1, 1, 1, 1])
       >>> metrics = ClassificationMetrics(true, pred)
       >>> print(metrics.false_positive_rate())
        """
        return false_positive_rate(true=self.true, predicted=self.predicted)

    def false_discovery_rate(self):
        """
        False discovery rate

        .. math::
            FDR = \\frac{FP}{TP + FP}

        Examples
       --------
       >>> import numpy as np
       >>> from SeqMetrics import ClassificationMetrics
       >>> true = np.array([1, 0, 0, 0])
       >>> pred = np.array([1, 1, 1, 1])
       >>> metrics = ClassificationMetrics(true, pred)
       >>> print(metrics.false_discovery_rate())
        """
        return false_discovery_rate(true=self.true, predicted=self.predicted)

    def false_negative_rate(self):
        """
        False Negative Rate or miss rate.

        .. math::
            \\text{FNR} = \\frac{\\text{FN}}{\\text{FN} + \\text{TP}}

        Examples
       --------
       >>> import numpy as np
       >>> from SeqMetrics import ClassificationMetrics
       >>> true = np.array([1, 0, 0, 0])
       >>> pred = np.array([1, 1, 1, 1])
       >>> metrics = ClassificationMetrics(true, pred)
       >>> print(metrics.false_negative_rate())
        """
        return false_negative_rate(true=self.true, predicted=self.predicted)

    def negative_predictive_value(self):
        """
        Negative Predictive Value

        .. math::
            \\text{NPV} = \\frac{TN}{TN + FN}

        Examples
       --------
       >>> import numpy as np
       >>> from SeqMetrics import ClassificationMetrics
       >>> true = np.array([1, 0, 0, 0])
       >>> pred = np.array([1, 1, 1, 1])
       >>> metrics = ClassificationMetrics(true, pred)
       >>> print(metrics.negative_predictive_value())
        """
        return negative_predictive_value(true=self.true, predicted=self.predicted)

    def error_rate(self):
        """
        Error rate is the number of all incorrect predictions divided by the total
        number of samples in data.

        .. math::
            \\text{Error Rate} = \\frac{\\text{FP} + \\text{FN}}{n}

        Examples
       --------
       >>> import numpy as np
       >>> from SeqMetrics import ClassificationMetrics
       >>> true = np.array([1, 0, 0, 0])
       >>> pred = np.array([1, 1, 1, 1])
       >>> metrics = ClassificationMetrics(true, pred)
       >>> print(metrics.error_rate())
        """
        return error_rate(true=self.true, predicted=self.predicted)

    def mathews_corr_coeff(self):
        """
        Methew's correlation coefficient

        .. math::
            \\text{MCC} = \\frac{TP \\cdot TN - FP \\cdot FN}{\\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}

        Examples
       --------
       >>> import numpy as np
       >>> from SeqMetrics import ClassificationMetrics
       >>> true = np.array([1, 0, 0, 0])
       >>> pred = np.array([1, 1, 1, 1])
       >>> metrics = ClassificationMetrics(true, pred)
       >>> print(metrics.mathews_corr_coeff())
        """
        return mathews_corr_coeff(true=self.true, predicted=self.predicted)

    def positive_likelihood_ratio(self, average=None):
        """
        Positive likelihood ratio

        .. math::
            LR+ = \\frac{\\text{Sensitivity}}{1 - \\text{Specificity}}

        Examples
       --------
       >>> import numpy as np
       >>> from SeqMetrics import ClassificationMetrics
       >>> true = np.array([1, 0, 0, 0])
       >>> pred = np.array([1, 1, 1, 1])
       >>> metrics = ClassificationMetrics(true, pred)
       >>> print(metrics.positive_likelihood_ratio(average="macro"))
       >>> print(metrics.positive_likelihood_ratio(average="weighted"))
        """
        return positive_likelihood_ratio(true=self.true, predicted=self.predicted, average=average)

    def negative_likelihood_ratio(self, average=None):
        """
        Negative likelihood ratio

        .. math::
            \\text{NLR} = 1 - \\frac{\\text{Sensitivity}}{\\text{Specificity}}

        https://en.wikipedia.org/wiki/Likelihood_ratios_in_diagnostic_testing#positive_likelihood_ratio

        Examples
       --------
       >>> import numpy as np
       >>> from SeqMetrics import ClassificationMetrics
       >>> true = np.array([1, 0, 0, 0])
       >>> pred = np.array([1, 1, 1, 1])
       >>> metrics = ClassificationMetrics(true, pred)
       >>> print(metrics.negative_likelihood_ratio(average="macro"))
       >>> print(metrics.negative_likelihood_ratio(average="weighted"))

        """
        return negative_likelihood_ratio(true=self.true, predicted=self.predicted, average=average)

    def youden_index(self, average=None):
        """
        Youden index, also known as informedness

        .. math::
            J = \\text{TPR} + \\text{TNR} - 1 = \\text{sensitivity} + \\text{specificity} - 1

        https://en.wikipedia.org/wiki/Youden%27s_J_statistic

        Examples
       --------
       >>> import numpy as np
       >>> from SeqMetrics import ClassificationMetrics
       >>> true = np.array([1, 0, 0, 0])
       >>> pred = np.array([1, 1, 1, 1])
       >>> metrics = ClassificationMetrics(true, pred)
       >>> print(metrics.youden_index(average="macro"))
       >>> print(metrics.youden_index(average="weighted"))

        """
        return youden_index(true=self.true, predicted=self.predicted, average=average)

    def fowlkes_mallows_index(self, average=None):
        """
        Fowlkes–Mallows index

        .. math::
            \\text{FMI} = \\sqrt{\\text{PPV} \\times \\text{TPR}}


        PPV is positive predictive value or precision.
        TPR is true positive rate or recall or sensitivity

        https://en.wikipedia.org/wiki/Fowlkes%E2%80%93Mallows_index

        Examples
       --------
       >>> import numpy as np
       >>> from SeqMetrics import ClassificationMetrics
       >>> true = np.array([1, 0, 0, 0])
       >>> pred = np.array([1, 1, 1, 1])
       >>> metrics = ClassificationMetrics(true, pred)
       >>> print(metrics.fowlkes_mallows_index(average="macro"))
       >>> print(metrics.fowlkes_mallows_index(average="weighted"))

        """
        return fowlkes_mallows_index(true=self.true, predicted=self.predicted, average=average)

    def prevalence_threshold(self, average=None):
        """
        Prevalence threshold

        .. math::
            PT = \\frac{\\sqrt{FPR}}{\\sqrt{TPR} + \\sqrt{FPR}}

        TPR is true positive rate or recall

        Examples
       --------
       >>> import numpy as np
       >>> from SeqMetrics import ClassificationMetrics
       >>> true = np.array([1, 0, 0, 0])
       >>> pred = np.array([1, 1, 1, 1])
       >>> metrics = ClassificationMetrics(true, pred)
       >>> print(metrics.prevalence_threshold(average="macro"))
       >>> print(metrics.prevalence_threshold(average="weighted"))

        """
        return prevalence_threshold(true=self.true, predicted=self.predicted, average=average)

    def false_omission_rate(self):
        """
        False omission rate

        .. math::
            \\text{FOR} = \\frac{\\text{FN}}{\\text{FN} + \\text{TN}}

        Examples
       --------
       >>> import numpy as np
       >>> from SeqMetrics import ClassificationMetrics
       >>> true = np.array([1, 0, 0, 0])
       >>> pred = np.array([1, 1, 1, 1])
       >>> metrics = ClassificationMetrics(true, pred)
       >>> print(metrics.false_omission_rate())

        """
        return false_omission_rate(true=self.true, predicted=self.predicted)


def cross_entropy(true, predicted, epsilon=1e-12) -> float:
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.

    .. math::
        CE = - \\frac{1}{N} \\sum_{i=1}^{N} \\left[ y_i \\log(\\hat{y}_i + \\epsilon) \\right]
    Returns
    -------
    scalar

    Parameters
    ----------
    true :
         ture/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values

    Examples
    --------
    >>> import numpy as np
    >>> from SeqMetrics import cross_entropy
    >>> true = np.array([1, 0, 0, 0])
    >>> pred = np.array([1, 1, 1, 1])
    >>> metrics = cross_entropy(true, pred)

    """
    cls = ClassificationMetrics(true, predicted)
    if cls.is_categorical:
        predictions = np.clip(cls.pred_encoded, epsilon, 1. - epsilon)
        n = predictions.shape[0]
        ce = -np.sum(cls.true_encoded * np.log(predictions + 1e-9)) / n
    else:
        predictions = np.clip(predicted, epsilon, 1. - epsilon)
        n = predictions.shape[0]
        ce = -np.sum(true * np.log(predictions + 1e-9)) / n
    return ce


def binarize(array):
    """must be used only for binary classification"""
    y = one_hot_encode(array)
    return y[:, -1].reshape((-1, 1))


def accuracy(true, predicted, normalize: bool = True) -> float:
    """
    calculates accuracy

    .. math::
        \\text{Accuracy} = \\frac{\\sum_{i=1}^{N} \\mathbb{1}(true_i = predicted_i)}{N}

    Parameters
    ----------
    normalize : bool
    true:  ture/observed/actual/target values. It must be a numpy array, or pandas series/DataFrame or a list.
    predicted: simulated values

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> from SeqMetrics import accuracy
    >>> true = np.array([1, 0, 0, 0])
    >>> pred = np.array([1, 1, 1, 1])
    >>> accuracy(true, pred)

    """

    cls = ClassificationMetrics(true, predicted)

    if normalize:
        return np.average(cls.true_labels == cls.pred_labels)
    return (cls.true_labels == cls.pred_labels).sum()


def precision(true, predicted, average=None):
    """
    Returns precision score, also called positive predictive value.
    It is number of correct positive predictions divided by the total
    number of positive predictions.
    TP/(TP+FP)

    .. math::
        \\text{Precision}_{\\text{micro}} = \\frac{\\sum TP}{\\sum (TP + FP)}

    .. math::
        \\text{Precision}_{\\text{macro}} = \\frac{1}{N} \\sum_{i=1}^{N} \\frac{TP_i}{TP_i + FP_i}

    .. math::
        \\text{Precision}_{\\text{weighted}} = \\frac{\\sum_{i=1}^{N} (TP_i + FN_i) \\cdot \\frac{TP_i}{TP_i + FP_i}}{\\sum_{i=1}^{N} (TP_i + FN_i)}

    Parameters
    ----------
    average : string, [None, ``macro``, ``weighted``, ``micro``]
    true : ture/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted : simulated values

    Examples
    --------
    >>> import numpy as np
    >>> from SeqMetrics import precision
    >>> true = np.array([1, 0, 0, 0])
    >>> pred = np.array([1, 1, 1, 1])
    >>> metrics = precision(true, pred, average="macro")
    >>> metrics = precision(true, pred, average="weighted")

    """

    cls = ClassificationMetrics(true, predicted)
    TP = cls._tp()
    FP = cls._fp()

    if average == "micro":
        return sum(TP) / (sum(TP) + sum(FP))

    _precision = TP / (TP + FP)
    _precision = np.nan_to_num(_precision)

    if average:
        assert average in ['macro', 'weighted']
        if average == 'macro':
            return np.mean(_precision)
            # return np.nanmean(_precision)

        elif average == 'weighted':

            return np.average(_precision, weights=TP + cls._fn())

    return _precision


def recall(true, predicted, average=None):
    """
    It is also called sensitivity or true positive rate. It is
    number of correct positive predictions divided by the total number of positives

    .. math::
        \\text{Recall} = \\frac{\\text{True Positive}}{\\text{True Positive} + \\text{False Negative}}

    .. math::
        \\text{Recall}_{\\text{micro}} = \\frac{\\sum_{i=1}^{n} \\text{TP}_i}{\\sum_{i=1}^{n} (\\text{TP}_i + \\text{FN}_i)}

    .. math::
        \\text{Recall}_{\\text{macro}} = \\frac{1}{n} \\sum_{i=1}^{n} \\frac{\\text{TP}_i}{\\text{TP}_i + \\text{FN}_i}

    .. math::
        \\text{Recall}_{\\text{weighted}} = \\sum_{i=1}^{n} w_i \\cdot \\frac{\\text{TP}_i}{\\text{TP}_i + \\text{FN}_i}

    Parameters
    ----------
        average : str (default=None)
            one of None, ``macro``, ``weighted``, or ``micro``
        true : ture/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
        predicted : simulated values

    Examples
    --------
    >>> import numpy as np
    >>> from SeqMetrics import recall
    >>> true = np.array([1, 0, 0, 0])
    >>> pred = np.array([1, 1, 1, 1])
    >>> metrics = recall(true, pred, average="macro")
    >>> metrics = recall(true, pred, average="weighted")

    """

    cls = ClassificationMetrics(true, predicted)
    TP = cls._tp()
    FN = cls._fn()

    if average == "micro":
        return sum(TP) / (sum(TP) + sum(FN))

    _recall = TP / (TP + FN)

    _recall = np.nan_to_num(_recall)

    if average:
        assert average in ['macro', 'weighted']
        if average == 'macro':
            return _recall.mean()

        elif average == 'weighted':
            return np.average(_recall, weights=TP + FN)

    return _recall


def specificity(true, predicted, average=None):
    """
    It is also called true negative rate or selectivity. It is the probability that
    the predictions are negative when the true labels are also negative.
    It is number of correct negative predictions divided by the total number of negatives.

    .. math::
        \\text{Specificity} = \\frac{TN}{TN + FP}

    Parameters
    ----------
    true : ture/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted : simulated values

    Examples
    --------
    >>> import numpy as np
    >>> from SeqMetrics import specificity
    >>> true = np.array([1, 0, 0, 0])
    >>> pred = np.array([1, 1, 1, 1])
    >>> print(metrics = specificity(true, pred, average="macro"))
    >>> print(metrics = specificity(true, pred, average="weighted"))

    """
    cls = ClassificationMetrics(true, predicted)
    TN = cls._tn()
    FP = cls._fp()

    if average == "micro":
        return sum(TN) / (sum(TN) + sum(FP))

    _spcificity = np.array(TN) / (TN + FP)

    if average:
        assert average in ['macro', 'weighted']
        if average == 'macro':
            return _spcificity.mean()

        else:
            return np.average(_spcificity, weights=TN + FP)

    return _spcificity


def balanced_accuracy(true, predicted, average=None) -> float:
    """
    balanced accuracy.
    It performs better on imbalanced datasets.

    .. math::
        \\text{Balanced Accuracy} = \\frac{1}{C} \\sum_{i=1}^{C} \\frac{TP_i}{TP_i + FN_i}

    Parameters
    ----------
    true : ture/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted : simulated values

    Examples
    --------
    >>> import numpy as np
    >>> from SeqMetrics import balanced_accuracy
    >>> true = np.array([1, 0, 0, 0])
    >>> pred = np.array([1, 1, 1, 1])
    >>> metrics = balanced_accuracy(true, pred)
    """
    cls = ClassificationMetrics(true, predicted)
    TP = cls._tp()
    score = TP / cls.cm.sum(axis=1)
    if np.any(np.isnan(score)):
        warnings.warn('y_pred contains classes not in y_true')
    score = np.nanmean(score).item()

    return score


def f1_score(true, predicted, average=None) -> Union[np.ndarray, float]:
    """
       Calculates f1 score according to following formula

        .. math::
            F1 = 2 \\cdot \\frac{\\text{precision} \\cdot \\text{recall}}{\\text{precision} + \\text{recall}}

       Parameters
       ----------
       average : str, optional
           It can be ``macro`` or ``weighted``. or ``micro``
       true : ture/observed/actual/target values. It must be a numpy array, or pandas series/DataFrame or a list.
       predicted : simulated values

       Returns
       -------
       array or float

       Examples
       --------
       >>> import numpy as np
       >>> from SeqMetrics import f1_score
       >>> true = np.array([1, 0, 0, 0])
       >>> pred = np.array([1, 1, 1, 1])
       >>> print(metrics = f1_score(true, pred, average="macro"))
       >>> print(metrics = f1_score(true, pred, average="weighted"))

           """
    cls = ClassificationMetrics(true, predicted)

    return cls._f_score(average, 1.0)


def f2_score(true, predicted, average=None):
    """
    f2 score

    .. math::
        F2 = \\left(1 + 2^2\\right) \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{(2^2 \\cdot \\text{Precision}) + \\text{Recall}}

    Parameters
    ----------
    average : str (default=None)
            one of None, ``macro``, ``weighted``, or ``micro``
    true : ture/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted : simulated values

    Examples
   --------
    >>> import numpy as np
    >>> from SeqMetrics import f2_score
    >>> true = np.array([1, 0, 0, 0])
    >>> pred = np.array([1, 1, 1, 1])
    >>> print(f2_score(true, pred, average="macro"))
    >>> print(f2_score(true, pred, average="weighted"))
    """
    cls = ClassificationMetrics(true, predicted)
    return cls._f_score(average, 2.0)


def false_positive_rate(true, predicted):
    """
    False positive rate is the number of incorrect positive predictions divided
    by the total number of negatives. Its best value is 0.0 and worst value is 1.0.
    It is also called probability of false alarm or fall-out.

    .. math::
        \\text{FPR} = \\frac{\\text{FP}}{\\text{FP} + \\text{TN}}S

    Parameters
    ----------
    true : ture/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted : simulated values

    Examples
   --------
    >>> import numpy as np
    >>> from SeqMetrics import false_positive_rate
    >>> true = np.array([1, 0, 0, 0])
    >>> pred = np.array([1, 1, 1, 1])
    >>> print(false_positive_rate(true, pred))
    """
    cls = ClassificationMetrics(true, predicted)
    FP = cls._fp()
    fpr = FP / (FP + cls._tn())

    fpr = np.nan_to_num(fpr)

    return fpr


def false_discovery_rate(true, predicted):
    """
    False discovery rate

    .. math::
        FDR = \\frac{FP}{TP + FP}

    Parameters
    ----------
    true : ture/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted : simulated values

    Examples
   --------
    >>> import numpy as np
    >>> from SeqMetrics import false_discovery_rate
    >>> true = np.array([1, 0, 0, 0])
    >>> pred = np.array([1, 1, 1, 1])
    >>> print(false_discovery_rate(true, pred))
    """
    cls = ClassificationMetrics(true, predicted)
    FP = cls._fp()

    fdr = FP / (cls._tp() + FP)

    fdr = np.nan_to_num(fdr)

    return fdr


def false_negative_rate(true, predicted):
    """
    False Negative Rate or miss rate.

    .. math::
        \\text{FNR} = \\frac{\\text{FN}}{\\text{FN} + \\text{TP}}

    Parameters
    ----------
    true : ture/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted : simulated values

    Examples
   --------
    >>> import numpy as np
    >>> from SeqMetrics import false_negative_rate
    >>> true = np.array([1, 0, 0, 0])
    >>> pred = np.array([1, 1, 1, 1])
    >>> print(false_negative_rate(true, pred))
    """
    cls = ClassificationMetrics(true, predicted)
    FN = cls._fn()
    fnr = FN / (FN + cls._tp())

    fnr = np.nan_to_num(fnr)

    return fnr


def negative_predictive_value(true, predicted):
    """
    Negative Predictive Value

    .. math::
        \\text{NPV} = \\frac{TN}{TN + FN}

    Parameters
    ----------
    true : ture/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted : simulated values

    Examples
   --------
    >>> import numpy as np
    >>> from SeqMetrics import negative_predictive_value
    >>> true = np.array([1, 0, 0, 0])
    >>> pred = np.array([1, 1, 1, 1])
    >>> print(negative_predictive_value(true, pred))
    """
    cls = ClassificationMetrics(true, predicted)
    TN = cls._tn()
    npv = TN / (TN + cls._fn())

    npv = np.nan_to_num(npv)
    return npv


def error_rate(true, predicted):
    """
    Error rate is the number of all incorrect predictions divided by the total
    number of samples in data.

    .. math::
        \\text{Error Rate} = \\frac{\\text{FP} + \\text{FN}}{n}

    Parameters
    ----------
    true : ture/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted : simulated values

    Examples
   --------
    >>> import numpy as np
    >>> from SeqMetrics import error_rate
    >>> true = np.array([1, 0, 0, 0])
    >>> pred = np.array([1, 1, 1, 1])
    >>> print(error_rate(true, pred))
    """
    cls = ClassificationMetrics(true, predicted)

    return (cls._fp() + cls._fn()) / cls.n_samples


def mathews_corr_coeff(true, predicted):
    """
    Methew's correlation coefficient

    .. math::
        \\text{MCC} = \\frac{TP \\cdot TN - FP \\cdot FN}{\\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}

    Parameters
    ----------
    true : ture/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted : simulated values

    Examples
   --------
    >>> import numpy as np
    >>> from SeqMetrics import mathews_corr_coeff
    >>> true = np.array([1, 0, 0, 0])
    >>> pred = np.array([1, 1, 1, 1])
    >>> print(mathews_corr_coeff(true, pred))

    """
    cls = ClassificationMetrics(true, predicted)
    TP, TN, FP, FN = cls._tp(), cls._tn(), cls._fp(), cls._fn()

    top = TP * TN - FP * FN
    bottom = np.sqrt(((TP + FP) * (FP + FN) * (TN + FP) * (TN + FN)))
    return top / bottom


def positive_likelihood_ratio(true, predicted, average=None):
    """
    Positive likelihood ratio

    .. math::
        LR+ = \\frac{\\text{Sensitivity}}{1 - \\text{Specificity}}

    Parameters
    ----------
    average : str (default=None)
            one of None, ``macro``, ``weighted``, or ``micro``
    true : ture/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted : simulated values

    Examples
   --------
    >>> import numpy as np
    >>> from SeqMetrics import positive_likelihood_ratio
    >>> true = np.array([1, 0, 0, 0])
    >>> pred = np.array([1, 1, 1, 1])
    >>> print(positive_likelihood_ratio(true, pred, average="macro"))
    >>> print(positive_likelihood_ratio(true, pred, average="weighted"))
    """
    cls = ClassificationMetrics(true, predicted)
    return cls.recall(average=average) / (1 - cls.specificity(average=average))


def negative_likelihood_ratio(true, predicted, average=None):
    """
    Negative likelihood ratio

    .. math::
        \\text{NLR} = 1 - \\frac{\\text{Sensitivity}}{\\text{Specificity}}

    https://en.wikipedia.org/wiki/Likelihood_ratios_in_diagnostic_testing#positive_likelihood_ratio

    Parameters
    ----------
    average : str (default=None)
            one of None, ``macro``, ``weighted``, or ``micro``
    true : ture/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted : simulated values

    Examples
   --------
    >>> import numpy as np
    >>> from SeqMetrics import negative_likelihood_ratio
    >>> true = np.array([1, 0, 0, 0])
    >>> pred = np.array([1, 1, 1, 1])
    >>> print(negative_likelihood_ratio(true, pred, average="macro"))
    >>> print(negative_likelihood_ratio(true, pred, average="weighted"))
    """
    cls = ClassificationMetrics(true, predicted)

    return 1 - cls.recall(average) / cls.specificity(average)


def youden_index(true, predicted, average=None):
    """
    Youden index, also known as informedness

    .. math::
        J = \\text{TPR} + \\text{TNR} - 1 = \\text{sensitivity} + \\text{specificity} - 1

    https://en.wikipedia.org/wiki/Youden%27s_J_statistic

    Parameters
    ----------
    average : str (default=None)
            one of None, ``macro``, ``weighted``, or ``micro``
    true : ture/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted : simulated values

    Examples
   --------
    >>> import numpy as np
    >>> from SeqMetrics import youden_index
    >>> true = np.array([1, 0, 0, 0])
    >>> pred = np.array([1, 1, 1, 1])
    >>> print(youden_index(true, pred, average="macro"))
    >>> print(youden_index(true, pred, average="weighted"))
    """
    cls = ClassificationMetrics(true, predicted)
    return cls.recall(average) + cls.specificity(average) - 1


def fowlkes_mallows_index(true, predicted, average=None):
    """
    Fowlkes–Mallows index

    .. math::
        \\text{FMI} = \\sqrt{\\text{PPV} \\times \\text{TPR}}

    PPV is positive predictive value or precision.
    TPR is true positive rate or recall or sensitivity

    https://en.wikipedia.org/wiki/Fowlkes%E2%80%93Mallows_index

    Parameters
    ----------
    average : str (default=None)
            one of None, ``macro``, ``weighted``, or ``micro``
    true : ture/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted : simulated values

    Examples
   --------
    >>> import numpy as np
    >>> from SeqMetrics import fowlkes_mallows_index
    >>> true = np.array([1, 0, 0, 0])
    >>> pred = np.array([1, 1, 1, 1])
    >>> print(fowlkes_mallows_index(true, pred, average="macro"))
    >>> print(fowlkes_mallows_index(true, pred, average="weighted"))
    """
    cls = ClassificationMetrics(true, predicted)
    return np.sqrt(cls.precision(average) * cls.recall(average))


def prevalence_threshold(true, predicted, average=None):
    """
    Prevalence threshold

    .. math::
        PT = \\frac{\\sqrt{FPR}}{\\sqrt{TPR} + \\sqrt{FPR}}

    TPR is true positive rate or recall

    Parameters
    ----------
    average : str (default=None)
            one of None, ``macro``, ``weighted``, or ``micro``
    true : ture/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted : simulated values

    Examples
   --------
    >>> import numpy as np
    >>> from SeqMetrics import prevalence_threshold
    >>> true = np.array([1, 0, 0, 0])
    >>> pred = np.array([1, 1, 1, 1])
    >>> print(prevalence_threshold(true, pred, average="macro"))
    >>> print(prevalence_threshold(true, pred, average="weighted"))
    """
    cls = ClassificationMetrics(true, predicted)
    FPR = cls.false_positive_rate()

    return np.sqrt(FPR) / (np.sqrt(cls.recall(average)) + np.sqrt(FPR))


def false_omission_rate(true, predicted):
    """
    False omission rate

    .. math::
        \\text{FOR} = \\frac{\\text{FN}}{\\text{FN} + \\text{TN}}

    Parameters
    ----------
    true : ture/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted : simulated values

    Examples
   --------
    >>> import numpy as np
    >>> from SeqMetrics import false_omission_rate
    >>> true = np.array([1, 0, 0, 0])
    >>> pred = np.array([1, 1, 1, 1])
    >>> print(false_omission_rate(true, pred))
    """
    cls = ClassificationMetrics(true, predicted)
    FN = cls._fn()
    FOR = FN / (FN + cls._tn())

    FOR = np.nan_to_num(FOR)

    return FOR



