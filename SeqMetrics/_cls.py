
import numbers
import warnings
from typing import Union

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

from .utils import list_subclass_methods
from ._main import Metrics

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
        multiclass:bool = False,
        *args,
        **kwargs
    ):

        self.multiclass = multiclass

        super().__init__(true, predicted, metric_type='classification', *args, **kwargs)

        self.is_categorical = False
        if self.true.dtype.kind in ['S', 'U']:
            self.is_categorical = True
            assert self.predicted.dtype.kind in ['S', 'U']

            self.true_cls , self.true_encoded = self._encode(self.true)
            self.pred_cls, self.pred_encoded = self._encode(self.predicted)

        self.true_labels = self._true_labels()
        self.true_logits = self._true_logits()
        self.pred_labels = self._pred_labels()
        self.pred_logits = self._pred_logits()

        self.all_methods = list_subclass_methods(ClassificationMetrics, True)
        
        self.n_samples = len(self.true_labels)
        self.labels = np.unique(np.stack((self.true_labels, self.pred_labels)))
        self.n_labels = self.labels.size

        self.cm = self._confusion_matrix()

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
        array = self.true_labels
        return np.unique(array[~np.isnan(array)])

    def _true_labels(self):
        """retuned array is 1d"""
    
        if self.multiclass:

            if self.true.size == len(self.true):
                return self.true.reshape(-1,1)
            
            # supposing this to be logits
            return np.argmax(self.true, axis=1)

        true = self.true
        # it should be 1 dimensional
        if true.size != len(true):
            true = np.argmax(true, 1)
        return true.reshape(-1,)

    def _true_logits(self):
        """returned array is 2d"""
        if self.multiclass:
            return self.true

        # for binary if the array is 2-d, consider it to be logits
        if len(self.true) == self.true.size:
            return binarize(self.true)

        return self.true

    def _pred_labels(self):
        """returns 1d"""

        if self.multiclass:

            if self.predicted.size == len(self.predicted):
                return self.predicted.reshape(-1,1)
            
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

    def cross_entropy(self, epsilon=1e-12)->float:
        """
        Computes cross entropy between targets (encoded as one-hot vectors)
        and predictions.

        Returns
        -------
        scalar
        
        """
        if self.is_categorical:
            predictions = np.clip(self.pred_encoded, epsilon, 1. - epsilon)
            n = predictions.shape[0]
            ce = -np.sum(self.true_encoded * np.log(predictions + 1e-9)) / n
        else:
            predictions = np.clip(self.predicted, epsilon, 1. - epsilon)
            n = predictions.shape[0]
            ce = -np.sum(self.true * np.log(predictions + 1e-9)) / n
        return ce

    # def hinge_loss(self):
    #     """hinge loss using sklearn"""
    #     if self.pred_logits is not None:
    #         return hinge_loss(self.true_labels, self.pred_logits)
    #     return None

    def accuracy(self, normalize:bool=True)->float:
        """
        calculates accuracy

        Parameters
        ----------
        normalize : bool

        Returns
        -------
        float

        Examples
        --------
        >>> import numpy as np
        >>> from SeqMetrics import ClassificationMetrics
        >>> true = np.array([1, 0, 0, 0])
        >>> pred = np.array([1, 1, 1, 1])
        >>> metrics = ClassificationMetrics(true, pred)
        >>> print(metrics.accuracy())
        """
        if normalize:
            return np.average(self.true_labels==self.pred_labels)
        return (self.true_labels==self.pred_labels).sum()

    def confusion_matrix(self, normalize=False):
        """
        calculates confusion matrix
        
        Parameters
        ----------
        normalize : str, [None, 'true', 'pred', 'all] 
            If None, no normalization is done.

        Returns
        -------
        ndarray 
            confusion matrix
        
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
        return self._confusion_matrix(normalize=normalize)

    def _confusion_matrix(self, normalize=None):

        pred = self.pred_labels.reshape(-1,)
        true = self.true_labels.reshape(-1,)
        # copying method of sklearn
        target_shape = (len(self.labels), len(self.labels))

        
        label_to_ind = {y: x for x, y in enumerate(self.labels)}
        # convert yt, yp into index
        pred = np.array([label_to_ind.get(x, self.n_labels + 1) for x in pred])
        true = np.array([label_to_ind.get(x, self.n_labels + 1) for x in true])

        # intersect y_pred, y_true with labels, eliminate items not in labels
        ind = np.logical_and(pred < self.n_labels, true < self.n_labels)
        y_pred = pred[ind]
        y_true = true[ind]

        cm = coo_matrix(
            (np.ones(self.n_samples, dtype=int), (y_true, y_pred)),
            shape=target_shape,
            dtype=int
        ).toarray()

        if normalize:
            assert normalize in ("true", "pred", "all")

            with np.errstate(all='ignore'):
                if normalize == 'true':
                    cm = cm / cm.sum(axis=1, keepdims=True)
                elif normalize == 'pred':
                    cm = cm / cm.sum(axis=0, keepdims=True)
                elif normalize == 'all':
                    cm = cm / cm.sum()

        return np.nan_to_num(cm)

    def _tp(self):
        return np.diag(self.cm)

    def _fp(self):
        return np.sum(self.cm, axis=0) - self._tp()

    def _fn(self):
        return np.sum(self.cm, axis=1) - self._tp()

    def _tn(self):

        TN = []
        for i in range(self.n_labels):
            temp = np.delete(self.cm, i, 0)    # delete ith row
            temp = np.delete(temp, i, 1)  # delete ith column
            TN.append(sum(sum(temp)))

        return TN

    @staticmethod
    def _is_scalar_nan(x):
        # same as sklearn function
        return bool(isinstance(x, numbers.Real) and np.isnan(x))

    def _encode(self, x:np.ndarray)->tuple:
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
        TP/(TP+FP)

        Parameters
        ----------
        average : string, [None, ``macro``, ``weighted``, ``micro``]

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
        TP = self._tp()
        FP = self._fp()

        if average == "micro":
            return sum(TP) / (sum(TP) + sum(FP))

        _precision =  TP / (TP + FP)
        _precision = np.nan_to_num(_precision)

        if average:
            assert average in ['macro', 'weighted']
            if average == 'macro':
                return np.mean(_precision)
                #return np.nanmean(_precision)
            
            elif average == 'weighted':
                
                return np.average(_precision, weights= TP + self._fn())

        return _precision

    def recall(self, average=None):
        """
        It is also called sensitivity or true positive rate. It is
        number of correct positive predictions divided by the total number of positives
        Formula :
            True Posivitive / True Positive + False Negative

        Parameters
        ----------
            average : str (default=None)
                one of None, ``macro``, ``weighted``, or ``micro``

        """

        TP = self._tp()
        FN = self._fn()

        if average == "micro":
            return sum(TP) / (sum(TP) + sum(FN))

        _recall = TP /( TP+ FN)

        _recall = np.nan_to_num(_recall)

        if average:
            assert average in ['macro', 'weighted']
            if average == 'macro':
                return _recall.mean()
            
            elif average == 'weighted':
                return np.average(_recall, weights= TP + FN)
        
        return _recall

    def specificity(self, average=None):
        """
        It is also called true negative rate or selectivity. It is the probability that
        the predictions are negative when the true labels are also negative.
        It is number of correct negative predictions divided by the total number of negatives.

        It's formula is following
        TN / TN+FP

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
        TN = self._tn()
        FP = self._fp()

        if average == "micro":
            return sum(TN) / (sum(TN) + sum(FP))

        _spcificity =  TN / (TN + FP)

        if average:
            assert average in ['macro', 'weighted']
            if average == 'macro':
                return _spcificity.mean()
            
            else:
                return np.average(_spcificity, weights= TN + FP)
        
        return _spcificity

    def balanced_accuracy(self, average=None)->float:
        """
        balanced accuracy.
        It performs better on imbalanced datasets.
        """
        TP = self._tp()
        score = TP / self.cm.sum(axis=1)
        if np.any(np.isnan(score)):
            warnings.warn('y_pred contains classes not in y_true')
        score = np.nanmean(score).item()

        return score

    def _f_score(self, average=None, beta=1.0):
        """calculates baseic f score"""

        precision = self.precision()
        recall = self.recall()

        if average == "micro":
            return ((1 + beta**2) * (self.precision("micro") * self.recall("micro"))) / (beta**2 * (self.precision("micro") + self.recall("micro")))

        _f_score = ((1 + beta**2) * (precision * recall))  / (beta**2 * (precision + recall))

        _f_score = np.nan_to_num(_f_score)

        if average:
            assert average in ['macro', 'weighted']

            if average == 'macro':
                return _f_score.mean()

            if average == 'weighted':
                return np.average(_f_score, weights = self._tp() + self._fn())

        return _f_score

    def f1_score(self, average=None)->Union[np.ndarray, float]:
        """
        Calculates f1 score according to following formula
        f1_score = 2 * (precision * recall)  / (precision + recall)

        Parameters
        ----------
        average : str, optional
            It can be ``macro`` or ``weighted``. or ``micro``

        Returns
        -------
        array or float

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

        return self._f_score(average, 1.0)

    def f2_score(self, average=None):
        """
        f2 score
        """
        return self._f_score(average, 2.0)

    def false_positive_rate(self):
        """
        False positive rate is the number of incorrect positive predictions divided
        by the total number of negatives. Its best value is 0.0 and worst value is 1.0.
        It is also called probability of false alarm or fall-out.

         TP / (TP + TN)

        """
        TP = self._tp()
        fpr = TP / (TP + self._tn())

        fpr = np.nan_to_num(fpr)

        return fpr

    def false_discovery_rate(self):
        """
        False discovery rate
         FP / (TP + FP)
        """
        FP = self._fp()

        fdr = FP / (self._tp() + FP)

        fdr = np.nan_to_num(fdr)

        return fdr

    def false_negative_rate(self):
        """
        False Negative Rate or miss rate.

        FN / (FN + TP)
        """
        FN = self._fn()
        fnr = FN / (FN + self._tp())

        fnr = np.nan_to_num(fnr)

        return fnr

    def negative_predictive_value(self):
        """
        Negative Predictive Value
        TN/(TN+FN)
        """
        TN = self._tn()
        npv = TN / (TN + self._fn())

        npv = np.nan_to_num(npv)
        return npv

    def error_rate(self):
        """
        Error rate is the number of all incorrect predictions divided by the total
        number of samples in data.
        """

        return (self._fp() + self._fn()) / self.n_samples

    def mathews_corr_coeff(self):
        """
        Methew's correlation coefficient

        """
        TP, TN, FP, FN = self._tp(), self._tn(), self._fp(), self._fn()

        top = TP * TN - FP * FN
        bottom = np.sqrt(((TP + FP) * (FP + FN) * (TN + FP) * (TN + FN)))
        return top/bottom

    def positive_likelihood_ratio(self, average=None):
        """
        Positive likelihood ratio
        sensitivity / 1-specificity

        """
        return self.recall(average=average) / (1 - self.specificity(average=average))

    def negative_likelihood_ratio(self, average=None):
        """
        Negative likelihood ratio

        1 - sensitivity / specificity

        https://en.wikipedia.org/wiki/Likelihood_ratios_in_diagnostic_testing#positive_likelihood_ratio
        """

        return 1 - self.recall(average) / self.specificity(average)

    def youden_index(self, average=None):
        """
        Youden index, also known as informedness

        j = TPR + TNR − 1 =   sensitivity +  specificity - 1

        https://en.wikipedia.org/wiki/Youden%27s_J_statistic
        """
        return  self.recall(average) + self.specificity(average) - 1

    def fowlkes_mallows_index(self, average=None):
        """
        Fowlkes–Mallows index

        sqrt(PPV * TPR)

        PPV is positive predictive value or precision.
        TPR is true positive rate or recall or sensitivity

        https://en.wikipedia.org/wiki/Fowlkes%E2%80%93Mallows_index
        """
        return np.sqrt(self.precision(average) * self.recall(average))

    def prevalence_threshold(self, average=None):
        """
        Prevalence threshold

        sqrt(FPR) / (sqrt(TPR) + sqrt(FPR))

        TPR is true positive rate or recall
        """
        FPR = self.false_positive_rate()

        return np.sqrt(FPR) / (np.sqrt(self.recall(average)) + np.sqrt(FPR))

    def false_omission_rate(self, average=None):
        """
        False omission rate

        FN / (FN + TN)
        """
        FN = self._fn()
        FOR = FN / (FN + self._tn())

        FOR = np.nan_to_num(FOR)

        return FOR


def one_hot_encode(array):
    """one hot encoding of an array like"""
    classes_ = np.unique(array)

    y_in_classes = np.in1d(array, classes_)
    y_seen = array[y_in_classes]
    indices = np.searchsorted(np.sort(classes_), y_seen)
    indptr = np.hstack((0, np.cumsum(y_in_classes)))

    container = np.empty_like(indices)
    container.fill(1)
    Y = csr_matrix((container, indices, indptr),
                      shape=(len(array), len(classes_)))

    Y = Y.toarray()
    Y = Y.astype(int, copy=False)

    if np.any(classes_ != np.sort(classes_)):
        indices = np.searchsorted(np.sort(classes_), classes_)
        Y = Y[:, indices]
    return Y


def binarize(array):
    """must be used only for binary classification"""
    y = one_hot_encode(array)
    return y[:, -1].reshape((-1, 1))

