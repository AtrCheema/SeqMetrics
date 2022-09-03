
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

    """
    # todo add very major erro and major error

    def __init__(
        self,
        true,
        predicted,
        multiclass=False, 
        *args,
        **kwargs):

        self.multiclass = multiclass

        super().__init__(true, predicted, metric_type='classification', *args, **kwargs)

        self.is_categorical = False
        if self.true.dtype.kind in ['S', 'U']:
            self.is_categorical = True
            assert self.predicted.dtype.kind in ['S', 'U']

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

        Parameters
        ----------
        normalize

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

    def precision(self, average=None):
        """
        Returns precision score, also called positive predictive value.
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
        compute recall.

        Parameters
        ----------
        average : string, [None, ``macro``, ``weighted``, ``micro``]

        """

        TP = self._tp()
        FP = self._fn()

        if average == "micro":
            return sum(TP) / (sum(TP) + sum(FP))

        _recall = TP /( TP+ FP)

        _recall = np.nan_to_num(_recall)

        if average:
            assert average in ['macro', 'weighted']
            if average == 'macro':
                return _recall.mean()
            
            elif average == 'weighted':
                return np.average(_recall, weights= self._tp() + self._fn())
        
        return _recall

    def specificity(self, average=None):
        """
        It is also called true negative rate. It is the probability that
        the predictions are negative when the true labels are also negative.

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
                return np.average(_spcificity, weights= self._tn() + self._fp())
        
        return _spcificity

    def sensitivity(self, average=None):
        """
        It is also called true positive rate.
         True Posivitive / True Positive + False Negative
        """
        TP = self._tp()
        FN = self._fn()

        if average == "micro":
            return sum(TP) / (sum(TP) + sum(FN))

        _sensitivity = TP / (TP + FN)

        if average:
            assert average in ['macro', 'weighted']
            if average == 'macro':
                return _sensitivity.mean()

            else:
                return np.average(_sensitivity, weights=TP + FN)

        return _sensitivity

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

    def f1_score(self, average=None)->Union[np.ndarray, float]:
        """calculates f1 score

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
        precision = self.precision()
        recall = self.recall()

        if average == "micro":
            return  2 * (self.precision("micro") * self.recall("micro"))  / (self.precision("micro") + self.recall("micro"))

        _f1_score = 2 * (precision * recall)  / (precision + recall)

        _f1_score = np.nan_to_num(_f1_score)

        if average:
            assert average in ['macro', 'weighted']

            if average == 'macro':
                return _f1_score.mean()

            if average == 'weighted':
                return np.average(_f1_score, weights = self._tp() + self._fn())
        
        return _f1_score

    def false_positive_rate(self):
        """
        False Positive Rate
         TP / (TP + TN)

        """
        TP = self._tp()
        fpr = TP / (TP + self._tn())

        return fpr

    def false_discovery_rate(self):
        """
        False discovery rate
         FP / (TP + FP)
        """
        FP = self._fp()

        fdr = FP / (self._tp() + FP)
        return fdr

    def false_negative_rate(self):
        """
        False Negative Rate
        FN / (FN + TP)
        """
        FN = self._fn()
        fnr = FN / (FN + self._tp())
        return fnr

    def negative_predictive_value(self):
        """
        Negative Predictive Value
        TN/(TN+FN)
        """
        TN = self._tn()
        npv = TN / (TN + self._fn())
        return npv




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

