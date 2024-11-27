__all__ = ['Metrics']

import json
import numpy as np
from typing import Union, List, Dict

from .utils import features
from .utils import maybe_treat_arrays


# TODO remove repeated calculation of mse, std, mean etc
# TODO make weights, class attribute
# TODO standardized residual sum of squares
# http://documentation.sas.com/?cdcId=fscdc&cdcVersion=15.1&docsetId=fsug&docsetTarget=n1sm8nk3229ttun187529xtkbtpu.htm&locale=en
# https://arxiv.org/ftp/arxiv/papers/1809/1809.03006.pdf
# https://www.researchgate.net/profile/Mark-Tschopp/publication/322147437_Quantifying_Similarity_and_Distance_Measures_for_Vector-Based_Datasets_Histograms_Signals_and_Probability_Distribution_Functions/links/5a48089ca6fdcce1971c8142/Quantifying-Similarity-and-Distance-Measures-for-Vector-Based-Datasets-Histograms-Signals-and-Probability-Distribution-Functions.pdf
# Jeffreys Divergence
# outliear percentage : pysteps
# skill score, mean absolute error skill score, https://doi.org/10.1016/j.ijforecast.2018.11.010
# root mean quartic error, Kolmogorov–Smirnov test integral, OVERPer, Rényi entropy,
# 95th percentile: https://doi.org/10.1016/j.solener.2014.10.016
# Friedman test: https://doi.org/10.1016/j.solener.2014.10.016
# https://arjun-sarkar786.medium.com/implementation-of-all-loss-functions-deep-learning-in-numpy-tensorflow-and-pytorch-e20e72626ebd
# Ritter and Munoz-Carpena (2013) https://doi.org/10.1016/j.jhydrol.2012.12.004

EPS = 1e-10  # epsilon

ERR_STATE = {}


# TODO probability losses
# log normal loss
# skill score

# TODO multi horizon metrics


class Metrics(object):
    """
    This class does some pre-processign and handles metadata regaring true and
    predicted arrays.

    The arguments other than `true` and `predicted` are dynamic i.e. they can be
    changed from outside the class. This means the user can change their value after
    creating the class. This will be useful if we want to calculate an error once by
    ignoring NaN and then by not ignoring the NaNs. However, the user has to run
    the method `treat_arrays` in order to have the changed values impact on true and
    predicted arrays. For ducussion about impact of performance metric see German
    climate computing website_.

    .. _website:
        https://www-miklip.dkrz.de/about/murcss/
    .. _[1]:
        https://doi.org/10.1007/s44150-021-00015-8
    .. _[2]:
        https://doi.org/10.28945/4184

    """

    def __init__(
            self,
            true: Union[np.ndarray, list],
            predicted: Union[np.ndarray, list],
            replace_nan: Union[int, float, None] = None,
            replace_inf: Union[int, float, None] = None,
            remove_zero: bool = False,
            remove_neg: bool = False,
            remove_nan: bool = True,
            remove_inf: bool = True,
            metric_type: str = 'regression',
            np_errstate: dict = None,
    ):

        """
        Parameters
        -----------
            true : array like,
                ture/observed/actual/measured/target values. This can be anything
                which can be converted to numpy array.
            predicted : array like,
                predicted/simulated values. This can be anything
                which can be converted to numpy array.
            replace_nan : default None. if not None, then NaNs in true
                and predicted will be replaced by this value.
            replace_inf : default None, if not None, then inf vlaues in true and
                predicted will be replaced by this value.
            remove_zero : default False, if True, the zero values in true
                or predicted arrays will be removed. If a zero is found in one
                array, the corresponding value in the other array will also be
                removed.
            remove_neg : default False, if True, the negative values in true
                or predicted arrays will be removed.
            remove_inf : bool (default=True)
                whether to remove infitinity (np.inf) values from ``true`` and ``predicted``
                arrays or not.
            remove_nan : bool (default=True)
                whether to remove nan (np.nan) values from ``true`` and ``predicted``
                arrays or not.
            metric_type : type of metric.
            np_errstate : dict
                any keyword options for np.errstate() to calculate np.log1p

        """

        global ERR_STATE

        self.metric_type = metric_type
        self.true, self.predicted = maybe_treat_arrays(
            preprocess=True,
            true=true,
            predicted=predicted,
            metric_type=metric_type,
            remove_nan=remove_nan,
            replace_nan=replace_nan,
            remove_zero=remove_zero,
            remove_neg=remove_neg,
            replace_inf=replace_inf,
            remove_inf=remove_inf,
            )
        self.replace_nan = replace_nan
        self.replace_inf = replace_inf
        self.remove_zero = remove_zero
        self.remove_neg = remove_neg
        self.remove_nan = remove_nan
        if np_errstate is None:
            np_errstate = {}
        self.err_state = np_errstate
        ERR_STATE = np_errstate

    @property
    def log1p_p(self):
        with np.errstate(**self.err_state):
            return np.log1p(self.predicted)

    @property
    def log1p_t(self):
        with np.errstate(**self.err_state):
            return np.log1p(self.true)

    @property
    def log_t(self):
        with np.errstate(**self.err_state):
            return np.log(self.true)

    @property
    def log_p(self):
        with np.errstate(**self.err_state):
            return np.log(self.predicted)

    @staticmethod
    def _minimal() -> list:
        raise NotImplementedError

    @staticmethod
    def _scale_independent_metrics() -> list:
        raise NotImplementedError

    @staticmethod
    def _scale_dependent_metrics() -> list:
        raise NotImplementedError

    @property
    def replace_nan(self):
        return self._replace_nan

    @replace_nan.setter
    def replace_nan(self, x):
        self._replace_nan = x

    @property
    def replace_inf(self):
        return self._replace_inf

    @replace_inf.setter
    def replace_inf(self, x):
        self._replace_inf = x

    @property
    def remove_zero(self):
        return self._remove_zero

    @remove_zero.setter
    def remove_zero(self, x):
        self._remove_zero = x

    @property
    def remove_neg(self):
        return self._remove_neg

    @remove_neg.setter
    def remove_neg(self, x):
        self._remove_neg = x

    def _assert_greater_than_one(self):
        # assert that both true and predicted arrays are greater than one.
        if len(self.true) <= 1 or len(self.predicted) <= 1:
            raise ValueError(f"""
            Expect length of true and predicted arrays to be larger than 1 but 
            they are {len(self.true)} and {len(self.predicted)}""")
        return

    def calculate(
            self,
            metric: Union[str, List[str]],
    )->Dict[str, float]:
        """
        Calculates the error using the given metric.

        Parameters
        ----------
        metric : str or list of str
            name of the metric/metrics to calculate.

        Returns
        -------
        dict
            calculated error.

        Examples
        --------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> true = np.random.random(100)
        >>> predicted = np.random.random(100)
        >>> metrics = RegressionMetrics(true, predicted)
        >>> metrics.calculate('mse')
        >>> metrics.calculate(['mse', 'rmse'])
        """
        if isinstance(metric, str):
            metric = [metric]
        
        assert all([m in self.all_methods + ['mse'] for m in metric]), f"Invalid metric name. Available metrics are {self.all_methods}"

        errors = {}
        for m in metric:
            errors[m] = getattr(self, m)()

        return errors

    def calculate_all(
            self, 
            statistics:bool = False, 
            verbose:bool = False, 
            write:bool = False, 
            name=None) -> dict:
        """ 
        calculates errors using all available methods except brier_score..
        write: bool, if True, will write the calculated errors in file.
        name: str, if not None, then must be path of the file in which to write.

        Parameters
        ----------
        statistics :
        verbose : bool, optional
            if True, will print the calculated errors. The default is False.
        write : bool, optional
            if True, will write the calculated errors in file. The default is False.
        name : str, optional
            if not None, then must be path of the file in which to write. The default is None.

        Returns
        -------
        dict
            dictionary of calculated errors.

        Examples
        --------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> true = np.random.random(100)
        >>> predicted = np.random.random(100)
        >>> metrics = RegressionMetrics(true, predicted)
        >>> metrics.calculate_all()
        """
        errors = {}
        for m in self.all_methods:
            if m not in ["brier_score"]:
                try:
                    error = float(getattr(self, m)())
                # some errors might not have been computed and returned a non float-convertible value e.g. None
                except TypeError:
                    error = getattr(self, m)()
                errors[m] = error
                if verbose:
                    if error is None:
                        print('{0:25} :  {1}'.format(m, error))
                    else:
                        print('{0:25} :  {1:<12.3f}'.format(m, error))

        if statistics:
            errors['stats'] = self.stats(verbose=verbose)

        if write:
            if name is not None:
                assert isinstance(name, str)
                fname = name
            else:
                fname = 'errors'

            with open(fname + ".json", 'w') as fp:
                json.dump(errors, fp, sort_keys=True, indent=4)

        return errors

    def calculate_minimal(self) -> dict:
        """
        Calculates some basic metrics.

        Returns
        -------
        dict
            Dictionary with all metrics

        Examples
        --------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> true = np.random.random(100)
        >>> predicted = np.random.random(100)
        >>> metrics = RegressionMetrics(true, predicted)
        >>> metrics.calculate_minimal()

        """
        metrics = {}

        for metric in self._minimal():
            metrics[metric] = getattr(self, metric)()

        return metrics

    def _error(self, true=None, predicted=None):
        """ simple difference """
        if true is None:
            true = self.true
        if predicted is None:
            predicted = self.predicted
        return true - predicted

    def _percentage_error(self):
        """
        Percentage error
        """
        return self._error() / (self.true + EPS) * 100

    def _naive_prognose(self, seasonality: int = 1):
        """ Naive forecasting method which just repeats previous samples """
        return self.true[:-seasonality]

    def _relative_error(self, benchmark: np.ndarray = None):
        """ Relative Error """
        if benchmark is None or isinstance(benchmark, int):
            # If no benchmark prediction provided - use naive forecasting
            if not isinstance(benchmark, int):
                seasonality = 1
            else:
                seasonality = benchmark
            return self._error(self.true[seasonality:], self.predicted[seasonality:]) / \
                (self._error(self.true[seasonality:], self._naive_prognose(seasonality)) + EPS)

        return self._error() / (self._error(self.true, benchmark) + EPS)

    def _bounded_relative_error(self, benchmark: np.ndarray = None):
        """ Bounded Relative Error """
        if benchmark is None or isinstance(benchmark, int):
            # If no benchmark prediction provided - use naive forecasting
            if not isinstance(benchmark, int):
                seasonality = 1
            else:
                seasonality = benchmark

            abs_err = np.abs(self._error(self.true[seasonality:], self.predicted[seasonality:]))
            abs_err_bench = np.abs(self._error(self.true[seasonality:], self._naive_prognose(seasonality)))
        else:
            abs_err = np.abs(self._error())
            abs_err_bench = np.abs(self._error())

        return abs_err / (abs_err + abs_err_bench + EPS)

    def _ae(self):
        """Absolute error """
        return np.abs(self.true - self.predicted)

    def calculate_scale_independent_metrics(self) -> dict:
        """
        Calculates scale independent metrics

        Returns
        -------
        dict
            Dictionary with all metrics

        Examples
        --------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> true = np.random.random(100)
        >>> predicted = np.random.random(100)
        >>> metrics = RegressionMetrics(true, predicted)
        >>> metrics.calculate_scale_independent_metrics()
        """
        metrics = {}

        for metric in self._scale_independent_metrics():
            metrics[metric] = getattr(self, metric)()

        return metrics

    def calculate_scale_dependent_metrics(self) -> dict:
        """
        Calculates scale dependent metrics

        Returns
        -------
        dict
            Dictionary with all metrics

        Examples
        --------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> true = np.random.random(100)
        >>> predicted = np.random.random(100)
        >>> metrics = RegressionMetrics(true, predicted)
        >>> metrics.calculate_scale_dependent_metrics()

        """
        metrics = {}

        for metric in self._scale_dependent_metrics():
            metrics[metric] = getattr(self, metric)()

        return metrics

    def stats(self, verbose: bool = False) -> dict:
        """ returs some important stats about true and predicted values."""
        _stats = dict()
        _stats['true'] = features(self.true)
        _stats['predicted'] = features(self.predicted)

        if verbose:
            print("\nName            True         Predicted  ")
            print("----------------------------------------")
            for k in _stats['true'].keys():
                print("{:<25},  {:<10},  {:<10}"
                      .format(k, round(_stats['true'][k], 4), round(_stats['predicted'][k])))

        return _stats

    def scale_dependent_metrics(self):
        pass

    def percentage_metrics(self):
        pass

    def relative_metrics(self):
        pass

    def composite_metrics(self):
        pass

    def mse(self, weights=None) -> float:
        """ mean square error """
        return float(np.average((self.true - self.predicted) ** 2, axis=0, weights=weights))
