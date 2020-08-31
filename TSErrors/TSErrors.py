from math import sqrt
from scipy.special import xlogy
import numpy as np
import warnings

# TODO remove repeated calculation of mse, std, mean etc
# TODO make weights, class attribute
# TODO write tests

EPS = 1e-10  # epsilon


class FindErrors(object):
    """
     actual: ture/observed values, 1D array or list
     predicted: simulated values, 1D array or list
    """

    def __init__(self, actual, predicted):

        self.true, self.predicted = self._pre_process(actual, predicted)
        self.all_methods = [method for method in dir(self) if callable(getattr(self, method)) if
                            not method.startswith('_') if method != 'calculate_all']

        # if arrays contain negative values, following three errors can not be computed
        for array in [self.true, self.predicted]:

            assert len(array) > 0, "Input arrays should not be empty"

            if len(array[array < 0.0]) > 0:
                self.all_methods = [m for m in self.all_methods if m not in ('mean_gamma_deviance',
                                                                             'mean_poisson_deviance',
                                                                             'mean_square_log_error')]
            if (array <= 0).any():  # mean tweedie error is not computable
                self.all_methods = [m for m in self.all_methods if m not in ('mean_gamma_deviance',
                                                                             'mean_poisson_deviance')]
    
    def _pre_process(self, true, predicted):

        predicted = self._assert_array(predicted)
        true = self._assert_array(true)
        assert len(predicted) == len(true), "lengths of provided arrays mismatch, predicted array: {}, true array: {}"\
            .format(len(predicted), len(true))

        return true, predicted

    def _assert_array(self, array_like) -> np.ndarray:

        if not isinstance(array_like, np.ndarray):
            if not isinstance(array_like, list):
                # it can be pandas series or datafrmae
                if array_like.__class__.__name__ in ['Series', 'DataFrame']:
                    if len(array_like.shape) > 1:  # 1d series has shape (x,) while 1d dataframe has shape (x,1)
                        if array_like.shape[1] > 1:  # it is a 2d datafrmae
                            raise TypeError("only 1d pandas Series or dataframe are allowed")
                    np_array = np.array(array_like)
                else:
                    raise TypeError(" all inputs must be numpy array or list")
            else:
                np_array = np.array(array_like)
        else:
            # maybe the dimension is >1 so make sure it is more
            np_array = array_like.reshape(-1, )

        return np_array

    def calculate_all(self, verbose=False):
        """ calculates errors using all available methods"""
        errors = {}
        for m in self.all_methods:
            error = float(getattr(self, m)())
            errors[m] = error
            if verbose:
                print('{0:25} :  {1:<12.3f}'.format(m, error))
        return errors

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
        """ Absolute error """
        return np.abs(self.true - self.predicted)

    def me(self):
        """ mean error """
        return np.mean(self._error())

    def mase(self, seasonality: int = 1):
        """
        Mean Absolute Scaled Error
        Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
        modified after https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
        """
        return self.mae() / self.mae(self.true[seasonality:], self._naive_prognose(seasonality))

    def rmsse(self, seasonality: int = 1):
        """ Root Mean Squared Scaled Error """
        q = np.abs(self._error()) / self.mae(self.true[seasonality:], self._naive_prognose(seasonality))
        return np.sqrt(np.mean(np.square(q)))

    def rmdspe(self):
        """
        Root Median Squared Percentage Error
        """
        return np.sqrt(np.median(np.square(self._percentage_error()))) * 100.0

    def inrse(self):
        """ Integral Normalized Root Squared Error """
        return np.sqrt(np.sum(np.square(self._error())) / np.sum(np.square(self.true - np.mean(self.true))))

    def rrse(self):
        """ Root Relative Squared Error """
        return np.sqrt(np.sum(np.square(self.true - self.predicted)) / np.sum(np.square(self.true-np.mean(self.true))))

    def mda(self):
        """ Mean Directional Accuracy
         modified after https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
         """
        dict_acc = np.sign(self.true[1:] - self.true[:-1]) == np.sign(self.predicted[1:] - self.predicted[:-1])
        return np.mean(dict_acc)

    def gmae(self):
        """ Geometric Mean Absolute Error """
        return _geometric_mean(np.abs(self._error()))

    def mpe(self):
        """ Mean Percentage Error """
        return np.mean(self._percentage_error())

    def mdape(self):
        """
        Median Absolute Percentage Error
        """
        return np.median(np.abs(self._percentage_error())) * 100

    def smdape(self):
        """
        Symmetric Median Absolute Percentage Error
        Note: result is NOT multiplied by 100
        """
        return np.median(2.0 * self._ae() / ((np.abs(self.true) + np.abs(self.predicted)) + EPS))

    def maape(self):
        """
        Mean Arctangent Absolute Percentage Error
        Note: result is NOT multiplied by 100
        """
        return np.mean(np.arctan(np.abs((self.true - self.predicted) / (self.true + EPS))))

    def norm_ae(self):
        """ Normalized Absolute Error """
        return np.sqrt(np.sum(np.square(self._error() - self.mae())) / (len(self.true) - 1))

    def norm_ape(self):
        """ Normalized Absolute Percentage Error """
        return np.sqrt(np.sum(np.square(self._percentage_error() - self.mape())) / (len(self.true) - 1))

    def rae(self):
        """ Relative Absolute Error (aka Approximation Error) """
        return np.sum(self._ae()) / (np.sum(np.abs(self.true - np.mean(self.true))) + EPS)

    def mrae(self, benchmark: np.ndarray = None):
        """ Mean Relative Absolute Error """
        return np.mean(np.abs(self._relative_error(benchmark)))

    def mdrae(self, benchmark: np.ndarray = None):
        """ Median Relative Absolute Error """
        return np.median(np.abs(self._relative_error(benchmark)))

    def gmrae(self, benchmark: np.ndarray = None):
        """ Geometric Mean Relative Absolute Error """
        return _geometric_mean(np.abs(self._relative_error(benchmark)))

    def mbrae(self, benchmark: np.ndarray = None):
        """ Mean Bounded Relative Absolute Error """
        return np.mean(self._bounded_relative_error(benchmark))

    def umbrae(self, benchmark: np.ndarray = None):
        """ Unscaled Mean Bounded Relative Absolute Error """
        return self.mbrae(benchmark) / (1 - self.mbrae(benchmark))

    def rmse(self, weights=None) -> float:
        """ root mean square error"""
        return sqrt(np.average((self.true - self.predicted) ** 2, axis=0,  weights=weights))

    def mse(self, weights=None) -> float:
        """ mean square error """
        return np.average((self.true - self.predicted) ** 2, axis=0,  weights=weights)

    def r2(self) -> float:
        """coefficient of determination"""
        zx = (self.true - np.mean(self.true)) / np.std(self.true, ddof=1)
        zy = (self.predicted - np.mean(self.predicted)) / np.std(self.predicted, ddof=1)
        r = np.sum(zx * zy) / (len(self.true) - 1)
        return r ** 2

    def r2_mod(self, weights=None) -> float:
        """
        This is not a symmetric function.
        Unlike most other scores, R^2 score may be negative (it need not actually
        be the square of a quantity R).
        This metric is not well-defined for single samples and will return a NaN
        value if n_samples is less than two.
        """

        if len(self.predicted) < 2:
            msg = "R^2 score is not well-defined with less than two samples."
            warnings.warn(msg)
            return float('nan')

        if weights is None:
            weight = 1.
        else:
            weight = weights[:, np.newaxis]

        numerator = (weight * (self.true - self.predicted) ** 2).sum(axis=0,
                                                                     dtype=np.float64)
        denominator = (weight * (self.true - np.average(
            self.true, axis=0, weights=weights)) ** 2).sum(axis=0, dtype=np.float64)

        output_scores = _foo(denominator, numerator)

        return np.average(output_scores, weights=weights)

    def rsr(self) -> float:
        return self.rmse() / np.std(self.true)

    def nse(self) -> float:
        _nse = 1 - sum((self.predicted - self.true) ** 2) / sum((self.true - np.mean(self.true)) ** 2)
        return _nse

    def abs_pbias(self) -> float:
        """ Absolute Percent bias"""
        _apb = 100.0 * sum(abs(self.predicted - self.true)) / sum(self.true)  # Absolute percent bias
        return _apb

    def pbias(self) -> float:
        """ Percent Bias"""
        return 100.0 * sum(self.predicted - self.true) / sum(self.true)  # percent bias

    def nrmse(self) -> float:
        """ Normalized Root Mean Squared Error """
        return self.rmse() / (self.true.max() - self.true.min())

    def mae(self, true=None, predicted=None):
        """ Mean Absolute Error """
        if true is None:
            true = self.true
        if predicted is None:
            predicted = self.predicted
        return np.mean(np.abs(true - predicted))

    def mape(self) -> float:
        """ Mean Absolute Percentage Error"""
        return np.mean(np.abs((self.true - self.predicted) / self.true)) * 100

    def smape(self) -> float:
        """
         Symmetric Mean Absolute Percentage Error
         https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
         https://stackoverflow.com/a/51440114/5982232
        """
        _temp = np.sum(2 * np.abs(self.predicted - self.true) / (np.abs(self.true) + np.abs(self.predicted)))
        return 100 / len(self.true) * _temp

    def wmape(self):
        """
         Weighted Mean Absolute Percent Error
         https://stackoverflow.com/a/54833202/5982232
        """
        # Take a series (actual) and a dataframe (forecast) and calculate wmape
        # for each forecast. Output shape is (1, num_forecasts)

        # Make an array of mape (same shape as forecast)
        se_mape = abs(self.true - self.predicted) / self.true

        # Calculate sum of actual values
        ft_actual_sum = self.true.sum(axis=0)

        # Multiply the actual values by the mape
        se_actual_prod_mape = self.true * se_mape

        # Take the sum of the product of actual values and mape
        # Make sure to sum down the rows (1 for each column)
        ft_actual_prod_mape_sum = se_actual_prod_mape.sum(axis=0)

        # Calculate the wmape for each forecast and return as a dictionary
        ft_wmape_forecast = ft_actual_prod_mape_sum / ft_actual_sum
        return ft_wmape_forecast

    def wape(self) -> float:
        """
        weighted absolute percentage error
        https://mattdyor.wordpress.com/2018/05/23/calculating-wape/
        """
        return float(np.sum(self._ae() / np.sum(self.true)))

    def mean_abs_rel_error(self) -> float:
        """ Mean Absolute Relative Error """
        return np.sum(self._ae(), axis=0, dtype=np.float64) / np.sum(self.true)

    def mean_bias_error(self) -> float:
        """
        It represents overall bias error or systematic error. It shows average interpolation bias; i.e. average over-
        or underestimation. [1][2]

    [2] Willmott, C. J., & Matsuura, K. (2006). On the use of dimensioned measures of error to evaluate the performance
        of spatial interpolators. International Journal of Geographical Information Science, 20(1), 89-102.
         https://doi.org/10.1080/1365881050028697
    [1] Valipour, M. (2015). Retracted: Comparative Evaluation of Radiation-Based Methods for Estimation of Potential
        Evapotranspiration. Journal of Hydrologic Engineering, 20(5), 04014068.
         http://dx.doi.org/10.1061/(ASCE)HE.1943-5584.0001066
         """
        return np.sum(self.true - self.predicted) / len(self.true)

    def bias(self) -> float:
        """
        Bias as shown in Gupta in Sorooshian (1998), Toward improved calibration of hydrologic models: 
        Multiple  and noncommensurable measures of information, Water Resources Research
            .. math::
            Bias=\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})
        """
        bias = np.nansum(self.true - self.predicted) / len(self.true)
        return float(bias)

    def log_nse(self, epsilon=0.0) -> float:
        """
        log Nash-Sutcliffe model efficiency
            .. math::
            NSE = 1-\\frac{\\sum_{i=1}^{N}(log(e_{i})-log(s_{i}))^2}{\\sum_{i=1}^{N}(log(e_{i})-log(\\bar{e})^2}-1)*-1
        """
        s, o = self.predicted + epsilon, self.true + epsilon
        return float(1 - sum((np.log(o) - np.log(o))**2) / sum((np.log(o) - np.mean(np.log(o)))**2))

    def log_prob(self) -> float:
        """
        Logarithmic probability distribution
        """
        scale = np.mean(self.true) / 10
        if scale < .01:
            scale = .01
        y = (self.true - self.predicted) / scale
        normpdf = -y**2 / 2 - np.log(np.sqrt(2 * np.pi))
        return float(np.mean(normpdf))

    def corr_coeff(self) -> float:
        """
        Correlation Coefficient
            .. math::
            r = \\frac{\\sum ^n _{i=1}(e_i - \\bar{e})(s_i - \\bar{s})}{\\sqrt{\\sum ^n _{i=1}(e_i - \\bar{e})^2}
             \\sqrt{\\sum ^n _{i=1}(s_i - \\bar{s})^2}}
        """
        correlation_coefficient = np.corrcoef(self.true, self.predicted)[0, 1]
        return correlation_coefficient

    def relative_rmse(self) -> float:
        """
        Relative Root Mean Squared Error
            .. math::   
            RRMSE=\\frac{\\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})^2}}{\\bar{e}}
        """ 
        rrmse = self.rmse() / np.mean(self.true)
        return rrmse

    def rmspe(self) -> float:
        """
        Root Mean Square Percentage Error
        https://stackoverflow.com/a/53166790/5982232
        """
        return np.sqrt(np.mean(np.square(((self.true - self.predicted) / self.true)), axis=0))

    def agreement_index(self) -> float:
        """
        Agreement Index (d) developed by Willmott (1981)
            .. math::   
            d = 1 - \\frac{\\sum_{i=1}^{N}(e_{i} - s_{i})^2}{\\sum_{i=1}^{N}(\\left | s_{i} - \\bar{e}
             \\right | + \\left | e_{i} - \\bar{e} \\right |)^2}
        """
        agreement_index = 1 - (np.sum((self.true - self.predicted)**2)) / (np.sum(
            (np.abs(self.predicted - np.mean(self.true)) + np.abs(self.true - np.mean(self.true)))**2))
        return agreement_index

    def covariance(self) -> float:
        """
        Covariance
            .. math::
            Covariance = \\frac{1}{N} \\sum_{i=1}^{N}((e_{i} - \\bar{e}) * (s_{i} - \\bar{s}))
        """
        obs_mean = np.mean(self.true)
        sim_mean = np.mean(self.predicted)
        covariance = np.mean((self.true - obs_mean)*(self.predicted - sim_mean))
        return float(covariance)

    def decomposed_mse(self) -> float:
        """
        Decomposed MSE developed by Kobayashi and Salam (2000)
            .. math ::
            dMSE = (\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i}))^2 + SDSD + LCS
            SDSD = (\\sigma(e) - \\sigma(s))^2
            LCS = 2 \\sigma(e) \\sigma(s) * (1 - \\frac{\\sum ^n _{i=1}(e_i - \\bar{e})(s_i - \\bar{s})}
            {\\sqrt{\\sum ^n _{i=1}(e_i - \\bar{e})^2} \\sqrt{\\sum ^n _{i=1}(s_i - \\bar{s})^2}})
        """
        e_std = np.std(self.true)
        s_std = np.std(self.predicted)

        bias_squared = self.bias()**2
        sdsd = (e_std - s_std)**2
        lcs = 2 * e_std * s_std * (1 - self.corr_coeff())

        decomposed_mse = bias_squared + sdsd + lcs

        return decomposed_mse

    def kge(self, return_all=False):
        """
        Kling-Gupta Efficiency 
        Gupta, Kling, Yilmaz, Martinez, 2009, Decomposition of the mean squared error and NSE performance
         criteria: Implications for improving hydrological modelling
        output:
            kge: Kling-Gupta Efficiency
            cc: correlation 
            alpha: ratio of the standard deviation
            beta: ratio of the mean
        """
        cc = np.corrcoef(self.true, self.predicted)[0, 1]
        alpha = np.std(self.predicted) / np.std(self.true)
        beta = np.sum(self.predicted) / np.sum(self.true)
        kge = 1 - np.sqrt((cc - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        if return_all:
            return np.vstack((kge, cc, alpha, beta))
        else:
            return kge

    def kge_np(self, return_all=False):
        """
        Non parametric Kling-Gupta Efficiency
        Corresponding paper:
        Pool, Vis, and Seibert, 2018 Evaluating model performance: towards a non-parametric variant of the
         Kling-Gupta efficiency, Hydrological Sciences Journal.
        https://doi.org/10.1080/02626667.2018.1552002
        output:
            kge: Kling-Gupta Efficiency
            cc: correlation 
            alpha: ratio of the standard deviation
            beta: ratio of the mean
        """
        # # self-made formula
        cc = _spearmann_corr(self.true, self.predicted)

        fdc_sim = np.sort(self.predicted / (np.nanmean(self.predicted)*len(self.predicted)))
        fdc_obs = np.sort(self.true / (np.nanmean(self.true)*len(self.true)))
        alpha = 1 - 0.5 * np.nanmean(np.abs(fdc_sim - fdc_obs))

        beta = np.mean(self.predicted) / np.mean(self.true)
        kge = 1 - np.sqrt((cc - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        if return_all:
            return np.vstack((kge, cc, alpha, beta))
        else:
            return kge

    def kge_mod(self, return_all=False):
        """
        Modified Kling-Gupta Efficiency (Kling et al. 2012 - https://doi.org/10.1016/j.jhydrol.2012.01.011)
        """
        # calculate error in timing and dynamics r (Pearson's correlation coefficient)
        sim_mean = np.mean(self.predicted, axis=0, dtype=np.float64)
        obs_mean = np.mean(self.true, dtype=np.float64)
        r = np.sum((self.predicted - sim_mean) * (self.true - obs_mean), axis=0, dtype=np.float64) / \
            np.sqrt(np.sum((self.predicted - sim_mean) ** 2, axis=0, dtype=np.float64) *
                    np.sum((self.true - obs_mean) ** 2, dtype=np.float64))
        # calculate error in spread of flow gamma (avoiding cross correlation with bias by dividing by the mean)
        gamma = (np.std(self.predicted, axis=0, dtype=np.float64) / sim_mean) / \
            (np.std(self.true, dtype=np.float64) / obs_mean)
        # calculate error in volume beta (bias of mean discharge)
        beta = np.mean(self.predicted, axis=0, dtype=np.float64) / np.mean(self.true, axis=0, dtype=np.float64)
        # calculate the modified Kling-Gupta Efficiency KGE'
        kgeprime_ = 1 - np.sqrt((r - 1) ** 2 + (gamma - 1) ** 2 + (beta - 1) ** 2)
        
        if return_all:
            return np.vstack((kgeprime_, r, gamma, beta))
        else:
            return kgeprime_

    def volume_error(self) -> float:
        """
        Returns the Volume Error (Ve).
        It is an indicator of the agreement between the averages of the simulated
        and observed runoff (i.e. long-term water balance).
        used in this paper:
        Reynolds, J.E., S. Halldin, C.Y. Xu, J. Seibert, and A. Kauffeldt. 2017.
        "Sub-Daily Runoff Predictions Using Parameters Calibrated on the Basis of Data with a 
        Daily Temporal Resolution."  Journal of Hydrology 550 (July):399?411. 
        https://doi.org/10.1016/j.jhydrol.2017.05.012.
            .. math::
            Sum(self.predicted- true)/sum(self.predicted)
        """
        # TODO written formula and executed formula are different.
        ve = np.sum(self.predicted - self.true) / np.sum(self.true)
        return float(ve)

    def mean_poisson_deviance(self, weights=None) -> float:
        """
        mean poisson deviance
        """
        return _mean_tweedie_deviance(self.true, self.predicted, weights=weights, power=1)

    def mean_gamma_deviance(self, weights=None):
        """
        mean gamma deviance
        """
        return _mean_tweedie_deviance(self.true, self.predicted, weights=weights, power=2)

    def median_abs_error(self) -> float:
        """
        median absolute error
        """
        return float(np.median(np.abs(self.predicted - self.true), axis=0))

    def mean_square_log_error(self, weights=None) -> float:
        """
        mean square logrithmic error
        """
        return np.average((np.log1p(self.true) - np.log1p(self.predicted)) ** 2, axis=0,  weights=weights)

    def max_error(self) -> float:
        """
        maximum error
        """
        return np.max(self._ae())

    def exp_var_score(self, weights=None) -> float:
        """
        Explained variance score
        https://stackoverflow.com/questions/24378176/python-sci-kit-learn-metrics-difference-between-r2-score-and-explained-varian
        best value is 1, lower values are less accurate.
        """
        y_diff_avg = np.average(self.true - self.predicted, weights=weights, axis=0)
        numerator = np.average((self.true - self.predicted - y_diff_avg) ** 2,
                               weights=weights, axis=0)

        y_true_avg = np.average(self.true, weights=weights, axis=0)
        denominator = np.average((self.true - y_true_avg) ** 2,
                                 weights=weights, axis=0)

        output_scores = _foo(denominator, numerator)

        return np.average(output_scores, weights=weights)

    def fdc_fhv(self, h: float = 0.02) -> float:
        """
        modified after: https://github.com/kratzert/ealstm_regional_modeling/blob/64a446e9012ecd601e0a9680246d3bbf3f002f6d/papercode/metrics.py#L190
        Peak flow bias of the flow duration curve (Yilmaz 2018).
        used in kratzert et al., 2018
        Returns
        -------
        float
            Bias of the peak flows

        Raises
        ------

        RuntimeError
            If `h` is not in range(0,1)
        """
        if (h <= 0) or (h >= 1):
            raise RuntimeError("h has to be in the range (0,1)")

        # sort both in descending order
        obs = -np.sort(-self.true)
        sim = -np.sort(-self.predicted)

        # subset data to only top h flow values
        obs = obs[:np.round(h * len(obs)).astype(int)]
        sim = sim[:np.round(h * len(sim)).astype(int)]

        fhv = np.sum(sim - obs) / (np.sum(obs) + 1e-6)

        return fhv * 100

    def nse_alpha(self) -> float:
        """
        Alpha decomposition of the NSE, see Gupta et al. 2009
        used in kratzert et al., 2018
        Returns
        -------
        float
            Alpha decomposition of the NSE

        """
        return np.std(self.predicted) / np.std(self.true)

    def nse_beta(self) -> float:
        """
        Beta decomposition of NSE. See Gupta et. al 2009
        used in kratzert et al., 2018
        Returns
        -------
        float
            Beta decomposition of the NSE
        """
        return (np.mean(self.predicted) - np.mean(self.true)) / np.std(self.true)

    def fdc_flv(self, low_flow: float = 0.3) -> float:
        """
        bias of the bottom 30 % low flows
        modified after: https://github.com/kratzert/ealstm_regional_modeling/blob/64a446e9012ecd601e0a9680246d3bbf3f002f6d/papercode/metrics.py#L237
        used in kratzert et al., 2018
        Parameters
        ----------
        low_flow : float, optional
            Upper limit of the flow duration curve. E.g. 0.3 means the bottom 30% of the flows are
            considered as low flows, by default 0.3

        Returns
        -------
        float
            Bias of the low flows.

        Raises
        ------
        RuntimeError
            If `low_flow` is not in the range(0,1)
        """

        low_flow = 1.0 - low_flow
        # make sure that metric is calculated over the same dimension
        obs = self.true.flatten()
        sim = self.predicted.flatten()

        if (low_flow <= 0) or (low_flow >= 1):
            raise RuntimeError("l has to be in the range (0,1)")

        # for numerical reasons change 0s to 1e-6
        sim[sim == 0] = 1e-6
        obs[obs == 0] = 1e-6

        # sort both in descending order
        obs = -np.sort(-obs)
        sim = -np.sort(-sim)

        # subset data to only top h flow values
        obs = obs[np.round(low_flow * len(obs)).astype(int):]
        sim = sim[np.round(low_flow * len(sim)).astype(int):]

        # transform values to log scale
        obs = np.log(obs + 1e-6)
        sim = np.log(sim + 1e-6)

        # calculate flv part by part
        qsl = np.sum(sim - sim.min())
        qol = np.sum(obs - obs.min())

        flv = -1 * (qsl - qol) / (qol + 1e-6)

        return flv * 100

    def nse_bound(self):
        """
        Bounded Version of the Nash-Sutcliffe Efficiency
        https://iahs.info/uploads/dms/13614.21--211-219-41-MATHEVET.pdf
        """
        nse_ = self.nse()
        nse_c2m_ = nse_ / (2 - nse_)

        return nse_c2m_

    def kge_bound(self):
        """
        Bounded Version of the Original Kling-Gupta Efficiency
        https://iahs.info/uploads/dms/13614.21--211-219-41-MATHEVET.pdf
        """
        kge_ = self.kge(return_all=True)[0, :]
        kge_c2m_ = kge_ / (2 - kge_)

        return kge_c2m_

    def kgeprime_c2m(self):
        """
        https://iahs.info/uploads/dms/13614.21--211-219-41-MATHEVET.pdf
         Bounded Version of the Modified Kling-Gupta Efficiency
        """
        kgeprime_ = self.kge_mod(return_all=True)[0, :]
        kgeprime_c2m_ = kgeprime_ / (2 - kgeprime_)

        return kgeprime_c2m_

    def kgenp_bound(self):
        """
        Bounded Version of the Non-Parametric Kling-Gupta Efficiency
        """
        kgenp_ = self.kge_np(return_all=True)[0, :]
        kgenp_c2m_ = kgenp_ / (2 - kgenp_)

        return kgenp_c2m_


def _foo(denominator, numerator):
    nonzero_numerator = numerator != 0
    nonzero_denominator = denominator != 0
    valid_score = nonzero_numerator & nonzero_denominator
    output_scores = np.ones(1)

    output_scores[valid_score] = 1 - (numerator[valid_score] /
                                      denominator[valid_score])
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.
    return output_scores


def _spearmann_corr(x, y):
    """Separmann correlation coefficient"""
    col = [list(a) for a in zip(x, y)]
    xy = sorted(col, key=lambda _x: _x[0], reverse=False)
    # rang of x-value
    for i, row in enumerate(xy):
        row.append(i+1)

    a = sorted(xy, key=lambda _x: _x[1], reverse=False)
    # rang of y-value
    for i, row in enumerate(a):
        row.append(i+1)

    mw_rank_x = np.nanmean(np.array(a)[:, 2])
    mw_rank_y = np.nanmean(np.array(a)[:, 3])

    numerator = np.nansum([float((a[j][2]-mw_rank_x)*(a[j][3]-mw_rank_y)) for j in range(len(a))])
    denominator1 = np.sqrt(np.nansum([(a[j][2]-mw_rank_x)**2. for j in range(len(a))]))
    denominator2 = np.sqrt(np.nansum([(a[j][3]-mw_rank_x)**2. for j in range(len(a))]))
    return float(numerator/(denominator1*denominator2))


def _mean_tweedie_deviance(y_true, y_pred, power=0, weights=None):
    # copying from https://github.com/scikit-learn/scikit-learn/blob/95d4f0841d57e8b5f6b2a570312e9d832e69debc/sklearn/metrics/_regression.py#L659

    message = ("Mean Tweedie deviance error with power={} can only be used on "
               .format(power))
    if power < 0:
        # 'Extreme stable', y_true any real number, y_pred > 0
        if (y_pred <= 0).any():
            raise ValueError(message + "strictly positive y_pred.")
        dev = 2 * (np.power(np.maximum(y_true, 0), 2 - power)
                   / ((1 - power) * (2 - power))
                   - y_true * np.power(y_pred, 1 - power)/(1 - power)
                   + np.power(y_pred, 2 - power)/(2 - power))
    elif power == 0:
        # Normal distribution, y_true and y_pred any real number
        dev = (y_true - y_pred)**2
    elif power < 1:
        raise ValueError("Tweedie deviance is only defined for power<=0 and "
                         "power>=1.")
    elif power == 1:
        # Poisson distribution, y_true >= 0, y_pred > 0
        if (y_true < 0).any() or (y_pred <= 0).any():
            raise ValueError(message + "non-negative y_true and strictly "
                             "positive y_pred.")
        dev = 2 * (xlogy(y_true, y_true/y_pred) - y_true + y_pred)
    elif power == 2:
        # Gamma distribution, y_true and y_pred > 0
        if (y_true <= 0).any() or (y_pred <= 0).any():
            raise ValueError(message + "strictly positive y_true and y_pred.")
        dev = 2 * (np.log(y_pred/y_true) + y_true/y_pred - 1)
    else:
        if power < 2:
            # 1 < p < 2 is Compound Poisson, y_true >= 0, y_pred > 0
            if (y_true < 0).any() or (y_pred <= 0).any():
                raise ValueError(message + "non-negative y_true and strictly "
                                           "positive y_pred.")
        else:
            if (y_true <= 0).any() or (y_pred <= 0).any():
                raise ValueError(message + "strictly positive y_true and "
                                           "y_pred.")

        dev = 2 * (np.power(y_true, 2 - power)/((1 - power) * (2 - power))
                   - y_true * np.power(y_pred, 1 - power)/(1 - power)
                   + np.power(y_pred, 2 - power)/(2 - power))

    return np.average(dev, weights=weights)


def _geometric_mean(a, axis=0, dtype=None):
    """ Geometric mean """
    if not isinstance(a, np.ndarray):  # if not an ndarray object attempt to convert it
        log_a = np.log(np.array(a, dtype=dtype))
    elif dtype:  # Must change the default dtype allowing array type
        if isinstance(a, np.ma.MaskedArray):
            log_a = np.log(np.ma.asarray(a, dtype=dtype))
        else:
            log_a = np.log(np.asarray(a, dtype=dtype))
    else:
        log_a = np.log(a)
    return np.exp(log_a.mean(axis=axis))


if __name__ == "__main__":

    t = np.random.random((20, 1))
    p = np.random.random((20, 1))

    er = FindErrors(t, p)

    all_errors = er.calculate_all()
