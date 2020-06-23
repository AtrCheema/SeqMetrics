from math import sqrt
from scipy.special import xlogy
from scipy.stats.stats import linregress
import numpy as np
import warnings

# TODO remove repeated calculation of mse, std, mean etc
# TODO make weights, class attribute
# TODO write tests


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
        assert len(predicted) == len(true), "lengths mismatch, predicted: {}, true: {}"\
            .format(len(predicted), len(true))

        return true, predicted
    
    def _assert_array(self, array_like) -> np.ndarray:

        if not isinstance(array_like, np.ndarray):
            if not isinstance(array_like, list):
                if array_like.__class__.__name__ in ['Series', 'DataFrame']:
                    if array_like.shape[1] > 1:
                        raise TypeError("only 1d pandas Series or dataframe are allowed")
                    np_array = np.array(array_like)
                else:
                    raise TypeError(" all inputs must be numpy array or list")
            else:
                np_array = np.array(array_like)
        else:
            # maybe the dimension is >1 so make sure it is more
            np_array = array_like.reshape(-1,)
        
        return np_array

    def calculate_all(self):
        """ calculates errors using all available methods"""
        errors = {}
        for m in self.all_methods:
            error = float(getattr(self, m)())
            errors[m] = error
            print('{0:25} :  {1:<12.3f}'.format(m, error))
        return errors

    def rmse(self, weights=None) -> float:
        """ root mean square error"""
        return sqrt(np.average((self.true - self.predicted) ** 2, axis=0,  weights=weights))

    def mse(self, weights=None) -> float:
        """ mean square error """
        return np.average((self.true - self.predicted) ** 2, axis=0,  weights=weights)

    def r2(self) -> float:
        """coefficient of determination"""
        slope, intercept, r_value, p_value, std_err = linregress(self.true, self.predicted)
        return r_value ** 2

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

    def abs_percent_bias(self) -> float:
        """ absolute percent bias"""
        _apb = 100.0 * sum(abs(self.predicted - self.true)) / sum(self.true)  # Absolute percent bias
        return _apb

    def percent_bias(self) -> float:
        pbias = 100.0 * sum(self.predicted - self.true) / sum(self.true)  # percent bias
        return pbias

    def norm_rmse(self) -> float:
        """ Normalized Root Mean Squared Error """
        return self.rmse() / (self.true.max() - self.true.min())

    def mean_abs_errore(self):
        """ Mean Absolute Error """
        return np.mean(np.abs(self.true - self.predicted))

    def mean_abs_rel_error(self) -> float:
        """ Mean Absolute Relative Error """
        mare_ = np.sum(np.abs(self.true - self.predicted), axis=0, dtype=np.float64) / np.sum(self.true)
        return mare_

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

    def agreementindex(self) -> float:
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
            return kge, cc, alpha, beta
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
            return kge, cc, alpha, beta
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
            return kgeprime_, r, gamma, beta
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
        return np.max(np.abs(self.true - self.predicted))

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

    def fdc_flv(self, l: float = 0.7) -> float:
        """
        bias of the bottom 30 % low flows
        modified after: https://github.com/kratzert/ealstm_regional_modeling/blob/64a446e9012ecd601e0a9680246d3bbf3f002f6d/papercode/metrics.py#L237
        used in kratzert et al., 2018
        Parameters
        ----------
        l : float, optional
            Upper limit of the flow duration curve. E.g. 0.7 means the bottom 30% of the flows are
            considered as low flows, by default 0.7

        Returns
        -------
        float
            Bias of the low flows.

        Raises
        ------
        RuntimeError
            If `l` is not in the range(0,1)
        """
        # make sure that metric is calculated over the same dimension
        obs = self.true.flatten()
        sim = self.predicted.flatten()

        if (l <= 0) or (l >= 1):
            raise RuntimeError("l has to be in the range (0,1)")

        # for numerical reasons change 0s to 1e-6
        sim[sim == 0] = 1e-6
        obs[obs == 0] = 1e-6

        # sort both in descending order
        obs = -np.sort(-obs)
        sim = -np.sort(-sim)

        # subset data to only top h flow values
        obs = obs[np.round(l * len(obs)).astype(int):]
        sim = sim[np.round(l * len(sim)).astype(int):]

        # transform values to log scale
        obs = np.log(obs + 1e-6)
        sim = np.log(sim + 1e-6)

        # calculate flv part by part
        qsl = np.sum(sim - sim.min())
        qol = np.sum(obs - obs.min())

        flv = -1 * (qsl - qol) / (qol + 1e-6)

        return flv * 100


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


if __name__ == "__main__":

    t = np.random.random((20, 1))
    p = np.random.random((20, 1))

    er = FindErrors(t, p)

    all_errors = er.calculate_all()
