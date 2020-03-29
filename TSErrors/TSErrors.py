from sklearn.metrics import mean_squared_error, mean_gamma_deviance, mean_poisson_deviance
from sklearn.metrics import median_absolute_error, mean_squared_log_error, max_error
from sklearn.metrics import explained_variance_score, mean_absolute_error
from math import sqrt
from scipy.stats.stats import linregress
import numpy as np


class FindErrors(object):
    """
     actual: ture/observed values, 1D array or list
     predicted: simulated values, 1D array or list
    """

    def __init__(self, actual, predicted):

        self.true, self.predicted = self._pre_process(actual, predicted)
    

    def _pre_process(self, true, predicted):

        predicted = self._assert_array(predicted)
        true = self._assert_array(true)
        assert len(predicted) == len(true)

        return true, predicted
    

    def _assert_array(self, array_like):

        if not isinstance(array_like, np.ndarray):
            if not isinstance(array_like, list):
                raise TypeError(" all inputs must be numpy array or list")
            else:
                np_array = np.array(array_like)
        else:
            # maybe the dimension is >1 so make sure it is more
            np_array = array_like.reshape(-1,)
        
        return np_array



    def rmse(self):
        return sqrt(mean_squared_error(self.true, self.predicted))

    def mse(self):
        return mean_squared_error(self.true, self.predicted)

    def r2(self):
        """coefficient of determination"""
        slope, intercept, r_value, p_value, std_err = linregress(self.true, self.predicted)
        return r_value ** 2

    def rsr(self):
        return self.rmse() / np.std(self.true)

    def nse(self):
        _nse = 1 - sum((self.predicted - self.true) ** 2) / sum((self.true - np.mean(self.true)) ** 2)
        return _nse

    def APB(self):
        """ absolute percent bias"""
        APB = 100.0 * sum(abs(self.predicted - self.true)) / sum(self.true)  # Absolute percent bias
        return APB

    def percent_bias(self):
        PerBias = 100.0 * sum(self.predicted - self.true) / sum(self.true)  # percent bias
        return PerBias

    def nrmse(self):
        """ Normalized Root Mean Squared Error """
        return self.rmse() / (self.true.max() - self.true.min())

    def mae(self):
        """ Mean Absolute Error """
        return np.mean(np.abs(self.true - self.predicted))

    
    def mare(self):
        """ Mean Absolute Relative Error """
        mare_ = np.sum(np.abs(self.true - self.predicted), axis=0, dtype=np.float64) / np.sum(self.true)
        return mare_


    def bias(self):
        """
        Bias as shown in Gupta in Sorooshian (1998), Toward improved calibration of hydrologic models: 
        Multiple  and noncommensurable measures of information, Water Resources Research
            .. math::
            Bias=\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})
        """
        bias = np.nansum(self.true - self.predicted) / len(self.true)
        return float(bias)


    def log_nse(self, epsilon=0.0):
        """
        log Nash-Sutcliffe model efficiency
            .. math::
            NSE = 1-\\frac{\\sum_{i=1}^{N}(log(e_{i})-log(s_{i}))^2}{\\sum_{i=1}^{N}(log(e_{i})-log(\\bar{e})^2}-1)*-1
        """
        s, o = self.predicted + epsilon, self.true + epsilon
        return float(1 - sum((np.log(o) - np.log(o))**2) / sum((np.log(o) - np.mean(np.log(o)))**2))

    def log_p(self):
        """
        Logarithmic probability distribution
        """
        scale = np.mean(self.true) / 10
        if scale < .01:
            scale = .01
        y = (self.true - self.predicted) / scale
        normpdf = -y**2 / 2 - np.log(np.sqrt(2 * np.pi))
        return np.mean(normpdf)


    def corr_coeff(self):
        """
        Correlation Coefficient
            .. math::
            r = \\frac{\\sum ^n _{i=1}(e_i - \\bar{e})(s_i - \\bar{s})}{\\sqrt{\\sum ^n _{i=1}(e_i - \\bar{e})^2} \\sqrt{\\sum ^n _{i=1}(s_i - \\bar{s})^2}}
        """
        correlation_coefficient = np.corrcoef(self.true, self.predicted)[0, 1]
        return correlation_coefficient


    def rrmse(self):
        """
        Relative Root Mean Squared Error
            .. math::   
            RRMSE=\\frac{\\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})^2}}{\\bar{e}}
        """ 
        rrmse = self.rmse() / np.mean(self.true)
        return rrmse


    def agreementindex(self):
        """
        Agreement Index (d) developed by Willmott (1981)
            .. math::   
            d = 1 - \\frac{\\sum_{i=1}^{N}(e_{i} - s_{i})^2}{\\sum_{i=1}^{N}(\\left | s_{i} - \\bar{e} \\right | + \\left | e_{i} - \\bar{e} \\right |)^2}  
        """
        Agreement_index = 1 - (np.sum((self.true - self.predicted)**2)) / (np.sum(
            (np.abs(self.predicted - np.mean(self.true)) + np.abs(self.true - np.mean(self.true)))**2))
        return Agreement_index


    def covariance(self):
        """
        Covariance
            .. math::
            Covariance = \\frac{1}{N} \\sum_{i=1}^{N}((e_{i} - \\bar{e}) * (s_{i} - \\bar{s}))
        """
        obs_mean = np.mean(self.true)
        sim_mean = np.mean(self.predicted)
        covariance = np.mean((self.true - obs_mean)*(self.predicted - sim_mean))
        return covariance


    def decomposed_mse(self):
        """
        Decomposed MSE developed by Kobayashi and Salam (2000)
            .. math ::
            dMSE = (\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i}))^2 + SDSD + LCS
            SDSD = (\\sigma(e) - \\sigma(s))^2
            LCS = 2 \\sigma(e) \\sigma(s) * (1 - \\frac{\\sum ^n _{i=1}(e_i - \\bar{e})(s_i - \\bar{s})}{\\sqrt{\\sum ^n _{i=1}(e_i - \\bar{e})^2} \\sqrt{\\sum ^n _{i=1}(s_i - \\bar{s})^2}})
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
        Gupta, Kling, Yilmaz, Martinez, 2009, Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling
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
        Pool, Vis, and Seibert, 2018 Evaluating model performance: towards a non-parametric variant of the Kling-Gupta efficiency, Hydrological Sciences Journal.
        https://doi.org/10.1080/02626667.2018.1552002
        output:
            kge: Kling-Gupta Efficiency
            cc: correlation 
            alpha: ratio of the standard deviation
            beta: ratio of the mean
        """
        ## self-made formula 
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


    def volume_error(self):
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


    def mpd(self):
        """
        mean poisson deviance
        """
        return mean_poisson_deviance(self.true, self.predicted)


    def mgd(self):
        """
        mean gamma deviance
        """
        return mean_gamma_deviance(self.true, self.predicted)


    def med_ae(self):
        """
        median absolute error
        """
        return median_absolute_error(self.true, self.predicted)


    def msle(self):
        """
        mean square logrithmic error
        """
        return mean_squared_log_error(self.true, self.predicted)


    def max_err(self):
        """
        maximum error
        """
        return max_error(self.true, self.predicted)


    def exp_var_score(self):
        """
        Explained variance score
        https://stackoverflow.com/questions/24378176/python-sci-kit-learn-metrics-difference-between-r2-score-and-explained-varian
        best value is 1, lower values are less accurate.
        """
        return explained_variance_score(self.true, self.predicted)


    
def _spearmann_corr(x, y):
    """Separmann correlation coefficient"""
    col = [list(a) for a in zip(x, y)]
    xy = sorted(col, key=lambda x: x[0], reverse=False)
    # rang of x-value
    for i, row in enumerate(xy):
        row.append(i+1)

    a = sorted(xy, key=lambda x: x[1], reverse=False)
    # rang of y-value
    for i, row in enumerate(a):
        row.append(i+1)

    MW_rank_x = np.nanmean(np.array(a)[:,2])
    MW_rank_y = np.nanmean(np.array(a)[:,3])

    numerator = np.nansum([float((a[j][2]-MW_rank_x)*(a[j][3]-MW_rank_y)) for j in range(len(a))])
    denominator1 = np.sqrt(np.nansum([(a[j][2]-MW_rank_x)**2. for j in range(len(a))]))
    denominator2 = np.sqrt(np.nansum([(a[j][3]-MW_rank_x)**2. for j in range(len(a))]))
    return float(numerator/(denominator1*denominator2))


if __name__ == "__main__":

    true = np.random.random((20, 1))
    pred = np.random.random((20, 1))

    er = FindErrors(true, pred)

    er_methods = [method for method in dir(er) if callable(getattr(er, method)) if
                               not method.startswith('_')]

    for m in er_methods:
        print('{0:15} :  {1:<12.3f}'.format(m, float(getattr(er, m)())))