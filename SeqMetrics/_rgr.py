
# https://github.com/aziele/statistical-distance/blob/main/distance.py
# https://github.com/pyRiemann/pyRiemann/blob/master/pyriemann/utils/distance.py

import warnings
from math import sqrt
from typing import Union

import numpy as np

try:
    from scipy.stats import kendalltau
    from scipy.signal import find_peaks
except (ImportError, ModuleNotFoundError):
    kendalltau, find_peaks = None, None

from .utils import maybe_treat_arrays
from .utils import _geometric_mean, _mean_tweedie_deviance, _foo, list_subclass_methods
from ._main import Metrics, EPS, ERR_STATE


class RegressionMetrics(Metrics):
    """
    Calculates more than 100 regression performance metrics related to sequence data.

    Example
    -------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> errors = RegressionMetrics(t,p)
        >>> all_errors = errors.calculate_all()
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes ``Metrics``.

        args and kwargs go to parent class :py:class:`SeqMetrics.Metrics`.
        """
        super().__init__(*args, **kwargs)
        self.all_methods: list = list_subclass_methods(RegressionMetrics, True,
                                                       additional_ignores=['calculate_hydro_metrics',
                                                                           # 'calculate_scale_dependent_metrics',
                                                                           # 'calculate_scale_independent_metrics'
                                                                           ])

        if kendalltau is None and 'kendall_tau' in self.all_methods:
            self.all_methods.remove('kendall_tau')
        if find_peaks is None and 'mape_for_peaks' in self.all_methods:
            self.all_methods.remove('mape_for_peaks')

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

    def _hydro_metrics(self) -> list:
        """Names of metrics related to hydrology"""

        return self._minimal() + [
            'fdc_flv', 'fdc_fhv',
            'kge', 'kge_np', 'kge_mod', 'kge_bound', 'kgeprime_bound', 'kgenp_bound',
            'nse', 'nse_alpha', 'nse_beta', 'nse_mod', 'nse_bound']

    @staticmethod
    def _scale_independent_metrics() -> list:
        """Names of scale independent metrics."""
        return ['mape', 'r2', 'nse']

    @staticmethod
    def _scale_dependent_metrics() -> list:
        """Names of scale dependent metrics."""
        return ['mse', 'rmse', 'mae']

    @staticmethod
    def _minimal() -> list:
        """some minimal and basic metrics"""

        return ['r2', 'mape', 'nrmse', 'corr_coeff', 'rmse', 'mae', 'mse', 'mpe',
                'mase', 'r2_score']

    # def abs_pbias(self) -> float:
    #     """Absolute Percent bias
    #
    #     Examples
    #     ---------
    #     >>> import numpy as np
    #     >>> from SeqMetrics import RegressionMetrics
    #     >>> t = np.random.random(10)
    #     >>> p = np.random.random(10)
    #     >>> metrics= RegressionMetrics(t, p)
    #     >>> metrics.abs_pbias()
    #     """
    #     return abs_pbias(true=self.true, predicted=self.predicted, treat_arrays=False)

    def acc(self) -> float:
        """Anomaly correction coefficient. See `Langland et al., 2012 <https://doi.org/10.3402/tellusa.v64i0.17531>`_;
        `Miyakoda_ et al., 1972 <https://doi.org/10.1080/02723646.1972.10642213>`_
        and Murphy_ et al., 1989.

        .. math::
            ACC = \\frac{\\sum_{i=1}^{N} \\left( (\\text{predicted}_i - \\overline{\\text{predicted}})(\\text{true}_i - \\overline{\\text{true}}) \\right)}{(N-1) \\cdot \\sigma_{\\text{true}} \\cdot \\sigma_{\\text{predicted}}}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.acc()
        """
        return acc(true=self.true, predicted=self.predicted, treat_arrays=False)

    def adjusted_r2(self) -> float:
        """Adjusted R squared.

        .. math::
            \\text{Adjusted } R^2 = 1 - \\left( \\frac{(1 - R^2) \\cdot (n - 1)}{n - k - 1} \\right)

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.adjusted_r2()
        """
        return adjusted_r2(true=self.true, predicted=self.predicted, treat_arrays=False)

    def agreement_index(self) -> float:
        """
        Agreement Index (d) developed by Willmott_, 1981.

        It detects additive and pro-portional differences in the observed and
        simulated means and vari-ances Moriasi_ et al., 2015. It is overly sensitive
        to extreme values due to the squared differences. It can also be used
        as a substitute for R2 to identify the degree to which model predic-tions
        are error-free.

        .. math::
            d = 1 - \\frac{\\sum_{i=1}^{N}(e_{i} - s_{i})^2}{\\sum_{i=1}^{N}(\\left | s_{i} - \\bar{e}
             \\right | + \\left | e_{i} - \\bar{e} \\right |)^2}

        .. _Willmott:
            https://doi.org/10.1080/02723646.1981.10642213

        .. _Moriasi:
            https://doi.org/10.13031/trans.58.10715

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.agreement_index()
        """
        return agreement_index(true=self.true, predicted=self.predicted, treat_arrays=False)

    def aic(self, p=1) -> float:
        """
        `Akaike_ Information Criterion <https://doi.org/10.1007/978-1-4612-1694-0_15>`_. Modifying from this `source <https://github.com/UBC-MDS/RegscorePy/blob/master/RegscorePy/aic.py>`_

        .. math::
            AIC = n \\cdot \\ln\\left(\\frac{\\sum_{i=1}^{n} (\\text{true}_i - \\text{predicted}_i)^2}{n}\\right) + 2p

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.aic( )
        """
        return aic(true=self.true, predicted=self.predicted, treat_arrays=False, p=p)

    def aitchison(self, center='mean') -> float:
        """Aitchison distance. used in `Zhang et al., 2020 <https://doi.org/10.5194/hess-24-2505-2020>`_

        .. math::
            d_{\\text{Aitchison}} = \\sqrt{\\sum_{i=1}^{n} \\left( \\log(\\text{true}_i) - \\text{center}(\\log(\\text{true})) - \\left(\\log(\\text{predicted}_i) - \\text{center}(\\log(\\text{predicted}))\\right) \\right)^2}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.aitchison( )
        """
        return aitchison(true=self.true, predicted=self.predicted, treat_arrays=False,
                         center=center)

    def amemiya_adj_r2(self) -> float:
        """Amemiya's Adjusted R-squared

        .. math::
            R^2_{\\text{adj, Amemiya}} = 1 - \\left( \\frac{(1 - R^2) \\cdot (n + k)}{n - k - 1} \\right)

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.amemiya_adj_r2( )
        """
        return amemiya_adj_r2(true=self.true, predicted=self.predicted, treat_arrays=False)

    def amemiya_pred_criterion(self) -> float:
        """Amemiya's Prediction Criterion

        .. math::
            \\text{APC} = \\left( \\frac{n + k}{n - k} \\right) \\left( \\frac{1}{n} \\sum_{i=1}^{n} (\\text{true}_i - \\text{predicted}_i)^2 \\right)

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.amemiya_pred_criterion()
        """
        return amemiya_pred_criterion(true=self.true, predicted=self.predicted,
                                      treat_arrays=False)

    def bias(self) -> float:
        """
        Bias as and given by `Gupta1998 et al., 1998 <https://doi.org/10.1029/97WR03495>`_

        .. math::
            Bias=\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.bias()
        """
        return bias(true=self.true, predicted=self.predicted, treat_arrays=False)

    def bic(self, p=1) -> float:
        """
        Bayesian Information Criterion

        Minimising the BIC_ is intended to give the best model. The
        model chosen by the BIC is either the same as that chosen by the AIC, or one
        with fewer terms. This is because the BIC penalises the number of parameters
        more heavily than the AIC.  Modified after RegscorePy_.

        .. math::
            BIC = n \\cdot \\ln\\left(\\frac{\\text{SSE}}{n}\\right) + p \\cdot \\ln(n)

        .. _BIC:
            https://otexts.com/fpp2/selecting-predictors.html#schwarzs-bayesian-information-criterion

        .. _RegscorePy:
            https://github.com/UBC-MDS/RegscorePy/blob/master/RegscorePy/bic.py

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.bic()
        """
        return bic(true=self.true, predicted=self.predicted, treat_arrays=False, p=p)

    def brier_score(self) -> float:
        """
        Adopted from SkillMetrics_
        Calculates the Brier score (BS), a measure of the mean-square error of
        probability forecasts for a dichotomous (two-category) event, such as
        the occurrence/non-occurrence of precipitation. The score is calculated
        using the formula:

        .. math::
            BS = sum_(n=1)^N (f_n - o_n)^2/N

        where f is the forecast probabilities, o is the observed probabilities
        (0 or 1), and N is the total number of values in f & o. Note that f & o
        must have the same number of values, and those values must be in the
        range_ [0,1].

        Returns
        --------
        float
            BS : Brier score

        References
        ---------
        `Glenn W. Brier, 1950: Verification of forecasts expressed in terms
        of probabilities. Mon. We. Rev., 78, 1-23.
        D. S. Wilks, 1995: Statistical Methods in the Atmospheric Sciences.
        Cambridge Press. 547 pp. <https://viterbi-web.usc.edu/~shaddin/teaching/cs699fa17/docs/Brier50.pdf>`_

        .. _SkillMetrics:
            https://github.com/PeterRochford/SkillMetrics/blob/master/skill_metrics/brier_score.py

        .. _range:
            https://data.library.virginia.edu/a-brief-on-brier-scores/

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.brier_score()

        """
        return brier_score(true=self.true, predicted=self.predicted, treat_arrays=False)

    def corr_coeff(self) -> float:
        """
        `Pearson correlation coefficient <https://royalsocietypublishing.org/doi/abs/10.1098/rsta.1895.0010>`_.
        It measures linear correlatin between true and predicted arrays.
        It is sensitive to outliers.

        .. math::
            r = \\frac{\\sum ^n _{i=1}(e_i - \\bar{e})(s_i - \\bar{s})}{\\sqrt{\\sum ^n _{i=1}(e_i - \\bar{e})^2}
             \\sqrt{\\sum ^n _{i=1}(s_i - \\bar{s})^2}}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.corr_coeff()

        """
        return corr_coeff(true=self.true, predicted=self.predicted, treat_arrays=False)

    def covariance(self) -> float:
        """
        `Covariance <https://scikit-learn.org/stable/api/sklearn.covariance.html>`_
            .. math::
            Covariance = \\frac{1}{N} \\sum_{i=1}^{N}((e_{i} - \\bar{e}) * (s_{i} - \\bar{s}))

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.covariance()
        """
        return covariance(true=self.true, predicted=self.predicted, treat_arrays=False)

    def cronbach_alpha(self) -> float:
        """
        It is a measure of internal consitency of data. See ucla and stackoverflow_
        pages for more info.

        .. math::
            alpha = \\frac{N}{N - 1} \\left(1 - \\frac{\\sum_{i=1}^{N} \\sigma^2_{i}}{\\sigma^2_{\\text{total}}}\\right)

        .. _ucla:
            https://stats.idre.ucla.edu/spss/faq/what-does-cronbachs-alpha-mean/

        .. _stackoverflow:
            https://stackoverflow.com/a/20799687/5982232

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.cronbach_alpha()
        """
        return cronbach_alpha(true=self.true, predicted=self.predicted, treat_arrays=False)

    def centered_rms_dev(self) -> float:
        """
        Modified after SkillMetrics_.
        Calculates the centered root-mean-square (RMS) difference between true and predicted
        using the formula:
        (E')^2 = sum_(n=1)^N [(p_n - mean(p))(r_n - mean(r))]^2/N
        where p is the predicted values, r is the true values, and
        N is the total number of values in p & r.

        .. math::
            CRMSD = \\sqrt{\\frac{1}{N} \\sum_{i=1}^{N} \\left( (p_i - \\text{mean}(p)) - (r_i - \\text{mean}(r)) \\right)^2}

        Output:
        CRMSDIFF : centered root-mean-square (RMS) difference (E')^2

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.centered_rms_dev()
        """
        return centered_rms_dev(true=self.true, predicted=self.predicted, treat_arrays=False)

    def cosine_similarity(self) -> float:
        """
        It is a judgment of orientation and not magnitude: two vectors with
        the same orientation have a cosine similarity of 1, two vectors oriented
        at 90° relative to each other have a similarity of 0, and two vectors diametrically
        opposed have a similarity of -1, independent of their magnitude. `See <https://en.wikipedia.org/wiki/Cosine_similarity>`_

        .. math::
            \\text{Cosine Similarity} = \\frac{\\sum_{i=1}^{n} \\text{true}_i \\cdot \\text{predicted}_i}{\\sqrt{\\sum_{i=1}^{n} (\\text{true}_i)^2} \\cdot \\sqrt{\\sum_{i=1}^{n} (\\text{predicted}_i)^2}}

        References
        ----------
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.cosine_similarity()
        """
        return cosine_similarity(true=self.true, predicted=self.predicted, treat_arrays=False)

    def decomposed_mse(self) -> float:
        """
        Decomposed MSE developed by Kobayashi and Salam (2000)

        .. math ::
            dMSE = (\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i}))^2 + SDSD + LCS

        .. math::
            SDSD = (\\sigma(e) - \\sigma(s))^2

        .. math::
            LCS = 2 \\sigma(e) \\sigma(s) * (1 - \\frac{\\sum ^n _{i=1}(e_i - \\bar{e})(s_i - \\bar{s})}
            {\\sqrt{\\sum ^n _{i=1}(e_i - \\bar{e})^2} \\sqrt{\\sum ^n _{i=1}(s_i - \\bar{s})^2}})

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.decomposed_mse()
        """
        return decomposed_mse(true=self.true, predicted=self.predicted, treat_arrays=False)

    def euclid_distance(self) -> float:
        """ `Euclidian distance <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html>`_
        taken from `this book <https://doi.org/10.1016/B978-0-12-088735-4.50006-7`_.

        .. math::
            D = \\sqrt{\\sum_{i=1}^{n} (\\text{true}_i - \\text{predicted}_i)^2}

        Referneces: Kennard et al., 2010

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.euclid_distance()
        """
        return euclid_distance(true=self.true, predicted=self.predicted, treat_arrays=False)

    def exp_var_score(self, weights=None) -> Union[float, None]:
        """
        Explained variance `score <https://stackoverflow.com/q/24378176/5982232>`_ . Best value is 1, lower values are less accurate.

        .. math::
            \\text{EVS} = 1 - \\frac{\\sum_{i=1}^{n} w_i \\left( (true_i - predicted_i) - \\frac{\\sum_{j=1}^{n} w_j (true_j - predicted_j)}{\\sum_{j=1}^{n} w_j} \\right)^2}{\\sum_{i=1}^{n} w_i (true_i - \\frac{\\sum_{j=1}^{n} w_j true_j}{\\sum_{j=1}^{n} w_j})^2}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.exp_var_score()
        """

        return exp_var_score(true=self.true, predicted=self.predicted, treat_arrays=False,
                             weights=weights)

    def expanded_uncertainty(self, cov_fact=1.96) -> float:
        """By default, it calculates uncertainty with 95% confidence interval.
        1.96 is the coverage factor corresponding 95% confidence level .This
        indicator is used in order to show more information about the model
        deviation. Using formula from by Behar_ et al., 2015 and Gueymard_ et al., 2014.

        .. math::
            U = \\text{cov_fact} \\times \\sqrt{\\frac{1}{n-1} \\sum_{i=1}^{n} \\left( \\left(\\text{true}_i - \\text{predicted}_i\\right) - \\overline{\\left(\\text{true} - \\text{predicted}\\right)} \\right)^2 + \\frac{1}{n} \\sum_{i=1}^{n} \\left(\\text{true}_i - \\text{predicted}_i\\right)^2}

        .. _Behar:
            https://doi.org/10.1016/j.enconman.2015.03.067

        .. _Gueymard:
            https://doi.org/10.1016/j.rser.2014.07.117

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.expanded_uncertainty()
        """
        return expanded_uncertainty(true=self.true, predicted=self.predicted,
                                    treat_arrays=False, cov_fact=cov_fact)

    def fdc_fhv(self, h: float = 0.02) -> float:
        """
        modified `Kratzert2018 <https://github.com/kratzert/ealstm_regional_modeling/blob/64a446e9012ecd601e0a9680246d3bbf3f002f6d/papercode/metrics.py#L190>`_
        code. Peak flow bias of the flow duration curve (Yilmaz 2008).
        used in `kratzert et al., 2019 <https://hess.copernicus.org/articles/23/5089/2019/hess-23-5089-2019.html>`_.


        .. math::
            FHV = \\frac{\\sum_{i=1}^{k} (predicted_i - true_i)}{\\sum_{i=1}^{k} true_i} \\times 100

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.fdc_fhv()
        """
        return fdc_fhv(true=self.true, predicted=self.predicted, treat_arrays=False, h=h)

    def fdc_flv(self, low_flow: float = 0.3) -> float:
        """
        bias of the bottom 30 % low flows. modified Kratzert_ code
        used in `kratzert et al., 2019 <https://hess.copernicus.org/articles/23/5089/2019/hess-23-5089-2019.html>`_.

        .. math::
            \\text{FLV} = -1 \\times \\frac{\\sum (\\log(\\text{predicted}) - \\min(\\log(\\text{predicted}))) - \\sum (\\log(\\text{true}) - \\min(\\log(\\text{true})))}{\\sum (\\log(\\text{true}) - \\min(\\log(\\text{true}))) + 1 \\times 10^{-6}}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.fdc_flv()
        """
        return fdc_flv(true=self.true, predicted=self.predicted, treat_arrays=False,
                       low_flow=low_flow)

    def gmae(self) -> float:
        """ `Geometric Mean Absolute Error <https://doi.org/10.1016/j.isprsjprs.2024.04.015>`_

        .. math::
            GMAE = \\left( \\prod_{i=1}^{n} \\left| \\text{true}_i - \\text{predicted}_i \\right| \\right)^{\\frac{1}{n}}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.gmae()
        """
        return gmae(true=self.true, predicted=self.predicted, treat_arrays=False)

    def gmean_diff(self) -> float:
        """
        `Geometric mean difference <https://www.sciencedirect.com/science/article/abs/pii/S0022316624002281>`_.
         First geometric mean is calculated for each
        of two samples and their difference is calculated.

        .. math::
            \\text{gmean_diff} = \\left( \\prod_{i=1}^{n} \\text{true}_i \\right)^{\\frac{1}{n}} - \\left( \\prod_{i=1}^{n} \\text{predicted}_i \\right)^{\\frac{1}{n}}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.gmean_diff()
        """
        return gmean_diff(true=self.true, predicted=self.predicted, treat_arrays=False)

    def gmrae(self, benchmark: np.ndarray = None) -> float:
        """ Geometric Mean Relative Absolute Error

        .. math::
            GMRAE = \\left( \\prod_{i=1}^{n} \\frac{|true_i - predicted_i|}{|true_i - benchmark_i|} \\right)^{\\frac{1}{n}}


        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.gmrae()
        """
        return gmrae(true=self.true, predicted=self.predicted, treat_arrays=False,
                     benchmark=benchmark)

    def calculate_hydro_metrics(self):
        """
        Calculates the following performance metrics related to hydrology.
            - fdc_flv
            - fdc_fhv
            - kge
            - kge_np
            - kge_mod
            - kge_bound
            - kgeprime_bound
            - kgenp_bound
            - nse
            - nse_alpha
            - nse_beta
            - nse_mod
            - nse_bound
            - r2
            - mape
            - nrmse
            - corr_coeff
            - rmse
            - mae
            - mse
            - mpe
            - mase
            - r2_score

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.calculate_hydro_metrics()
        """
        return calculate_hydro_metrics(true=self.true, predicted=self.predicted,
                                       treat_arrays=False)

    def inrse(self) -> float:
        """ `Integral Normalized Root Squared Error <https://doi.org/10.1016/j.engappai.2023.107559>`_

        .. math::
            IN\\text{-}RSE = \\sqrt{\\frac{\\sum_{i=1}^{n} (\\text{true}_i - \\text{predicted}_i)^2}{\\sum_{i=1}^{n} (\\text{true}_i - \\overline{\\text{true}})^2}}


        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.inrse()

        """
        return inrse(true=self.true, predicted=self.predicted, treat_arrays=False)

    def irmse(self) -> float:
        """ `Inertial RMSE <https://link.springer.com/article/10.1007/s11069-008-9299-2>`_.
        RMSE divided by standard deviation of the gradient of true.

        .. math::
            \\text{IRMSE} = \\frac{\\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} \\left( \\text{true}_i - \\text{predicted}_i \\right)^2}}{\\sqrt{\\frac{1}{n-2} \\sum_{i=1}^{n-1} \\left( (\\text{true}_{i+1} - \\text{true}_i) - \\overline{(\\text{true}_{i+1} - \\text{true}_i)} \\right)^2}}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.irmse()
        """
        return irmse(true=self.true, predicted=self.predicted, treat_arrays=False)

    def JS(self) -> float:
        """Jensen-shannon divergence

        .. math::
            JS(P \parallel Q) = \\frac{1}{2} \\sum_{i} \\left( P(i) \\log_2 \\left( \\frac{2P(i)}{P(i) + Q(i)} \\right) + Q(i) \\log_2 \\left( \\frac{2Q(i)}{P(i) + Q(i)} \\right) \\right)

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.JS()
        """
        return JS(true=self.true, predicted=self.predicted, treat_arrays=False)

    def kendall_tau(self, return_p=False) -> Union[float, tuple]:
        """Kendall's tau_ .used in Probst_ et al., 2019.

        .. math::
            tau = \\frac{(C - D)}{\\sqrt{(C + D + T_{\\text{true}})(C + D + T_{\\text{predicted}})}}

        .. _tau:
            https://machinelearningmastery.com/how-to-calculate-nonparametric-rank-correlation-in-python/

        .. _Probst:
            https://www.jmlr.org/papers/volume20/18-444/18-444.pdf

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.kendall_tau()
        """
        if kendalltau is None:
            raise NotImplementedError("scipy is not installed. Please install scipy to use this method")

        return kendall_tau(true=self.true, predicted=self.predicted, return_p=return_p,
                            treat_arrays=False)

    def kge(self):
        """
        Kling-Gupta Efficiency following `Gupta_ et al. 2009 <https://doi.org/10.1016/j.jhydrol.2009.08.003>`_.

        .. math::
            \\text{KGE} = 1 - \\sqrt{(r - 1)^2 + (\\alpha - 1)^2 + (\\beta - 1)^2}
        .. math::
            \\r = \\frac{\\sum_{i=1}^{N} ( \\text{true}_i - \\bar{\\text{true}} ) ( \\text{predicted}_i - \\bar{\\text{predicted}} )}{\\sqrt{\\sum_{i=1}^{N} ( \\text{true}_i - \\bar{\\text{true}} )^2} \\sqrt{\\sum_{i=1}^{N} ( \\text{predicted}_i - \\bar{\\text{predicted}} )^2}}
        .. math::
            \\alpha = \\frac{\\sigma_{\\text{predicted}}}{\\sigma_{\\text{true}}}
        .. math::
            \\beta = \\frac{\\mu_{\\text{predicted}}}{\\mu_{\\text{true}}}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.kge()
        """
        return kge(true=self.true, predicted=self.predicted, treat_arrays=False)

    def kge_bound(self) -> float:
        """
        Bounded Version of the Original Kling-Gupta Efficiency after
        `Mathevet et al. 2006 <https://iahs.info/uploads/dms/13614.21--211-219-41-MATHEVET.pdf>`_.

        .. math::
            \\text{KGE}_{\\text{bound}} = \\frac{\\text{KGE}}{2 - \\text{KGE}}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.kge_bound()
        """
        return kge_bound(true=self.true, predicted=self.predicted, treat_arrays=False)

    def kge_mod(self):
        """
        Modified Kling-Gupta Efficiency after `Kling et al. 2012 <https://doi.org/10.1016/j.jhydrol.2012.01.011>`_.

        .. math::
            \\text{KGE}_{\\text{mod}} = 1 - \\sqrt{ \\left( \\frac{\\sum_{i=1}^{n} (true_i - \\bar{true})(predicted_i - \\bar{predicted})}{\\sqrt{\\sum_{i=1}^{n} (true_i - \\bar{true})^2} \\sqrt{\\sum_{i=1}^{n} (predicted_i - \\bar{predicted})^2}} - 1 \\right)^2 +
            \\left( \\frac{\\frac{\\sigma_{predicted}}{\\bar{predicted}}}{\\frac{\\sigma_{true}}{\\bar{true}}} - 1 \\right)^2 +
            \\left( \\frac{\\bar{predicted}}{\\bar{true}} - 1 \\right)^2}


        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.kge_mod()

        """
        return kge_mod(true=self.true, predicted=self.predicted, treat_arrays=False)

    def kge_np(self):
        """
        Non-parametric Kling-Gupta Efficiency after `Pool et al. 2018 <https://doi.org/10.1080/02626667.2018.1552002>`_.

        .. math::
            cc = \\rho(\\text{true}, \\text{predicted})

        .. math::
            \\alpha = 1 - 0.5 \\sum_{i=1}^{n} \\left| \\frac{\\text{sorted(predicted}_i\\text{)}}{\\text{mean(predicted)} \\cdot n} - \\frac{\\text{sorted(true}_i\\text{)}}{\\text{mean(true)} \\cdot n} \\right|

        .. math::
            \\beta = \\frac{\\text{mean(predicted)}}{\\text{mean(true)}}

        .. math::
            \\text{KGE}_{\\text{np}} = 1 - \\sqrt{(cc - 1)^2 + (\\alpha - 1)^2 + (\\beta - 1)^2}


        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.kge_np()

        """
        return kge_np(true=self.true, predicted=self.predicted, treat_arrays=False)

    def kgeprime_bound(self) -> float:
        """
        `Bounded Version of the Modified Kling-Gupta Efficiency <https://iahs.info/uploads/dms/13614.21--211-219-41-MATHEVET.pdf>`_

        .. math::
            KGE'_{\\text{bounded}} = \\frac{1 - \\sqrt{(r - 1)^2 + (\\gamma - 1)^2 + (\\beta - 1)^2}}{2 - (1 - \\sqrt{(r - 1)^2 + (\\gamma - 1)^2 + (\\beta - 1)^2})}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.kgeprime_bound()

        """
        return kgeprime_bound(true=self.true, predicted=self.predicted, treat_arrays=False)

    def kgenp_bound(self):
        """
        Bounded Version of the Non-Parametric Kling-Gupta Efficiency

        .. math::
            KGE_{np_{bound}} = \\frac{1 - \\sqrt{\\left(\\rho(t, p) - 1\\right)^2 + \\left(1 - 0.5 \\sum_{i=1}^{n} \\left| \\frac{\\text{sorted}(p_i)}{\\text{mean}(p) \\cdot n} - \\frac{\\text{sorted}(t_i)}{\\text{mean}(t) \\cdot n} \\right| - 1\\right)^2 + \\left(\\frac{\\text{mean}(p)}{\\text{mean}(t)} - 1\\right)^2}}{2 - \\left(1 - \\sqrt{\\left(\\rho(t, p) - 1\\right)^2 + \\left(1 - 0.5 \\sum_{i=1}^{n} \\left| \\frac{\\text{sorted}(p_i)}{\\text{mean}(p) \\cdot n} - \\frac{\\text{sorted}(t_i)}{\\text{mean}(t) \\cdot n} \\right| - 1\\right)^2 + \\left(\\frac{\\text{mean}(p)}{\\text{mean}(t)} - 1\\right)^2}\\right)}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.kgenp_bound()

        """
        return kgenp_bound(true=self.true, predicted=self.predicted, treat_arrays=False)

    def kl_sym(self) -> Union[float, None]:
        """
        `Symmetric kullback-leibler divergence <https://doi.org/10.1016/j.procs.2018.10.144>`_

        .. math::
            \\text{KL}_{\\text{sym}}(P || Q) = \\frac{1}{2} \\sum_{i=1}^{n} \\left( P_i - Q_i \\right) \\left( \\log_2 \\frac{P_i}{Q_i} \\right)

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.kl_sym()
        """
        return kl_sym(true=self.true, predicted=self.predicted, treat_arrays=False)

    def lm_index(self, obs_bar_p=None) -> float:
        """
        `Legate-McCabe Efficiency Index <https://doi.org/10.1016/j.cmpb.2023.107737>`_.
        Less sensitive to outliers in the data. The larger, the better

        .. math::
            a_i = |predicted_i - true_i|

        .. math::
            b_i = |true_i - \\text{obs\\_bar\\_p}| \\text{if } \\text{obs\\_bar\\_p} \\text{ is provided} \\|true_i - \\bar{true}| \\text{otherwise}

        .. math::
            \\text{LM Index} = 1 - \\frac{\\sum_{i=1}^{n} a_i}{\\sum_{i=1}^{n} b_i}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.lm_index()
        """
        return lm_index(true=self.true, predicted=self.predicted, treat_arrays=False,
                        obs_bar_p=obs_bar_p)

    def maape(self) -> float:
        """
        `Mean Arctangent Absolute Percentage Error <https://doi.org/10.1016/j.ijforecast.2015.12.003>`_
        Note: result is NOT multiplied by 100

        .. math::
            MAAPE = \\frac{1}{n} \\sum_{i=1}^{n} \\arctan \\left( \\frac{| \\text{true}_i - \\text{predicted}_i |}{| \\text{true}_i | + \\epsilon} \\right)

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.maape()
        """
        return maape(true=self.true, predicted=self.predicted, treat_arrays=False)

    def mae(self) -> float:
        """ `Mean Absolute Error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html>`_.
        It is less sensitive to outliers as compared to mse/rmse.

        .. math::
            \\text{MAE} = \\frac{1}{n} \\sum_{i=1}^{n} \\left| \\text{true}_i - \\text{predicted}_i \\right|

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.mae()
        """
        return mae(true=self.true, predicted=self.predicted, treat_arrays=False)

    def mape(self) -> float:
        """ Mean Absolute Percentage Error.
        The MAPE is often used when the quantity to predict is known to remain
        way above zero_. It is useful when the size or size of a prediction variable
        is significant in evaluating the accuracy of a prediction_. It has advantages
        of scale-independency and interpretability. However, it has the significant
        disadvantage that it produces infinite or undefined values for zero or
        close-to-zero actual values_.

        .. math::
            MAPE = \\frac{1}{n} \\sum_{i=1}^{n} \\left| \\frac{true_i - predicted_i}{true_i} \\right| \\times 100

        .. _zero:
            https://doi.org/10.1016/j.neucom.2015.12.114

        .. _prediction:
            https://doi.org/10.1088/1742-6596/930/1/012002

        .. _values:
            https://doi.org/10.1016/j.ijforecast.2015.12.003

        References
        ---------
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.mape()
        """
        return mape(true=self.true, predicted=self.predicted, treat_arrays=False)

    def mbrae(self, benchmark: np.ndarray = None) -> float:
        """ `Mean Bounded Relative Absolute Error <https://doi.org/10.1371/journal.pone.0174202>`_

        .. math::
            MBRAE = \\frac{1}{n} \\sum_{i=1}^{n} \\frac{| \\text{true}_i - \\text{predicted}_i |}{| \\text{true}_i - \\text{benchmark}_i |}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.mbrae()
        """
        return mbrae(true=self.true, predicted=self.predicted, benchmark=benchmark,
                     treat_arrays=False)

    def mapd(self) -> float: #ToDo equation not multiplied by 100
        """ `Mean absolute percentage deviation <https://doi.org/10.1016/j.rinma.2022.100347>`_

        .. math::
            MAPD = \\frac{\\sum_{i=1}^{n} \\left| predicted_i - true_i \\right|}{\\sum_{i=1}^{n} \\left| true_i \\right|}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.mapd(t, p)
        """
        return mapd(true=self.true, predicted=self.predicted, treat_arrays=False)

    def mase(self, seasonality: int = 1):
        """
        Mean Absolute Scaled Error. Baseline (benchmark) is computed with naive
        forecasting (shifted by @seasonality) modified after `this <https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9>`_. It is the
        ratio of MAE of used model and MAE of naive forecast.

        .. math::
            \\text{MASE} = \\frac{\\frac{1}{n} \\sum_{i=1}^{n} \\left| \\text{true}_i - \\text{predicted}_i \\right|}{\\frac{1}{n-s} \\sum_{i=s+1}^{n} \\left| \\text{true}_i - \\text{true}_{i-s} \\right|}

        Hyndman, R. J. (2006). Another look at forecast-accuracy metrics for intermittent demand.
        Foresight: The International Journal of Applied Forecasting, 4(4), 43-46.

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.mase()
        """
        return mase(true=self.true, treat_arrays=False, predicted=self.predicted, seasonality= seasonality)

    def mare(self) -> float:
        """ Mean Absolute Relative Error. When expressed in %age, it is also known as mape_.

        .. math::
            \\text{MARE} = \\frac{1}{n} \\sum_{i=1}^{n} \\left| \\frac{\\text{true}_i - \\text{predicted}_i}{\\text{true}_i} \\right|

        .. _mape:
            https://doi.org/10.1016/j.rser.2015.08.035

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.mare()
        """
        return mare(true=self.true, predicted=self.predicted,
                    treat_arrays=False)

    def max_error(self) -> float:
        """
        `maximum absolute error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.max_error.html>`_
        In Sklearn, there is "absolute" in equation but not in name of metric.

        .. math::
            \\text{Max Error} = \\max_{i=1}^n \\left| \\text{true}_i - \\text{predicted}_i \\right|

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.max_error()
        """
        return max_error(true=self.true, predicted=self.predicted, treat_arrays=False)

    def mb_r(self) -> float:
        """
        `Mielke-Berry R value <https://link.springer.com/book/10.1007/978-1-4757-3449-2>`_.
        Berry and Mielke, 1988.

        .. math::
            R = 1 - \\frac{n^2 \\cdot \\frac{1}{n} \\sum_{i=1}^{n} \\left| \\text{predicted}_i - \\text{true}_i \\right|}{\\sum_{i=1}^{n} \\sum_{j=1}^{n} \\left| \\text{predicted}_j - \\text{true}_i \\right|}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.mb_r()
        """
        return mb_r(true=self.true, predicted=self.predicted, treat_arrays=False)

    def mda(self) -> float:
        """ Mean Directional Accuracy
        modified `after <https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9>`_

        .. math::
            \\text{MDA} = \\frac{1}{n-1} \\sum_{i=1}^{n-1} \\left( \\text{sign}( \\text{true}_{i+1} - \\text{true}_i) == \\text{sign}( \\text{predicted}_{i+1} - \\text{predicted}_i) \\right)


        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.mda()
         """
        return mda(true=self.true, predicted=self.predicted, treat_arrays=False)

    def mde(self) -> float:
        """
        `Median Error <https://doi.org/10.1016/j.cma.2024.116842>`_

        .. math::
            MDE = \\text{median}(\\text{predicted}_i - \\text{true}_i)

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.mde()
        """
        return mde(true=self.true, predicted=self.predicted, treat_arrays=False)

    def mdape(self) -> float:
        """
        `Median Absolute Percentage Error <https://doi.org/10.1016/j.petrol.2021.109265>`_. The value is multiplied by 100.

        .. math::
            \\text{MdAPE} = 100 \\times \\text{Median} \\left( \\left\\{ \\frac{|\\text{true}_i - \\text{predicted}_i|}{|\\text{true}_i|} \\right\\}_{i=1}^n \\right)

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.mdape()
        """
        return mdape(true=self.true, predicted=self.predicted, treat_arrays=False)

    def mdrae(self, benchmark: np.ndarray = None) -> float:
        """ `Median Relative Absolute Error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html>`_
        In Sklearn, there is "absolute" in equation but not in name of metric.

        .. math::
            MdRAE = \\text{median} \\left( \\left| \\frac{true_i - predicted_i}{true_i - benchmark_i} \\right| \\right)

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.mdrae()
        """
        return mdrae(true=self.true, predicted=self.predicted, treat_arrays=False,
                     benchmark=benchmark)

    def me(self):
        """ `Mean error <https://doi.org/10.1016/j.scitotenv.2024.174533>`_

        .. math::
            ME = \\frac{1}{n} \\sum_{i=1}^{n} (\\text{true}_i - \\text{predicted}_i)

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.me()
        """
        return me(true=self.true, predicted=self.predicted, treat_arrays=False)

    def mean_bias_error(self) -> float:
        """
        Mean Bias Error
        It represents overall bias error or systematic error. It shows average interpolation bias; i.e. average over-
        or underestimation. [1][2].This indicator expresses a tendency of model to underestimate (negative value)
        or overestimate (positive value) global radiation, while the MBE values closest to zero are desirable.
        The drawback of this test is that it does not show the correct performance when the model presents
        overestimated and underestimated values at the same time, since overestimation and underestimation
        values cancel each other.

        .. math::
            \\text{MBE} = \\frac{1}{N} \\sum_{i=1}^{N} (true_i - predicted_i)

        References
        ----------

        - `Willmott, C. J., & Matsuura, K. (2006). On the use of dimensioned measures of error to evaluate the performance
            of spatial interpolators. International Journal of Geographical Information Science, 20(1), 89-102.
            <https://doi.org/10.1080/1365881050028697>`_

        - `Valipour, M. (2015). Retracted: Comparative Evaluation of Radiation-Based Methods for Estimation of Potential
            Evapotranspiration. Journal of Hydrologic Engineering, 20(5), 04014068.
            <https://dx.doi.org/10.1061/(ASCE)HE.1943-5584.0001066>`_

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.mean_bias_error()
         """
        return mean_bias_error(true=self.true, predicted=self.predicted, treat_arrays=False)

    def mean_var(self) -> float:
        """Mean variance

        .. math::
            \\text{mean_var} = \\text{Var} \\left( \\log(1 + \\text{true}) - \\log(1 + \\text{predicted}) \\right)

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.mean_var()
        """
        return mean_var(true=self.true, predicted=self.predicted, treat_arrays=False)

    def mean_poisson_deviance(self, weights=None) -> float:
        """
        `mean poisson deviance <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_poisson_deviance.html>`_

        .. math::
            \\text{MPD} = \\frac{1}{n} \\sum_{i=1}^{n} 2 \\left( \\text{true}_i \\log \\left( \\frac{\\text{true}_i}{\\text{predicted}_i} \\right) - (\\text{true}_i - \\text{predicted}_i) \\right)

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.mean_poisson_deviance()
        """
        return mean_poisson_deviance(true=self.true, predicted=self.predicted,
                                     weights=weights, treat_arrays=False)

    def mean_gamma_deviance(self, weights=None) -> float:
        """
        `mean gamma deviance <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_gamma_deviance.html>`_

        .. math::
            \\text{Mean Gamma Deviance (Weighted)} = \\frac{1}{\\sum_{i=1}^{n} w_i} \\sum_{i=1}^{n} w_i \\frac{2}{\\text{true}_i} \\left( \\text{predicted}_i - \\text{true}_i - \\text{true}_i \\ln \\left( \\frac{\\text{predicted}_i}{\\text{true}_i} \\right) \\right)

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.mean_gamma_deviance()
        """

        return mean_gamma_deviance(true=self.true, predicted=self.predicted, weights=weights,
                                   treat_arrays=False)

    def median_abs_error(self) -> float:
        """
        median absolute error

        .. math::
            \\text{MedAE} = \\text{median} \\left( \\left| \\text{true}_i - \\text{predicted}_i \\right| \\right)

        References
        ----------
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.median_abs_error()
        """
        return median_abs_error(true=self.true, predicted=self.predicted, treat_arrays=False)

    def med_seq_error(self) -> float:
        """`Median Squared Error <https://www.sciencedirect.com/science/article/pii/S2468227620301757>`_
        Same as mse, but it takes median which reduces the impact of outliers.

        .. math::
            \\text{MedSE} = \\text{median} \\left( (\\text{predicted}_i - \\text{true}_i)^2 \\right)

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics = RegressionMetrics(t, p)
        >>> metrics.med_seq_error()
        """
        return med_seq_error(true=self.true, predicted=self.predicted, treat_arrays=False)

    def mle(self) -> float:
        """ `Mean log error <https://doi.org/10.1038/s41598-023-29871-8>`_

        .. math::
            \\text{MLE} = \\frac{1}{n} \\sum_{i=1}^{n} \\left( \\log(1 + \\text{predicted}_i) - \\log(1 + \\text{true}_i) \\right)

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics = RegressionMetrics(t, p)
        >>> metrics.mle()
        """
        return mle(true=self.true, predicted=self.predicted, treat_arrays=False)

    def mod_agreement_index(self, j=1) -> float:
        """Modified agreement of index.
        j: int, when j==1, this is same as agreement_index. Higher j means more impact of outliers.

        .. math::
            MAI = 1 - \\frac{\\sum_{i=1}^{n} \\left| \\text{predicted}_i - \\text{true}_i \\right|^j}{\\sum_{i=1}^{n} \\left( \\left| \\text{predicted}_i - \\overline{\\text{true}} \\right| + \\left| \\text{true}_i - \\overline{\\text{true}} \\right| \\right)^j}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics = RegressionMetrics(t, p)
        >>> metrics.mod_agreement_index()
        """
        return mod_agreement_index(true=self.true, predicted=self.predicted,
                                   treat_arrays=False, j=j)

    def mpe(self) -> float:
        """ `Mean Percentage Error <https://doi.org/10.1016/j.molliq.2023.123378>`_
        The value is multiplied by 100 to reflect percentage.

        .. math::
            MPE = \\frac{1}{n} \\sum_{i=1}^{n} \\left( \\frac{true_i - predicted_i}{true_i} \\right) \\times 100

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.mpe()
        """
        return mpe(true=self.true, predicted=self.predicted, treat_arrays=False)

    def mrae(self, benchmark: np.ndarray = None):
        """ `Mean Relative Absolute Error <https://doi.org/10.1016/j.comnet.2024.110237>`_

        .. math::
            MRAE = \\frac{1}{n} \\sum_{i=1}^{n} \\left| \\frac{\\text{true}_i - \\text{predicted}_i}{\\text{benchmark}_i} \\right|

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.mrae()
        """

        return mrae(true=self.true, predicted=self.predicted, treat_arrays=False,
                    benchmark=benchmark)

    def msle(self, weights=None) -> float:
        """
        `Mean square logrithmic error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_log_error.html>`_

        .. math::
            \\text{MSLE} = \\frac{\\sum_{i=1}^{n} w_i \\cdot \\text{sq_log_error}_i}{\\sum_{i=1}^{n} w_i}
        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.msle()
        """

        return msle(true=self.true, predicted=self.predicted, treat_arrays=False,
                    weights=weights)

    def mre(self, benchmark:np.ndarray=None):
        """
        `mean relative error <https://doi.org/10.1016/j.trd.2022.103505>`_

        .. math::
            \\text{MRE} = \\frac{1}{n} \\sum_{i=1}^{n} \\left| \\frac{\\text{true}_i - \\text{predicted}_i}{\\text{true}_i} \\right|

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.mre()

        """
        return mre(self.true, self.predicted, treat_arrays=False, benchmark=benchmark)

    def norm_euclid_distance(self) -> float:
        """ `Normalized Euclidian distance <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html>`_

        .. math::
            D_{norm} = \\sqrt{\\sum_{i=1}^{n} \\left( \\frac{\\text{true}_i}{\\bar{\\text{true}}} - \\frac{\\text{predicted}_i}{\\bar{\\text{predicted}}} \\right)^2}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.norm_euclid_distance()
        """
        return norm_euclid_distance(true=self.true, predicted=self.predicted,
                                    treat_arrays=False)

    def nrmse_range(self) -> float:
        """Range Normalized Root Mean Squared Error.
        RMSE normalized by true values. This allows comparison between data sets
        with different scales. It is more sensitive to outliers.

        Reference: `Pontius et al., 2008 <https://link.springer.com/article/10.1007/s10651-007-0043-y>`_

        .. math::
            \\text{NRMSE} = \\frac{\\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (\\text{predicted}_i - \\text{true}_i)^2}}{\\max(\\text{true}) - \\min(\\text{true})}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.nrmse_range()
        """
        return nrmse_range(true=self.true, treat_arrays=False, predicted=self.predicted)

    def nrmse_ipercentile(self, q1=25, q2=75) -> float:
        """
        RMSE normalized by inter percentile range of true. This is the least sensitive to outliers.
        q1: any interger between 1 and 99
        q2: any integer between 2 and 100. Should be greater than q1.
        Reference: `Pontius et al., 2008 <https://link.springer.com/article/10.1007/s10651-007-0043-y>`_

        .. math::
            \\text{NRMSE}_{\\text{IP}} = \\frac{\\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (\\text{true}_i - \\text{predicted}_i)^2}}{Q_{q2} - Q_{q1}}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.nrmse_ipercentile()
        """
        return nrmse_ipercentile(true=self.true, treat_arrays=False, q1=q1, q2=q2, predicted=self.predicted)

    def nrmse_mean(self) -> float:
        """Mean Normalized RMSE
        RMSE normalized by mean of true values.This allows comparison between datasets with different scales.

        Reference: `Pontius et al., 2008 <https://link.springer.com/article/10.1007/s10651-007-0043-y>`_

        .. math::
            NRMSE_{mean} = \\frac{\\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (\\text{true}_i - \\text{predicted}_i)^2}}{\\bar{\\text{true}}}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.nrmse_mean()
        """
        return nrmse_mean(true=self.true, predicted=self.predicted, treat_arrays=False)

    def norm_ae(self) -> float:
        """ `Normalized Absolute Error <https://doi.org/10.1016/j.apor.2024.104042>`_

        .. math::
            norm\\_ae = \\sqrt{\\frac{\\sum_{i=1}^{n} (error_i - MAE)^2}{n - 1}}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.norm_ae()
        """
        return norm_ae(true=self.true, predicted=self.predicted, treat_arrays=False)

    def norm_ape(self) -> float:
        """ Normalized Absolute Percentage Error

        .. math::
            \\text{norm_APE} = \\sqrt{ \\frac{1}{n-1} \\sum_{i=1}^{n} \\left( \\left| \\frac{\\text{true}_i - \\text{predicted}_i}{\\text{true}_i} \\right| - \\frac{1}{n} \\sum_{j=1}^{n} \\left| \\frac{\\text{true}_j - \\text{predicted}_j}{\\text{true}_j} \\right| \\right)^2 }


        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.norm_ape()
        """
        return norm_ape(true=self.true, predicted=self.predicted, treat_arrays=False)

    def nrmse(self) -> float:
        """ `Normalized Root Mean Squared Error <https://www.sciencedirect.com/science/article/pii/S0957417411003289>`_

        .. math::
            NRMSE = \\frac{\\sqrt{\\frac{1}{N} \\sum_{i=1}^{N} (\\text{true}_i - \\text{predicted}_i)^2}}{\\max(\\text{true}) - \\min(\text{true})}

            Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.nrmse()
        """
        return nrmse(true=self.true, predicted=self.predicted, treat_arrays=False)

    def nse(self) -> float:
        """Nash-Sutcliff Efficiency.

        The Nash-Sutcliffe efficiency (NSE) is a normalized statistic that determines
        the relative magnitude of the residual variance compared to the measured data
        variance It determines how well the model simulates trends for the output response
        of concern. But cannot help identify model bias and cannot be used to identify
        differences in timing and magnitude of peak flows and shape of recession curves;
        in other words, it cannot be used for single-event simulations. It is sensitive
        to extreme values due to the squared differ-ences `[1] <https://elibrary.asabe.org/abstract.asp?aid=46548>`_. To make it less sensitive
        to outliers, `[2] <https://dx.doi.org/10.5194/adgeo-5-89-2005>`_ proposed log and relative nse.

        .. math::
            \\text{NSE} = 1 - \\frac{\\sum_{i=1}^{N} (predicted_i - true_i)^2}{\\sum_{i=1}^{N} (true_i - \\bar{true})^2}

        References
        ----------
        - Moriasi, D. N., Gitau, M. W., Pai, N., & Daggupati, P. (2015). Hydrologic and water quality models:
            Performance measures and evaluation criteria. Transactions of the ASABE, 58(6), 1763-1785.
        - Krause, P., Boyle, D., & Bäse, F. (2005). Comparison of different efficiency criteria for hydrological
            model assessment. Adv. Geosci., 5, 89-97. https://dx.doi.org/10.5194/adgeo-5-89-2005.

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.nse()
        """
        return nse(true=self.true, predicted=self.predicted, treat_arrays=False)

    def nse_alpha(self) -> float:
        """

        Alpha decomposition of the NSE, see `Gupta et al., 2009 <https://doi.org/10.1029/97WR03495>`_
        used in `Kratzert et al., 2019 <https://hess.copernicus.org/articles/23/5089/2019/hess-23-5089-2019.html>`_.

        .. math::
            \\text{NSE}_{\\text{alpha}} = \\frac{\\sigma_{\\text{predicted}}}{\\sigma_{\\text{true}}}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.nse_alpha()
        """
        return nse_alpha(true=self.true, predicted=self.predicted, treat_arrays=False)

    def nse_beta(self) -> float:
        """
        Beta decomposition of NSE. `Gupta et al. 2009 <https://doi.org/10.1029/97WR03495>`_
        used in `kratzert et al., 2019 <https://hess.copernicus.org/articles/23/5089/2019/hess-23-5089-2019.html>`_.
        .. math::
            \\text{NSE}_{\\text{beta}} = \\frac{\\mu_{\\text{predicted}} - \\mu_{\\text{true}}}{\\sigma_{\\text{true}}}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.nse_beta()

        """
        return nse_beta(true=self.true, predicted=self.predicted, treat_arrays=False)

    def nse_mod(self, j=1) -> float:
        """
        Gives less weightage to outliers if j=1 and if j>1 then it gives more
        weightage to outliers. Reference: `Krause_ et al., 2005 <https://adgeo.copernicus.org/articles/5/89/2005/adgeo-5-89-2005.html>`_.

        .. math::
            \\text{NSE}_{\\text{mod}} = 1 - \\frac{\\sum_{i=1}^{N} \\left| \\text{predicted}_i - \\text{true}_i \\right|^j}{\\sum_{i=1}^{N} \\left| \\text{true}_i - \\bar{\text{true}} \\right|^j}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.nse_mod()

        """
        return nse_mod(true=self.true, predicted=self.predicted, treat_arrays=False, j=j)

    def nse_rel(self) -> float:
        """
        `Relative Nash-Sutcliff Efficiency <https://doi.org/10.5194/adgeo-5-89-2005>`_.

        .. math::
            \\text{NSE}_{\\text{rel}} = 1 - \\frac{\\sum_{i=1}^{N} \\left( \\frac{|\\text{predicted}_i - \\text{true}_i|}{\\text{true}_i} \\right)^2}{\\sum_{i=1}^{N} \\left( \\frac{|\\text{true}_i - \\overline{\\text{true}}|}{\\overline{\\text{true}}} \\right)^2}


        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.nse_rel()
        """
        return nse_rel(true=self.true, predicted=self.predicted, treat_arrays=False)

    def nse_bound(self) -> float:
        """
        Bounded Version of the Nash-Sutcliffe Efficiency (nse_)

        .. math::
            \\text{NSE}_{\\text{bound}} = \\frac{\\text{NSE}}{2 - \\text{NSE}}

        .. _nse:
            https://iahs.info/uploads/dms/13614.21--211-219-41-MATHEVET.pdf

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.nse_bound()
        """
        return nse_bound(true=self.true, predicted=self.predicted, treat_arrays=False)

    # def log_nse(self, epsilon=0.0) -> float:
    #     """
    #     log Nash-Sutcliffe model efficiency
    #
    #     .. math::
    #         NSE = 1-\\frac{\\sum_{i=1}^{N}(log(e_{i})-log(s_{i}))^2}{\\sum_{i=1}^{N}(log(e_{i})-log(\\bar{e})^2}-1)*-1
    #
    #         Examples
    #     ---------
    #     >>> import numpy as np
    #     >>> from SeqMetrics import RegressionMetrics
    #     >>> t = np.random.random(10)
    #     >>> p = np.random.random(10)
    #     >>> metrics= RegressionMetrics(t, p)
    #     >>> metrics.log_nse()
    #
    #     """
    #     return log_nse(true=self.true, predicted=self.predicted, epsilon=epsilon, treat_arrays=False)

    def log_prob(self) -> float:
        """
        Logarithmic probability distribution

        .. math::
            \\text{log_prob} = \\frac{1}{N} \\sum_{i=1}^{N} \\left( -\\frac{\\left( \\frac{\\text{true}_i - \\text{predicted}_i}{\\text{scale}} \\right)^2}{2} - \\log(\\sqrt{2\\pi}) \\right)

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.log_prob()
        """
        return log_prob(true=self.true, predicted=self.predicted, treat_arrays=False)

    def pbias(self) -> float:
        """
        Percent Bias.
        It determines how well the model simulates the average magnitudes for the
        output response of interest. It can also determine over and under-prediction.
        It cannot be used (1) for single-event simula-tions to identify differences
        in timing and magnitude of peak flows and the shape of recession curves nor (2)
        to determine how well the model simulates residual variations and/or trends
        for the output response of interest. It can  give a deceiving rating of
        model performance if the model overpredicts as much as it underpredicts,
        in which case PBIAS will be close to zero even though the model simulation
        is poor. `[1] <https://elibrary.asabe.org/abstract.asp?aid=46548>`_

        .. math::
            PBIAS = 100 \\times \\frac{\\sum_{i=1}^{N} (\\text{true}_i - \\text{predicted}_i)}{\\sum_{i=1}^{N} \\text{true}_i}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.pbias()
        """

        return pbias(true=self.true, predicted=self.predicted, treat_arrays=False)

    def rmsle(self) -> float:
        """`Root mean square log error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_log_error.html>`_.

        This error is less sensitive to `outliers <https://stats.stackexchange.com/q/56658/314919>`_ .
        Compared to RMSE, RMSLE only considers the relative error between predicted
        and actual values, and the scale of the error is nullified by the log-transformation.
        Furthermore, RMSLE penalizes underestimation more than overestimation.
        This is especially useful in those studies where the underestimation
        of the target variable is not acceptable but overestimation can be
        `tolerated <https://doi.org/10.1016/j.scitotenv.2020.137894>`_ .

        .. math::
            RMSLE = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} \\left( \\log(1 + \\text{predicted}_i) - \\log(1 + \\text{true}_i) \\right)^2}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.rmsle()

         """
        return rmsle(true=self.true, predicted=self.predicted, treat_arrays=False)

    def rmdspe(self) -> float:
        """
        `Root Median Squared Percentage Error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html>`_.
        The value is multiplied by 100 to reflect percentage.

        .. math::
            \\text{RMDSPE} = \\sqrt{\\text{median}\\left(\\left(\\frac{\\text{true}_i - \\text{predicted}_i}{\\text{true}_i} \\times 100\\right)^2\\right)}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.rmdspe()
        """
        return rmdspe(true=self.true, predicted=self.predicted, treat_arrays=False)

    def rse(self) -> float:
        """Relative Squared Error

        .. math::
            \\text{RSE} = \\frac{\\sum_{i=1}^{n} (\\text{true}_i - \\text{predicted}_i)^2}{\\sum_{i=1}^{n} (\\text{true}_i - \\bar{\\text{true}})^2}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.rse()
        """
        return rse(true=self.true, predicted=self.predicted, treat_arrays=False)

    def rrse(self) -> float:
        """ `Root Relative Squared Error <https://www.sciencedirect.com/science/article/pii/S0360319923031798>`_

        .. math::
            RRSE = \\sqrt{\\frac{\\sum_{i=1}^{n} (\\text{true}_i - \\text{predicted}_i)^2}{\\sum_{i=1}^{n} (\\text{true}_i - \\bar{\\text{true}})^2}}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.rrse()
        """
        return rrse(true=self.true, predicted=self.predicted, treat_arrays=False)

    def rae(self) -> float:
        """ `Relative Absolute Error <https://doi.org/10.1016/j.compbiomed.2017.02.010>`_ (aka Approximation Error)

        .. math::
            \\text{RAE} = \\frac{\\sum_{i=1}^{n} \\left| \\text{true}_i - \\text{predicted}_i \\right|}{\\sum_{i=1}^{n} \\left| \\text{true}_i - \\overline{\\text{true}} \\right|}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.rae()
        """
        return rae(true=self.true, treat_arrays=False, predicted=self.predicted)

    def ref_agreement_index(self) -> float:
        """`Refined Index of Agreement <https://www.researchgate.net/profile/Scott-Robeson/publication/235961403_A_refined_index_of_model_performance/links/5a6114254585158bca49f8e4/A-refined-index-of-model-performance.pdf>`_
        . From -1 to 1. Larger the better.

        .. math::
            a = \\sum_{i=1}^{n} \\left| \\text{predicted}_i - \\text{true}_i \\right|

        .. math::
            b = 2 \\sum_{i=1}^{n} \\left| \\text{true}_i - \\overline{\\text{true}} \\right|

        .. math::
            d_{\\text{ref}} =
            \\begin{cases}
            1 - \\frac{a}{b} & \\text{if } a \\leq b \\
            \\frac{b}{a} - 1 & \\text{if } a > b
            \\end{cases}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.ref_agreement_index()
        """
        return ref_agreement_index(true=self.true, predicted=self.predicted, treat_arrays=False)

    def rel_agreement_index(self) -> float:
        """ `Relative index of agreement <https://doi.org/10.1007/s10661-022-10844-9>`_. from 0 to 1. larger the better.

        .. math::
            \\text{rel_agreement_index} = 1 - \\frac{\\sum_{i=1}^{n} \\left( \\frac{\\text{predicted}_i - \\text{true}_i}{\\text{true}_i} \\right)^2}{\\sum_{i=1}^{n} \\left( \\frac{|\\text{predicted}_i - \\bar{\\text{true}}| + |\\text{true}_i - \\bar{\\text{true}}|}{\\bar{\\text{true}}} \\right)^2}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.rel_agreement_index()
        """
        return rel_agreement_index(true=self.true, predicted=self.predicted, treat_arrays=False)

    def rmse(self, weights=None) -> float:
        """ `Root mean squared error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html>`_

        .. math::
            \\text{RMSE} = \\sqrt{\\frac{\\sum_{i=1}^{n} w_i (\\text{true}_i - \\text{predicted}_i)^2}{\\sum_{i=1}^{n} w_i}}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.rmse()
        """
        return rmse(true=self.true, predicted=self.predicted, treat_arrays=False, weights=weights)

    def r2(self) -> float:
        """
        R2 is a statistical measure of how well the regression line approximates the actual data.
        Quantifies the percent of variation in the response that the 'model'
        explains_. The 'model' here is anything from which we obtained predicted
        array. It is also called coefficient of determination or square of pearson
        correlation coefficient. More heavily affected by outliers than pearson correlatin r.

        .. math ::
            R^2 = \\left( \\frac{\\sum_{i=1}^{N} \\left( \\frac{true_i - \\bar{true}}{\\sigma_{true}} \\cdot \\frac{predicted_i - \\bar{predicted}}{\\sigma_{predicted}} \\right)}{N - 1} \\right)^2

        .. _explains:
            https://data.library.virginia.edu/is-r-squared-useless/

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> r_square= metrics.r2()
        >>> r_square
        """

        return r2(true=self.true, predicted=self.predicted, treat_arrays=False)

    def r2_score(self, weights=None):
        """
        This is not a symmetric function.
        Unlike most other scores, `R^2 score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html>`_
        may be negative (it need not actually
        be the square of a quantity R).
        This metric is not well-defined for single samples and will return a NaN
        value if n_samples is less than two.

        .. math::
            \\text{R2}_{\\text{score}} = 1 - \\frac{\\sum_{i=1}^{n} w_i (\\text{true}_i - \\text{predicted}_i)^2}{\\sum_{i=1}^{n} w_i (\\text{true}_i - \\bar{\\text{true}})^2}


        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.r2_score()

        """
        return r2_score(true=self.true, predicted=self.predicted, treat_arrays=False, weights=weights)

    def relative_rmse(self) -> float:
        """
        Relative Root Mean Squared Error

        .. math::
            RRMSE=\\frac{\\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})^2}}{\\bar{e}}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.relative_rmse()
        """
        return relative_rmse(true=self.true, treat_arrays=False, predicted=self.predicted)

    def rmspe(self) -> float:
        """
        `Root Mean Square Percentage Error <https://stackoverflow.com/a/53166790/5982232>`_ .

        .. math::
            RMSPE = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} \\left(PE_i\\right)^2} = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} \\left(\\frac{\\text{true}_i - \\text{predicted}_i}{\\text{true}_i}\\right)^2}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.rmspe()
        """
        return rmspe(true=self.true, predicted=self.predicted, treat_arrays=False)

    def rsr(self) -> float:
        """
        ratio of the root mean square error to the standard deviation of measured data `(RSR) <https://elibrary.asabe.org/abstract.asp?aid=23153>`_,

        It incorporates the benefits of error index statistics andincludes a
        scaling/normalization factor, so that the resulting statistic and reported
        values can apply to various constitu-ents.

        .. math::
            \\text{RSR} = \\frac{\\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (\\text{true}_i - \\text{predicted}_i)^2}}{\\sqrt{\\frac{1}{n-1} \\sum_{i=1}^{n} (\\text{true}_i - \\bar{\\text{true}})^2}}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.rsr()
        """
        return rsr(true=self.true, predicted=self.predicted, treat_arrays=False)

    def rmsse(self) -> float:
        """ Root Mean Squared Scaled Error

        .. math::
            \\text{RMSSE} = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} \\left( \\frac{\\left| \\text{true}_i - \\text{predicted}_i \\right|}{\\frac{1}{n-s} \\sum_{j=s+1}^{n} \\left| \\text{true}_j - \\text{true}_{j-s} \\right|} \\right)^2}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.rmsse()
        """
        return rmsse(true=self.true, predicted=self.predicted, treat_arrays=False)

    def sa(self) -> float:
        """Spectral angle. From -pi/2 to pi/2. Closer to 0 is better.
        It measures angle between two vectors in hyperspace indicating
        how well the shape of two arrays match instead of their magnitude.
        Reference: Robila and Gershman, 2005.

        .. math::
            SA = \\arccos \\left( \\frac{\\sum_{i=1}^{n} (\\text{true}_i \\cdot \\text{predicted}_i)}{\\sqrt{\\sum_{i=1}^{n} (\\text{true}_i)^2} \\cdot \\sqrt{\\sum_{i=1}^{n} (\\text{predicted}_i)^2}} \\right)

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.sa()
        """
        return sa(true=self.true, predicted=self.predicted, treat_arrays=False)

    def sc(self) -> float:
        """Spectral correlation.
        It varies from -pi/2 to pi/2. Closer to 0 is better.

        .. math::
            sc = \\arccos \\left( \\frac{ \\sum_{i=1}^{n} (t_i - \\bar{t}) \\cdot (p_i - \\bar{p}) }{ \\sqrt{\\sum_{i=1}^{n} (t_i - \\bar{t})^2} \\cdot \\sqrt{\\sum_{i=1}^{n} (p_i - \\bar{p})^2} } \\right)

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.sc()
        """
        return sc(true=self.true, predicted=self.predicted, treat_arrays=False)

    def sga(self) -> float:
        """Spectral gradient angle.
        It varies from -pi/2 to pi/2. Closer to 0 is better.

        .. math::
            \\text{SGA} = \\arccos \\left( \\frac{\\sum_{i=1}^{n-1} \\left( (true_{i+1} - true_i) \\cdot (predicted_{i+1} - predicted_i) \\right)}{\\sqrt{\\sum_{i=1}^{n-1} (true_{i+1} - true_i)^2} \\times \\sqrt{\\sum_{i=1}^{n-1} (predicted_{i+1} - predicted_i)^2}} \\right)

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.sga()
        """
        return sga(true=self.true, predicted=self.predicted, treat_arrays=False)

    def smape(self) -> float:
        """
        `Symmetric Mean Absolute Percentage Error <https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error>`_.
        Adoption from `this <https://stackoverflow.com/a/51440114/5982232>`_.

        .. math::
            SMAPE = \\frac{100}{n} \\sum_{i=1}^{n} \\frac{2 \\left| \\text{predicted}_i - \\text{true}_i \\right|}{\\left| \\text{true}_i \\right| + \\left| \\text{predicted}_i \\right|}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.smape()
        """
        return smape(true=self.true, predicted=self.predicted, treat_arrays=False)

    def smdape(self) -> float:
        """
        Symmetric Median Absolute Percentage Error
        Note: result is NOT multiplied by 100

        .. math::
            \\text{smdape} = \\text{median} \\left( \\frac{2 \\cdot | \\text{predicted} - \\text{true} |}{| \\text{true} | + | \\text{predicted} | + \\epsilon} \\right)

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.smdape()
        """
        return smdape(true=self.true, predicted=self.predicted, treat_arrays=False)

    def sid(self) -> float:
        """Spectral Information Divergence.
        From -pi/2 to pi/2. Closer to 0 is better.

        .. math::
            \\text{SID} = \\left( \\frac{\\text{t}}{\\text{mean(t)}} - \\frac{\\text{p}}{\\text{mean(p)}} \\right) \\cdot \\left( \\log_{10}(\\text{t}) - \\log_{10}(\\text{mean(t)}) - \\log_{10}(\\text{p}) + \\log_{10}(\\text{mean(p)}) \\right)

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.sid()
        """
        return sid(true=self.true, predicted=self.predicted, treat_arrays=False)

    def skill_score_murphy(self) -> float:
        """
        Adopted from `here <https://github.com/PeterRochford/SkillMetrics/blob/278b2f58c7d73566f25f10c9c16a15dc204f5869/skill_metrics/skill_score_murphy.py>`_ .
        Calculate non-dimensional skill score (SS) between two variables using
        definition of Murphy (1988) using the formula:

        .. math::
            SS = 1 - RMSE^2/SDEV^2

        .. math::
            SDEV is the standard deviation of the true values

        .. math::
            SDEV^2 = sum_(n=1)^N [r_n - mean(r)]^2/(N-1)

        where p is the predicted values, r is the reference values, and N is the total number of values in p & r.
        Note that p & r must have the same number of values. A positive skill score can be interpreted as the percentage
        of improvement of the new model forecast in comparison to the reference. On the other hand, a negative skill
        score denotes that the forecast of interest is worse than the referencing forecast. Consequently, a value of
        zero denotes that both forecasts perform equally [MLAir, 2020].

        References
        ---------
            `Allan H. Murphy, 1988: Skill Scores Based on the Mean Square Error
            and Their Relationships to the Correlation Coefficient. Mon. Wea.
            Rev., 116, 2417-2424 <doi: http//dx.doi.org/10.1175/1520-0493(1988)<2417:SSBOTM>2.0.CO;2>`_.

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.skill_score_murphy()
        """
        return skill_score_murphy(true=self.true, predicted=self.predicted, treat_arrays=False)

    def spearmann_corr(self) -> float:
        """ `Separmann correlation coefficient <https://hess.copernicus.org/articles/24/2505/2020/hess-24-2505-2020.pdf>`_.

        This is a nonparametric metric and assesses how well the relationship
        between the true and predicted data can be described using a monotonic
        function.

        .. math::
            r = \\frac{\\sum_{i=1}^{n} \\left( R_{t,i} - \\overline{R_t} \\right) \\left( R_{p,i} - \\overline{R_p} \\right)}{\\sqrt{ \\sum_{i=1}^{n} \\left( R_{t,i} - \\overline{R_t} \\right)^2 \\sum_{i=1}^{n} \\left( R_{p,i} - \\overline{R_p} \\right)^2 }}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.spearmann_corr()
        """
        return spearmann_corr(true=self.true, predicted=self.predicted, treat_arrays=False)

    def sse(self) -> float:
        """
        Sum of squared errors_ (model vs actual). It is measure of how far off
        our model's predictions are from the observed values. A value of 0 indicates
        that all predications are spot on. A non-zero value indicates errors.

        This is also called residual sum of squares (RSS) or sum of squared residuals
        as per tutorialspoint_ .

        .. math::
            \\text{SSE} = \\sum_{i=1}^{n} (true_i - predicted_i)^2

        .. _errors:
            https://dziganto.github.io/data%20science/linear%20regression/machine%20learning/python/Linear-Regression-101-Metrics/

        .. _tutorialspoint:
            https://www.tutorialspoint.com/statistics/residual_sum_of_squares.html

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.sse()
        """
        return sse(true=self.true, predicted=self.predicted, treat_arrays=False)

    def std_ratio(self, **kwargs) -> float:
        """ `Ratio of standard deviations of predictions and trues <https://doi.org/10.1016/j.engfracmech.2024.110057>`_.
        Also known as standard ratio, it varies from 0.0 to infinity while
        1.0 being the perfect value.

        .. math::
            \\text{std_ratio} = \\frac{\\sigma_{\\text{predicted}}}{\\sigma_{\\text{true}}}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.std_ratio()
        """
        return std_ratio(true=self.true, predicted=self.predicted, treat_arrays=False, **kwargs)

    def umbrae(self, benchmark: np.ndarray = None):
        """ `Unscaled Mean Bounded Relative Absolute Error <https://doi.org/10.1016/j.jclepro.2022.135414>`_

        .. math::
            UMBRAE = \\frac{\\frac{1}{n} \\sum_{i=1}^{n} \\frac{|t_i - p_i|}{|t_i - b_i|}}{1 - \\frac{1}{n} \\sum_{i=1}^{n} \\frac{|t_i - p_i|}{|t_i - b_i|}}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.umbrae()
        """
        return umbrae(true=self.true, predicted=self.predicted, treat_arrays=False, benchmark=benchmark)

    def ve(self) -> float:
        """
        `Volumetric efficiency <https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007WR006415>`_. from 0 to 1. Smaller the better.

        .. math::
            VE = 1 - \\frac{\\sum_{i=1}^{n} \\left| \\text{predicted}_i - \\text{true}_i \\right|}{\\sum_{i=1}^{n} \\text{true}_i}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.ve()
        """
        return ve(true=self.true, predicted=self.predicted, treat_arrays=False)

    def volume_error(self) -> float:
        """
        Returns the Volume Error (Ve).
        It is an indicator of the agreement between the averages of the simulated
        and observed runoff (i.e. long-term water balance).
        used in `Reynolds <https://doi.org/10.1016/j.jhydrol.2017.05.012>`_ paper:

        .. math::
            \\text{volume_error}= Sum(self.predicted- true)/sum(self.predicted)

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.volume_error()
        """
        return volume_error(true=self.true, predicted=self.predicted, treat_arrays=False)

    def wape(self) -> float:
        """
        weighted absolute percentage error (wape_)

        It is a variation of mape but more suitable for intermittent and low-volume
        data_.

        .. math::
            \\text{WAPE} = \\frac{\\sum_{i=1}^{n} \\left| \\text{true}_i - \\text{predicted}_i \\right|}{\\sum_{i=1}^{n} \\text{true}_i}

        .. _wape:
            https://mattdyor.wordpress.com/2018/05/23/calculating-wape/

        .. _data:
            https://arxiv.org/pdf/2103.12057v1.pdf

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.wape()


        """
        return wape(true=self.true, predicted=self.predicted, treat_arrays=False)

    def watt_m(self) -> float:
        """`Watterson's M. <https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1097-0088(199604)16:4%3C379::AID-JOC18%3E3.0.CO;2-U>`_

        .. math::
            M = \\frac{2}{\\pi} \\cdot \\arcsin \\left( 1 - \\frac{\\frac{1}{n} \\sum_{i=1}^{n} ( \\text{true}_i - \\text{predicted}_i )^2}{\\sigma_{\\text{true}}^2 + \\sigma_{\\text{predicted}}^2 + (\\mu_{\\text{predicted}} - \\mu_{\\text{true}})^2} \\right)

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.watt_m()
        """
        return watt_m(true=self.true, predicted=self.predicted, treat_arrays=False)

    def wmape(self) -> float:
        """
        `Weighted Mean Absolute Percent Error <https://stackoverflow.com/a/54833202/5982232>`_

        .. math::
            \\text{WMAPE} = \\frac{\\sum_{i=1}^{n} \\left| \\text{true}_i - \\text{predicted}_i \\right|}{\\sum_{i=1}^{n} \\text{true}_i}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.wmape()
        """
        return wmape(true=self.true, predicted=self.predicted, treat_arrays=False)

    def variability_ratio(self) -> float:
        """
        Variability Ratio
        It is the ratio of the variance of the predicted values to the variance of the true values.
        It is used to measure the variability of the predicted values relative to the true values.

        .. math::
            VR = 1 - \\left| \\frac{\\frac{\\sigma_{\\text{predicted}}}{\\mu_{\\text{predicted}}}}{\\frac{\\sigma_{\\text{true}}}{\\mu_{\\text{true}}}} - 1 \\right|

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.variability_ratio()
        """
        return variability_ratio(true=self.true, predicted=self.predicted, treat_arrays=False)

    def concordance_corr_coef(self) -> float:
        """
        `Concordance Correlation Coefficient (CCC) <https://en.wikipedia.org/wiki/Concordance_correlation_coefficient>`_
        taken from this `paper <https://doi.org/10.2307/2532051>`_.

        .. math::
            CCC = \\frac{2 \\rho \\sigma_{true} \\sigma_{predicted}}{\\sigma_{true}^2 + \\sigma_{predicted}^2 + (\\bar{true} - \\bar{predicted})^2}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.concordance_corr_coef()
        """
        return concordance_corr_coef(true= self.true, predicted= self.predicted, treat_arrays=False)

    def critical_success_index(self, threshold= 0.5) -> float:
        """
        `Critical Success Index (CSI) <https://doi.org/10.1016/j.heliyon.2024.e26371>`_

        .. math::
            CSI = \\frac{TP}{TP + FN + FP}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.array([0, 1, 1, 0, 0, 1])
        >>> p = np.array([0, 1, 0, 1, 1, 1])
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.critical_success_index()
        """
        return critical_success_index(true= self.true, predicted= self.predicted, treat_arrays=False, threshold= threshold)


    def kl_divergence(self) -> float:
        """
        `Kullback-Leibler Divergence <https://doi.org/10.1016/j.imu.2024.101510>`_

        .. math::
            D_{KL}(P||Q) = \sum_{x\in\mathcal{X}} P(x) \log\frac{P(x)}{Q{x}}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        >>> p = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        >>> metrics= RegressionMetrics(t, p)
        >>> divergence = metrics.kl_divergence()
        """
        return kl_divergence(true= self.true, predicted= self.predicted, treat_arrays=False)

    def log_cosh_error(self) -> float:
        """
        `Log-Cosh Error <https://doi.org/10.1016/j.compchemeng.2022.107933>`_

        .. math::
            \\text{Log-Cosh Error} = \\frac{1}{n} \\sum_{i=1}^{n} \\log \\left( \\cosh(\\text{predicted}_i - \\text{true}_i) \\right)

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.array([1, 2, 3, 4, 5])
        >>> p = np.array([1.1, 1.9, 3.1, 4.2, 4.8])
        >>> metrics= RegressionMetrics(t, p)
        >>> error = metrics.log_cosh_error()
        """
        return log_cosh_error(true= self.true, predicted= self.predicted, treat_arrays=False)

    def minkowski_distance(self, order= 1) -> float:
        """
        `Minkowski Distance <https://doi.org/10.1016/j.imu.2024.101492>`_

        .. math::
            D_{Minkowski} = \\left( \\sum_{i=1}^{n} \\left| \\text{true}_i - \\text{predicted}_i \\right|^p \\right)^{\\frac{1}{p}}


        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.array([1, 2, 3, 4, 5])
        >>> p = np.array([1.1, 1.9, 3.1, 4.2, 4.8])
        >>> metrics= RegressionMetrics(t, p)
        >>> distance = metrics.minkowski_distance()
        """
        return minkowski_distance(true= self.true, predicted= self.predicted, treat_arrays=False, order= order)

    def tweedie_deviance_score(self, power=0) -> float:
        """
        `Tweedie Deviance Score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_tweedie_deviance.html>`_

        .. math::
            D(\\text{true}, \\text{predicted}) = \\frac{1}{n} \\sum_{i=1}^{n} (\\text{true}_i - \\text{predicted}_i)^2

        .. math::
            D(\\text{true}, \\text{predicted}) = 2 \\sum_{i=1}^{n} \\left( \\text{true}_i \\log\\left(\\frac{\\text{true}_i + (\\text{true}_i = 0)}{\\text{predicted}_i}\\right) - \\text{true}_i + \\text{predicted}_i \\right)

        .. math::
            D(\\text{true}, \\text{predicted}) = 2 \\sum_{i=1}^{n} \\left( \\frac{\\text{true}_i}{\\text{predicted}_i} - \\log\\left(\\frac{\\text{true}_i}{\\text{predicted}_i}\\right) - 1 \\right)

        .. math::
            D(\\text{true}, \\text{predicted}) = 2 \\sum_{i=1}^{n} \\left( \\frac{(\\text{true}_i - \\text{predicted}_i)^2}{\\text{true}_i^2 \\text{predicted}_i} \\right)

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.array([1, 2, 3, 4, 5])
        >>> p = np.array([1.1, 1.9, 3.1, 4.2, 4.8])
        >>> metrics= RegressionMetrics(t, p)
        >>> score = metrics.tweedie_deviance_score()
        """
        return tweedie_deviance_score(true= self.true, predicted= self.predicted, treat_arrays=False, power= power)

    # def peak_flow_ratio(self, power=0) -> float:
    #     """
    #     Peak flow ratio is defined the ratio of the highest simulated to the highest
    #     observed flow rates (Broekhuizen et al., 2020).
    #
    #     https://doi.org/10.5194/hess-24-869-2020
    #
    #     Examples
    #     ---------
    #     >>> import numpy as np
    #     >>> from SeqMetrics import RegressionMetrics
    #     >>> t = np.array([1, 2, 3, 4, 5])
    #     >>> p = np.array([1.1, 1.9, 3.1, 4.2, 4.8])
    #     >>> metrics= RegressionMetrics(t, p)
    #     >>> score = metrics.peak_flow_ratio()
    #     """
    #     return peak_flow_ratio(true= self.true, predicted= self.predicted, treat_arrays=False, power= power)

    def legates_coeff_eff(self, power=0) -> float:
        """
        Legates Coefficient of Efficiency. Its value varies between 0 and 1.
        It is not as sensitive to extreme values as agreement_index and coefficcient of
        determination because of the utilization of the absolute value of the difference
        instead of the squared difference. See Equaltion 23 in `Dodo et al., 2022 <https://doi.org/10.1016/j.nexus.2022.100157>`_

        .. math::
            LCE = 1 - \\frac{\\sum_{i=1}^{n} |true_i - predicted_i|}{\\sum_{i=1}^{n} |true_i - \\bar{true}|}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.array([1, 2, 3, 4, 5])
        >>> p = np.array([1.1, 1.9, 3.1, 4.2, 4.8])
        >>> metrics= RegressionMetrics(t, p)
        >>> score = metrics.legates_coeff_eff()
        """
        return legates_coeff_eff(true= self.true, predicted= self.predicted, treat_arrays=False, power= power)

    # def relative_error(self, power=0) -> float:
    #     """
    #     Relative Error. It indicates the mismatch that
    #     occurs between the observed and modeled values, expressed
    #     in terms of percentages.
    #     It quantifies the relative deviations between observed/true
    #     and predicted values. This significantly reduces the influence of absolute
    #     differences at peaks. The absolute lower differences during low flow
    #     periods are enhanced because they are significant if looked at in a
    #     relative sense. As a result, there might be a systematic over- or underprediction during low flow periods.
    #     It used along with other statistics to quantify low
    #     flow simulations Moriasi et al., 2007.
    #
    #     Examples
    #     ---------
    #     >>> import numpy as np
    #     >>> from SeqMetrics import RegressionMetrics
    #     >>> t = np.array([1, 2, 3, 4, 5])
    #     >>> p = np.array([1.1, 1.9, 3.1, 4.2, 4.8])
    #     >>> metrics= RegressionMetrics(t, p)
    #     >>> score = metrics.relative_error()
    #     """
    #     return relative_error(true= self.true, predicted= self.predicted, treat_arrays=False, power= power)

    def manhattan_distance(self) -> float:
        """
        Manhattan distance, also known as cityblock distance or taxicab norm.

        .. math::
            D_{\\text{manhattan}} = \\sum_{i=1}^{n} \\left| \\text{true}_i - \\text{predicted}_i \\right|

        See `Blanco-Mallo et al., 2023 <https://doi.org/10.1016/j.patcog.2023.109646>`_ and `Cha et al., 2007 <https://pdodds.w3.uvm.edu/research/papers/others/everything/cha2007a.pdf>`_
         and `Alexei Botchkarev 2019 <https://www.ijikm.org/Volume14/IJIKMv14p045-076Botchkarev5064.pdf>`_ on the use of distances in performance measures.


        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.array([1, 2, 3, 4, 5])
        >>> p = np.array([1.1, 1.9, 3.1, 4.2, 4.8])
        >>> metrics= RegressionMetrics(t, p)
        >>> score = metrics.manhattan_distance()
        """
        return manhattan_distance(true= self.true, predicted= self.predicted, treat_arrays=False)

    def mse(self) -> float:
        """
        `Mean Square Error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html>`_

        .. math::
            MSE = \\frac{\\sum_{i=1}^{N} w_i (true_i - predicted_i)^2}{\\sum_{i=1}^{N} w_i}

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.mse()
            """
        return mse(true=self.true, predicted=self.predicted, treat_arrays=False, weights= None)


    def mape_for_peaks(self) -> float:
        """
        Mean Absolute Percentage Error for peaks which are found using
        `scipy.singnal.find_peaks <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`_

        .. math::
            \\text{MAPE}_\\text{peak} = \\frac{1}{P}\\sum_{p=1}^{P} \\left |\\frac{Q_{s,p} - Q_{o,p}}{Q_{o,p}} \\right | \\times 100,

        Examples
        ---------
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> t = np.random.random(10)
        >>> p = np.random.random(10)
        >>> metrics= RegressionMetrics(t, p)
        >>> metrics.mape_for_peaks()
            """
        return mape_for_peaks(true=self.true, predicted=self.predicted, treat_arrays=False, weights= None)

#*************************************
#        FUNCTIONAL API              #
#*************************************

def post_process_kge(cc, alpha, beta, return_all=False):
    kge_ = float(1 - np.sqrt((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    if return_all:
        return np.vstack((kge_, cc, alpha, beta))
    else:
        return kge_


def r2(true, predicted, treat_arrays: bool = True,
       **treat_arrays_kws) -> float:
    """
    R2 is a statistical measure of how well the regression line approximates the actual data.
    Quantifies the percent of variation in the response that the 'model'
    explains_. The 'model' here is anything from which we obtained predicted
    array. It is also called coefficient of determination or square of pearson
    correlation coefficient. More heavily affected by outliers than pearson correlatin r.

    .. _explains:
        https://data.library.virginia.edu/is-r-squared-useless/

    .. math ::
        R^2 = \\left( \\frac{\\sum_{i=1}^{N} \\left( \\frac{true_i - \\bar{true}}{\\sigma_{true}} \\cdot \\frac{predicted_i - \\bar{predicted}}{\\sigma_{predicted}} \\right)}{N - 1} \\right)^2

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import r2
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> r2(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)

    zx = (true - np.mean(true)) / np.std(true, ddof=1)
    zy = (predicted - np.mean(predicted)) / np.std(predicted, ddof=1)
    r = np.sum(zx * zy) / (len(true) - 1)
    return float(r ** 2)


def nse(true, predicted, treat_arrays: bool = True,
        **treat_arrays_kws) -> float:
    """Nash-Sutcliff Efficiency.

    The Nash-Sutcliffe efficiency (NSE) is a normalized statistic that determines
    the relative magnitude of the residual variance compared to the measured data
    variance It determines how well the model simulates trends for the output response
    of concern. But cannot help identify model bias and cannot be used to identify
    differences in timing and magnitude of peak flows and shape of recession curves;
    in other words, it cannot be used for single-event simulations. It is sensitive
    to extreme values due to the squared differ-ences `[1] <https://elibrary.asabe.org/abstract.asp?aid=46548>`_. To make it less sensitive
    to outliers, `[2] <https://dx.doi.org/10.5194/adgeo-5-89-2005>`_ proposed log and relative nse.

    .. math::
        \\text{NSE} = 1 - \\frac{\\sum_{i=1}^{N} (predicted_i - true_i)^2}{\\sum_{i=1}^{N} (true_i - \\bar{true})^2}
    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import nse
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> nse(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)

    _nse = 1 - sum((predicted - true) ** 2) / sum((true - np.mean(true)) ** 2)
    return float(_nse)


def nse_alpha(true, predicted, treat_arrays: bool = True,
              **treat_arrays_kws) -> float:
    """
    Alpha decomposition of the NSE, see `Gupta et al. 2009 <https://doi.org/10.1029/97WR03495>`_
    used in `kratzert et al., 2019 <https://hess.copernicus.org/articles/23/5089/2019/hess-23-5089-2019.html>`_.

    .. math::
        \\text{NSE}_{\\text{alpha}} = \\frac{\\sigma_{\\text{predicted}}}{\\sigma_{\\text{true}}}

    Returns
    -------
    float
        Alpha decomposition of the NSE
    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import nse_alpha
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> nse_alpha(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.std(predicted) / np.std(true))


def nse_beta(true, predicted, treat_arrays: bool = True,
             **treat_arrays_kws) -> float:
    """
    Beta decomposition of NSE. See `Gupta et al. 2009 <https://doi.org/10.1029/97WR03495>`_
    used in `kratzert et al., 2019 <https://hess.copernicus.org/articles/23/5089/2019/hess-23-5089-2019.html>`_.

    .. math::
        \\text{NSE}_{\\text{beta}} = \\frac{\\mu_{\\text{predicted}} - \\mu_{\\text{true}}}{\\sigma_{\\text{true}}}

    Returns
    -------
    float
        Beta decomposition of the NSE
    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import nse_beta
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> nse_beta(t, p)

    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float((np.mean(predicted) - np.mean(true)) / np.std(true))


def nse_mod(true, predicted, treat_arrays: bool = True,
            j=1,
            **treat_arrays_kws
            ) -> float:
    """
    Gives less weightage to outliers if j=1 and if j>1 then it gives more
    weightage to outliers. Reference: `Krause_ et al., 2005 <https://adgeo.copernicus.org/articles/5/89/2005/adgeo-5-89-2005.html>`_.

    .. math::
        \\text{NSE}_{\\text{mod}} = 1 - \\frac{\\sum_{i=1}^{N} \\left| \\text{predicted}_i - \\text{true}_i \\right|^j}{\\sum_{i=1}^{N} \\left| \\text{true}_i - \\bar{\text{true}} \\right|^j}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    j:

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import nse_mod
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> nse_mod(t, p)

    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    a = (np.abs(predicted - true)) ** j
    b = (np.abs(true - np.mean(true))) ** j
    return float(1 - (np.sum(a) / np.sum(b)))


def nse_rel(true, predicted,
            treat_arrays: bool = True,
            **treat_arrays_kws
            ) -> float:
    """
    `Relative Nash-Sutcliff Efficiency <https://doi.org/10.5194/adgeo-5-89-2005>`_.

    .. math::
        \\text{NSE}_{\\text{rel}} = 1 - \\frac{\\sum_{i=1}^{N} \\left( \\frac{|\\text{predicted}_i - \\text{true}_i|}{\\text{true}_i} \\right)^2}{\\sum_{i=1}^{N} \\left( \\frac{|\\text{true}_i - \\overline{\\text{true}}|}{\\overline{\\text{true}}} \\right)^2}


    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import nse_rel
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> nse_rel(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)

    a = (np.abs((predicted - true) / true)) ** 2
    b = (np.abs((true - np.mean(true)) / np.mean(true))) ** 2
    return float(1 - (np.sum(a) / np.sum(b)))


def nse_bound(true, predicted, treat_arrays: bool = True,
              **treat_arrays_kws) -> float:
    """
    Bounded Version of the Nash-Sutcliffe Efficiency (nse_)

    .. math::
        \\text{NSE}_{\\text{bound}} = \\frac{\\text{NSE}}{2 - \\text{NSE}}
    .. _nse:
        https://iahs.info/uploads/dms/13614.21--211-219-41-MATHEVET.pdf



    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import nse_bound
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> nse_bound(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    nse_ = nse(true, predicted, treat_arrays=False)
    nse_c2m_ = nse_ / (2 - nse_)
    return nse_c2m_


def r2_score(true, predicted, treat_arrays: bool = True, weights=None,
             **treat_arrays_kws):
    """
    This is not a symmetric function.
    Unlike most other scores, `R^2 score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html>`_
    score may be negative (it need not actually
    be the square of a quantity R).
    This metric is not well-defined for single samples and will return a NaN
    value if n_samples is less than two.

    .. math::
        \\text{R2}_{\\text{score}} = 1 - \\frac{\\sum_{i=1}^{n} w_i (\\text{true}_i - \\text{predicted}_i)^2}{\\sum_{i=1}^{n} w_i (\\text{true}_i - \\bar{\\text{true}})^2}


    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    weights:

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import r2_score
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> r2_score(t, p)

    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    if len(predicted) < 2:
        msg = "R^2 score is not well-defined with less than two samples."
        warnings.warn(msg)
        return None

    if weights is None:
        weight = 1.
    else:
        weight = weights[:, np.newaxis]

    numerator = (weight * (true - predicted) ** 2).sum(axis=0, dtype=np.float64)
    denominator = (weight * (true - np.average(true, axis=0, weights=weights)) ** 2).sum(axis=0, dtype=np.float64)

    if numerator == 0.0:
        return None
    output_scores = _foo(denominator, numerator)

    return float(np.average(output_scores, weights=weights))


def adjusted_r2(true, predicted, treat_arrays: bool = True,
                **treat_arrays_kws) -> float:
    """`Adjusted R squared <https://people.duke.edu/~rnau/rsquared.html>`_.

    .. math::
        \\text{Adjusted } R^2 = 1 - \\left( \\frac{(1 - R^2) \\cdot (n - 1)}{n - k - 1} \\right)

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import adjusted_r2
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> adjusted_r2(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    k = 1
    n = len(predicted)
    adj_r = 1 - ((1 - r2(true, predicted, treat_arrays=False)) * (n - 1)) / (n - k - 1)
    return float(adj_r)


def kge(true,
        predicted,
        treat_arrays: bool = True,
        return_all=False,
        **treat_arrays_kws):
    """
    Kling-Gupta Efficiency following `Gupta_ et al. 2009 <https://doi.org/10.1016/j.jhydrol.2009.08.003>`_.

    .. math::
        \\text{KGE} = 1 - \\sqrt{(r - 1)^2 + (\\alpha - 1)^2 + (\\beta - 1)^2}
    .. math::
        \\r = \\frac{\\sum_{i=1}^{N} ( \\text{true}_i - \\bar{\\text{true}} ) ( \\text{predicted}_i - \\bar{\\text{predicted}} )}{\\sqrt{\\sum_{i=1}^{N} ( \\text{true}_i - \\bar{\\text{true}} )^2} \\sqrt{\\sum_{i=1}^{N} ( \\text{predicted}_i - \\bar{\\text{predicted}} )^2}}
    .. math::
        \\alpha = \\frac{\\sigma_{\\text{predicted}}}{\\sigma_{\\text{true}}}
    .. math::
        \\beta = \\frac{\\mu_{\\text{predicted}}}{\\mu_{\\text{true}}}

    output:
        If return_all is True, it returns a numpy array of shape (4, ) containing
        kge, cc, alpha, beta. Otherwise, it returns kge.

        kge: Kling-Gupta Efficiency
        cc: correlation
        alpha: ratio of the standard deviation
        beta: ratio of the mean

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    return_all:

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import kge
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> kge(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    cc = np.corrcoef(true, predicted)[0, 1]
    alpha = np.std(predicted) / np.std(true)
    beta = np.sum(predicted) / np.sum(true)
    return post_process_kge(cc, alpha, beta, return_all)


def kge_bound(true, predicted, treat_arrays: bool = True,
              **treat_arrays_kws) -> float:
    """
    Bounded Version of the Original Kling-Gupta Efficiency after
    `Mathevet et al. 2006 <https://iahs.info/uploads/dms/13614.21--211-219-41-MATHEVET.pdf>`_.

    .. math::
        \\text{KGE}_{\\text{bound}} = \\frac{\\text{KGE}}{2 - \\text{KGE}}


    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import kge_bound
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> kge_bound(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    kge_ = kge(true, predicted, return_all=True, treat_arrays=False)[0, :]
    kge_c2m_ = kge_ / (2 - kge_)

    return float(kge_c2m_)


def kge_mod(true, predicted, treat_arrays: bool = True, return_all=False,
            **treat_arrays_kws):
    """
    Modified Kling-Gupta Efficiency after `Kling et al. 2012 <https://doi.org/10.1016/j.jhydrol.2012.01.011>`_.

    .. math::
        \\text{KGE}_{\\text{mod}} = 1 - \\sqrt{ \\left( \\frac{\\sum_{i=1}^{n} (true_i - \\bar{true})(predicted_i - \\bar{predicted})}{\\sqrt{\\sum_{i=1}^{n} (true_i - \\bar{true})^2} \\sqrt{\\sum_{i=1}^{n} (predicted_i - \\bar{predicted})^2}} - 1 \\right)^2 +
        \\left( \\frac{\\frac{\\sigma_{predicted}}{\\bar{predicted}}}{\\frac{\\sigma_{true}}{\\bar{true}}} - 1 \\right)^2 +
        \\left( \\frac{\\bar{predicted}}{\\bar{true}} - 1 \\right)^2}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    return_all:

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import kge_mod
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> kge_mod(t, p)

    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    # calculate error in timing and dynamics r (Pearson's correlation coefficient)
    sim_mean = np.mean(predicted, axis=0, dtype=np.float64)
    obs_mean = np.mean(true, dtype=np.float64)
    r = np.sum((predicted - sim_mean) * (true - obs_mean), axis=0, dtype=np.float64) / \
        np.sqrt(np.sum((predicted - sim_mean) ** 2, axis=0, dtype=np.float64) *
                np.sum((true - obs_mean) ** 2, dtype=np.float64))
    # calculate error in spread of flow gamma (avoiding cross correlation with bias by dividing by the mean)
    gamma = (np.std(predicted, axis=0, dtype=np.float64) / sim_mean) / \
            (np.std(true, dtype=np.float64) / obs_mean)
    # calculate error in volume beta (bias of mean discharge)
    beta = np.mean(predicted, axis=0, dtype=np.float64) / np.mean(true, axis=0, dtype=np.float64)
    # calculate the modified Kling-Gupta Efficiency KGE'
    return post_process_kge(r, gamma, beta, return_all)


def kge_np(
        true,
        predicted,
        treat_arrays: bool = True,
        return_all=False,
        **treat_arrays_kws):
    """
    Non-parametric Kling-Gupta Efficiency after `Pool et al. 2018 <https://doi.org/10.1080/02626667.2018.1552002>`_.

    .. math::
        cc = \\rho(\\text{true}, \\text{predicted})
    .. math::
        \\alpha = 1 - 0.5 \\sum_{i=1}^{n} \\left| \\frac{\\text{sorted(predicted}_i\\text{)}}{\\text{mean(predicted)} \\cdot n} - \\frac{\\text{sorted(true}_i\\text{)}}{\\text{mean(true)} \\cdot n} \\right|
    .. math::
        \\beta = \\frac{\\text{mean(predicted)}}{\\text{mean(true)}}
    .. math::
        \\text{KGE}_{\\text{np}} = 1 - \\sqrt{(cc - 1)^2 + (\\alpha - 1)^2 + (\\beta - 1)^2}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    return_all :

    output
    ------
        kge: Kling-Gupta Efficiency
        cc: correlation
        alpha: ratio of the standard deviation
        beta: ratio of the mean

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import kge_np
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> kge_np(t, p)

    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    # # self-made formula
    cc = spearmann_corr(true, predicted, treat_arrays=False)

    fdc_sim = np.sort(predicted / (np.nanmean(predicted) * len(predicted)))
    fdc_obs = np.sort(true / (np.nanmean(true) * len(true)))
    alpha = 1 - 0.5 * np.sum(np.abs(fdc_sim - fdc_obs))

    beta = np.mean(predicted) / np.mean(true)
    return post_process_kge(cc, alpha, beta, return_all)


def spearmann_corr(
        true,
        predicted,
        treat_arrays:bool = True,
        **treat_arrays_kws
)->float:
    """
    `Separmann correlation coefficient <https://hess.copernicus.org/articles/24/2505/2020/hess-24-2505-2020.pdf>`_.

    .. math::
        r = \\frac{\\sum_{i=1}^{n} \\left( R_{t,i} - \\overline{R_t} \\right) \\left( R_{p,i} - \\overline{R_p} \\right)}{\\sqrt{ \\sum_{i=1}^{n} \\left( R_{t,i} - \\overline{R_t} \\right)^2 \\sum_{i=1}^{n} \\left( R_{p,i} - \\overline{R_p} \\right)^2 }}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import spearmann_corr
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> spearmann_corr(t, p)
    """

    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)

    sim_rank = np.argsort(np.argsort(predicted, axis=0), axis=0)
    obs_rank = np.argsort(np.argsort(true, axis=0), axis=0)

    r_num = np.sum((obs_rank - np.mean(obs_rank, axis=0, dtype=np.float64))
                    * (sim_rank - np.mean(sim_rank, axis=0, dtype=np.float64)),
                    axis=0)
    r_den = np.sqrt(
        np.sum((obs_rank - np.mean(obs_rank, axis=0, dtype=np.float64)) ** 2,
                axis=0)
        * np.sum((sim_rank - np.mean(sim_rank, axis=0, dtype=np.float64)) ** 2,
                    axis=0)
    )
    r = r_num / r_den

    return float(r)

# def spearmann_rank_corr(
#         true,
#         predicted,
#         treat_arrays: bool = True,
#         **treat_arrays_kws
# ) -> float:
#     """Separmann rank correlation coefficient_.
#
#     This is a nonparametric metric and assesses how well the relationship
#     between the true and predicted data can be described using a monotonic
#     function.
#
#     .. _coefficient:
#         https://hess.copernicus.org/articles/24/2505/2020/hess-24-2505-2020.pdf
#
#     Parameters
#     ----------
#     true :
#          true/observed/actual/target values. It must be a numpy array,
#          or pandas series/DataFrame or a list.
#     predicted :
#          simulated values
#     treat_arrays :
#         process the true and predicted arrays using maybe_treat_arrays function
#
#     Examples
#     ---------
#     >>> import numpy as np
#     >>> from SeqMetrics import spearmann_rank_corr
#     >>> t = np.random.random(10)
#     >>> p = np.random.random(10)
#     >>> spearmann_rank_corr(t, p)
#     """
#     true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
#     col = [list(a) for a in zip(true, predicted)]
#     xy = sorted(col, key=lambda _x: _x[0], reverse=False)
#     # rang of x-value
#     for i, row in enumerate(xy):
#         row.append(i + 1)
#
#     a = sorted(xy, key=lambda _x: _x[1], reverse=False)
#     # rang of y-value
#     for i, row in enumerate(a):
#         row.append(i + 1)
#
#     mw_rank_x = np.nanmean(np.array(a)[:, 2])
#     mw_rank_y = np.nanmean(np.array(a)[:, 3])
#
#     numerator = np.nansum([float((a[j][2] - mw_rank_x) * (a[j][3] - mw_rank_y)) for j in range(len(a))])
#     denominator1 = np.sqrt(np.nansum([(a[j][2] - mw_rank_x) ** 2. for j in range(len(a))]))
#     denominator2 = np.sqrt(np.nansum([(a[j][3] - mw_rank_x) ** 2. for j in range(len(a))]))
#     return float(numerator / (denominator1 * denominator2))


# def log_nse(true, predicted, treat_arrays: bool = True, epsilon=0.0,
#             **treat_arrays_kws) -> float:
#     """
#     log Nash-Sutcliffe model efficiency
#
#     .. math::
#         NSE = 1-\\frac{\\sum_{i=1}^{N}(log(e_{i})-log(s_{i}))^2}{\\sum_{i=1}^{N}(log(e_{i})-log(\\bar{e})^2}-1)*-1
#
#     Parameters
#     ----------
#     true :
#          true/observed/actual/target values. It must be a numpy array,
#          or pandas series/DataFrame or a list.
#     predicted :
#          simulated values
#     treat_arrays :
#         process the true and predicted arrays using maybe_treat_arrays function
#     epsilon :
#
#         Examples
#     ---------
#     >>> import numpy as np
#     >>> from SeqMetrics import log_nse
#     >>> t = np.random.random(10)
#     >>> p = np.random.random(10)
#     >>> log_nse(t, p)
#
#     """
#     true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
#     s, o = predicted + epsilon, true + epsilon  # todo, check why s is here
#     return float(1 - sum((np.log(o) - np.log(o)) ** 2) / sum((np.log(o) - np.mean(np.log(o))) ** 2))


def corr_coeff(true, predicted, treat_arrays: bool = True,
               **treat_arrays_kws) -> float:
    """
    `Pearson correlation coefficient <https://royalsocietypublishing.org/doi/abs/10.1098/rsta.1895.0010>`_.
    It measures linear correlatin between true and predicted arrays.
    It is sensitive to outliers.

    .. math::
        r = \\frac{\\sum ^n _{i=1}(e_i - \\bar{e})(s_i - \\bar{s})}{\\sqrt{\\sum ^n _{i=1}(e_i - \\bar{e})^2}
         \\sqrt{\\sum ^n _{i=1}(s_i - \\bar{s})^2}}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import corr_coeff
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> corr_coeff(t, p)

    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    correlation_coefficient = np.corrcoef(true, predicted)[0, 1]
    return float(correlation_coefficient)


def rmse(true, predicted, treat_arrays: bool = True, weights=None,
         **treat_arrays_kws) -> float:
    """ `Root mean squared error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html>`_

    .. math::
        \\text{RMSE} = \\sqrt{\\frac{\\sum_{i=1}^{n} w_i (\\text{true}_i - \\text{predicted}_i)^2}{\\sum_{i=1}^{n} w_i}}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    weights:

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import rmse
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> rmse(t, p)
    """
    if treat_arrays:
        true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return sqrt(np.average((true - predicted) ** 2, axis=0, weights=weights))


def rmsle(true, predicted, treat_arrays: bool = True,
          **treat_arrays_kws) -> float:
    """Root mean square log error.

    This error is less sensitive to `outliers <https://stats.stackexchange.com/q/56658/314919>`_ .
    Compared to RMSE, RMSLE only considers the relative error between predicted
    and actual values, and the scale of the error is nullified by the log-transformation.
    Furthermore, RMSLE penalizes underestimation more than overestimation.
    This is especially useful in those studies where the underestimation
    of the target variable is not acceptable but overestimation can be
    `tolerated <https://doi.org/10.1016/j.scitotenv.2020.137894>`_ .

    .. math::
        RMSLE = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} \\left( \\log(1 + \\text{predicted}_i) - \\log(1 + \\text{true}_i) \\right)^2}

    References
    ----------
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_log_error.html

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import rmsle
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> rmsle(t, p)

     """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.sqrt(np.mean(np.power(log1p(predicted) - log1p(true), 2))))


def mape(
        true, 
        predicted, 
        treat_arrays: bool = True,
         **treat_arrays_kws
         ) -> float:
    """ Mean Absolute Percentage Error.
    The MAPE is often used when the quantity to predict is known to remain
    way above zero_. It is useful when the size or size of a prediction variable
    is significant in evaluating the accuracy of a prediction_. It has advantages
    of scale-independency and interpretability. However, it has the significant
    disadvantage that it produces infinite or undefined values for zero or
    close-to-zero actual values_.

    .. _zero:
        https://doi.org/10.1016/j.neucom.2015.12.114

    .. _prediction:
        https://doi.org/10.1088/1742-6596/930/1/012002

    .. _values:
        https://doi.org/10.1016/j.ijforecast.2015.12.003

    .. math::
        MAPE = \\frac{1}{n} \\sum_{i=1}^{n} \\left| \\frac{true_i - predicted_i}{true_i} \\right| \\times 100

    References
    ---------
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import mape
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> mape(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.mean(np.abs((true - predicted) / true)) * 100)


def nrmse(true, predicted, treat_arrays: bool = True,
          **treat_arrays_kws) -> float:
    """ `Normalized Root Mean Squared Error <https://www.sciencedirect.com/science/article/pii/S0957417411003289>`_

    .. math::
        NRMSE = \\frac{\\sqrt{\\frac{1}{N} \\sum_{i=1}^{N} (\\text{true}_i - \\text{predicted}_i)^2}}{\\max(\\text{true}) - \\min(\text{true})}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import nrmse
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> nrmse(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(rmse(true, predicted, treat_arrays=False) / (np.max(true) - np.min(true)))


def pbias(true, predicted, treat_arrays: bool = True,
          **treat_arrays_kws) -> float:
    """
    Percent Bias.
    It determines how well the model simulates the average magnitudes for the
    output response of interest. It can also determine over and under-prediction.
    It cannot be used (1) for single-event simula-tions to identify differences
    in timing and magnitude of peak flows and the shape of recession curves nor (2)
    to determine how well the model simulates residual variations and/or trends
    for the output response of interest. It can  give a deceiving rating of
    model performance if the model overpredicts as much as it underpredicts,
    in which case PBIAS will be close to zero even though the model simulation
    is poor. `[1] <https://elibrary.asabe.org/abstract.asp?aid=46548>`_

    .. math::
        PBIAS = 100 \\times \\frac{\\sum_{i=1}^{N} (\\text{true}_i - \\text{predicted}_i)}{\\sum_{i=1}^{N} \\text{true}_i}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import pbias
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> pbias(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(100.0 * sum(true - predicted) / sum(true))


def bias(true, predicted, treat_arrays: bool = True,
         **treat_arrays_kws) -> float:
    """
    Bias as and given by Gupta1998_ et al., 1998 in Table 1

    .. math::
        Bias=\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})

    .. _Gupta1998:
        https://doi.org/10.1029/97WR03495

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import bias
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> bias(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    bias_ = np.nansum(true - predicted) / len(true)
    return float(bias_)


def mae(true, predicted, treat_arrays: bool = True,
        **treat_arrays_kws) -> float:
    """ `Mean Absolute Error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html>`_.
    It is less sensitive to outliers as compared to mse/rmse.

    .. math::
        \\text{MAE} = \\frac{1}{n} \\sum_{i=1}^{n} \\left| \\text{true}_i - \\text{predicted}_i \\right|

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import mae
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> mae(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    if true is None:
        true = true
    if predicted is None:
        predicted = predicted
    return float(np.mean(np.abs(true - predicted)))


# def abs_pbias(true, predicted, treat_arrays: bool = True,
#               **treat_arrays_kws) -> float:
#     """Absolute Percent bias
#
#     Parameters
#     ----------
#     true :
#          true/observed/actual/target values. It must be a numpy array,
#          or pandas series/DataFrame or a list.
#     predicted :
#          simulated values
#     treat_arrays :
#         process the true and predicted arrays using maybe_treat_arrays function
#
#     Examples
#     ---------
#     >>> import numpy as np
#     >>> from SeqMetrics import abs_pbias
#     >>> t = np.random.random(10)
#     >>> p = np.random.random(10)
#     >>> abs_pbias(t, p)
#     """
#     true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
#     _apb = 100.0 * sum(abs(predicted - true)) / sum(true)
#     return float(_apb)


def gmae(true, predicted, treat_arrays: bool = True,
         **treat_arrays_kws) -> float:
    """ `Geometric Mean Absolute Error <https://doi.org/10.1016/j.isprsjprs.2024.04.015>`_

    .. math::
        GMAE = \\left( \\prod_{i=1}^{n} \\left| \\text{true}_i - \\text{predicted}_i \\right| \\right)^{\\frac{1}{n}}


    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import gmae
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> gmae(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    error = true - predicted
    return _geometric_mean(np.abs(error))


def inrse(true, predicted, treat_arrays: bool = True,
          **treat_arrays_kws) -> float:
    """ `Integral Normalized Root Squared Error <https://doi.org/10.1016/j.engappai.2023.107559>`_

    .. math::
        IN\\text{-}RSE = \\sqrt{\\frac{\\sum_{i=1}^{n} (\\text{true}_i - \\text{predicted}_i)^2}{\\sum_{i=1}^{n} (\\text{true}_i - \\overline{\\text{true}})^2}}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import inrse
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> inrse(t, p)

    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    error = true - predicted
    return float(np.sqrt(np.sum(np.square(error)) / np.sum(np.square(true - np.mean(true)))))


def irmse(true, predicted, treat_arrays: bool = True,
          **treat_arrays_kws) -> float:
    """ `Inertial RMSE <https://link.springer.com/article/10.1007/s11069-008-9299-2>`_.
    RMSE divided by standard deviation of the gradient of true.

    .. math::
        \\text{IRMSE} = \\frac{\\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} \\left( \\text{true}_i - \\text{predicted}_i \\right)^2}}{\\sqrt{\\frac{1}{n-2} \\sum_{i=1}^{n-1} \\left( (\\text{true}_{i+1} - \\text{true}_i) - \\overline{(\\text{true}_{i+1} - \\text{true}_i)} \\right)^2}}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import irmse
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> irmse(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    # Getting the gradient of the observed data
    obs_len = true.size
    obs_grad = true[1:obs_len] - true[0:obs_len - 1]

    # Standard deviation of the gradient
    obs_grad_std = np.std(obs_grad, ddof=1)

    # Divide RMSE by the standard deviation of the gradient of the observed data
    return float(rmse(true, predicted, treat_arrays=False) / obs_grad_std)


def mase(true, predicted, treat_arrays: bool = True, seasonality: int = 1, **treat_arrays_kws):
    """
    Mean Absolute Scaled Error. Baseline (benchmark) is computed with naive
    forecasting (shifted by @seasonality) modified after `this <https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9>`_. It is the
    ratio of MAE of used model and MAE of naive forecast.

    .. math::
        \\text{MASE} = \\frac{\\frac{1}{n} \\sum_{i=1}^{n} \\left| \\text{true}_i - \\text{predicted}_i \\right|}{\\frac{1}{n-s} \\sum_{i=s+1}^{n} \\left| \\text{true}_i - \\text{true}_{i-s} \\right|}


    References
    ----------

    `Hyndman, R. J. (2006). Another look at forecast-accuracy metrics for intermittent demand.
    Foresight: The International Journal of Applied Forecasting, 4(4), 43-46. <http://datascienceassn.org/sites/default/files/Another%20Look%20at%20Measures%20of%20Forecast%20Accuracy.pdf>`_

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
        process the true and predicted arrays using maybe_treat_arrays function
    seasonality:

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import mase
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> mase(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)

    return mae(true, predicted, False) / mae(true[seasonality:], _naive_prognose(true, seasonality), treat_arrays=False)


def mare(true, predicted, treat_arrays: bool = True, **treat_arrays_kws) -> float:
    """ Mean Absolute Relative Error. When expressed in %age, it is also known as mape_.

    .. math::
        \\text{MARE} = \\frac{1}{n} \\sum_{i=1}^{n} \\left| \\frac{\\text{true}_i - \\text{predicted}_i}{\\text{true}_i} \\right|

    .. _mape:
        https://doi.org/10.1016/j.rser.2015.08.035

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import mare
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> mare(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    error = true - predicted
    return float(np.mean(np.abs(error / true)))


def msle(true, predicted, treat_arrays=True, weights=None, **treat_arrays_kws) -> float:
    """
    `Mean square logrithmic error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_log_error.html>`_

    .. math::
        \\text{MSLE} = \\frac{\\sum_{i=1}^{n} w_i \\cdot \\text{sq_log_error}_i}{\\sum_{i=1}^{n} w_i}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    weights:

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import msle
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> msle(t, p)
    """

    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.average((log1p(true) - log1p(predicted)) ** 2, axis=0, weights=weights))


def log1p(array):
    with np.errstate(**ERR_STATE):
        return np.log1p(array)


def covariance(
        true, 
        predicted, 
        treat_arrays: bool = True,
        **treat_arrays_kws
        ) -> float:
    """
    `Covariance <https://scikit-learn.org/stable/api/sklearn.covariance.html>`_

    .. math::
        Covariance = \\frac{1}{N} \\sum_{i=1}^{N}((e_{i} - \\bar{e}) * (s_{i} - \\bar{s}))

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import covariance
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> covariance(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    obs_mean = np.mean(true)
    sim_mean = np.mean(predicted)
    covariance_ = np.mean((true - obs_mean) * (predicted - sim_mean))
    return float(covariance_)


def brier_score(true, predicted, treat_arrays: bool = True,
                **treat_arrays_kws) -> float:
    """
    Adopted from SkillMetrics_
    Calculates the Brier score (BS), a measure of the mean-square error of
    probability forecasts for a dichotomous (two-category) event, such as
    the occurrence/non-occurrence of precipitation. The score is calculated
    using the formula:

    .. math::
        BS = sum_(n=1)^N (f_n - o_n)^2/N

    where f is the forecast probabilities, o is the observed probabilities
    (0 or 1), and N is the total number of values in f & o. Note that f & o
    must have the same number of values, and those values must be in the
    range_ [0,1].

    Returns
    --------
    float
        BS : Brier score

    References
    ---------
    `Glenn W. Brier, 1950: Verification of forecasts expressed in terms
    of probabilities. Mon. We. Rev., 78, 1-23.
    D. S. Wilks, 1995: Statistical Methods in the Atmospheric Sciences.
    Cambridge Press. 547 pp <https://viterbi-web.usc.edu/~shaddin/teaching/cs699fa17/docs/Brier50.pdf>`_

    .. _SkillMetrics:
        https://github.com/PeterRochford/SkillMetrics/blob/master/skill_metrics/brier_score.py

    .. _range:
        https://data.library.virginia.edu/a-brief-on-brier-scores/
    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import brier_score
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> brier_score(t, p)

    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    # Check for valid values

    index = np.where(np.logical_and(true != 0, true != 1))
    if np.sum(index) > 0:
        msg = 'Observed has values not equal to 0 or 1.'
        raise ValueError(msg)

    index = np.where(np.logical_or(predicted < 0, predicted > 1))
    if np.sum(index) > 0:
        msg = 'Forecast has values outside interval [0,1].'
        raise ValueError(msg)

    # Calculate score
    bs = np.sum(np.square(predicted - true)) / len(predicted)

    return bs


def bic(
        true, 
        predicted, 
        treat_arrays: bool = True, 
        p=1,
        **treat_arrays_kws) -> float:
    """
    Bayesian Information Criterion

    Minimising the BIC_ is intended to give the best model. The
    model chosen by the BIC is either the same as that chosen by the AIC, or one
    with fewer terms. This is because the BIC penalises the number of parameters
    more heavily than the AIC.  Modified after RegscorePy_.

    .. math::
        BIC = n \\cdot \\ln\\left(\\frac{\\text{SSE}}{n}\\right) + p \\cdot \\ln(n)


    .. _BIC:
        https://otexts.com/fpp2/selecting-predictors.html#schwarzs-bayesian-information-criterion

    .. _RegscorePy:
        https://github.com/UBC-MDS/RegscorePy/blob/master/RegscorePy/bic.py
    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    p:

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import bic
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> bic(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    assert p >= 0

    n = len(true)
    return float(n * np.log(sse(true, predicted, treat_arrays=False) / n) + p * np.log(n))


def sse(true, predicted, treat_arrays: bool = True,
        **treat_arrays_kws) -> float:
    """
    Sum of squared errors_ (model vs actual). It is measure of how far off
    our model's predictions are from the observed values. A value of 0 indicates
    that all predications are spot on. A non-zero value indicates errors_.

    This is also called residual sum of squares (RSS) or sum of squared residuals
    as per tutorialspoint_ .

    .. math::
        \\text{SSE} = \\sum_{i=1}^{n} (true_i - predicted_i)^2

    .. _errors:
        https://dziganto.github.io/data%20science/linear%20regression/machine%20learning/python/Linear-Regression-101-Metrics/

    .. _tutorialspoint:
        https://www.tutorialspoint.com/statistics/residual_sum_of_squares.html
    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import sse
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> sse(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    squared_errors = (true - predicted) ** 2
    return float(np.sum(squared_errors))


def amemiya_pred_criterion(true, predicted, treat_arrays: bool = True,
                           **treat_arrays_kws) -> float:
    """`Amemiya's Prediction Criterion <https://www.sfu.ca/sasdoc/sashtml/ets/chap30/sect19.htm#:~:text=Amemiya>`_

    .. math::
        \\text{APC} = \\left( \\frac{n + k}{n - k} \\right) \\left( \\frac{1}{n} \\sum_{i=1}^{n} (\\text{true}_i - \\text{predicted}_i)^2 \\right)

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import amemiya_pred_criterion
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> amemiya_pred_criterion(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression',
                                       **treat_arrays_kws)
    k = 1
    n = len(predicted)
    return float(((n + k) / (n - k)) * (1 / n) * sse(true, predicted, treat_arrays=False))


def amemiya_adj_r2(true, predicted, treat_arrays: bool = True,
                   **treat_arrays_kws) -> float:
    """`Amemiya's Adjusted R-squared <https://www.sfu.ca/sasdoc/sashtml/ets/chap30/sect19.htm#:~:text=Amemiya>`_

    .. math::
        R^2_{\\text{adj, Amemiya}} = 1 - \\left( \\frac{(1 - R^2) \\cdot (n + k)}{n - k - 1} \\right)

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import amemiya_adj_r2
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> amemiya_adj_r2(t, p)
    """
    if treat_arrays:
        true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    k = 1
    n = len(predicted)
    adj_r = 1 - ((1 - r2(true, predicted, treat_arrays=False)) * (n + k)) / (n - k - 1)
    return float(adj_r)


def aitchison(true, predicted, treat_arrays: bool = True, center='mean',
              **treat_arrays_kws) -> float:
    """Aitchison distance. used in Zhang_ et al., 2020

    .. math::
        d_{\\text{Aitchison}} = \\sqrt{\\sum_{i=1}^{n} \\left( \\log(\\text{true}_i) - \\text{center}(\\log(\\text{true})) - \\left(\\log(\\text{predicted}_i) - \\text{center}(\\log(\\text{predicted}))\\right) \\right)^2}

    .. _Zhang:
        https://doi.org/10.5194/hess-24-2505-2020
    
    https://doi.org/10.1007/bf00891269 

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    center:

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import aitchison
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> aitchison(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    lx = log_t(true)
    ly = log_p(predicted)
    if center.upper() == 'MEAN':
        m = np.mean
    elif center.upper() == 'MEDIAN':
        m = np.median
    else:
        raise ValueError

    clr_x = lx - m(lx)
    clr_y = ly - m(ly)
    d = (sum((clr_x - clr_y) ** 2)) ** 0.5
    return float(d)


def log_t(true):
    with np.errstate(**ERR_STATE):
        return np.log(true)


def log_p(predicted):
    with np.errstate(**ERR_STATE):
        return np.log(predicted)


def _assert_greater_than_one(true, predicted):
    # assert that both true and predicted arrays are greater than one.

    if len(true) <= 1 or len(predicted) <= 1:
        raise ValueError(f"""
        Expect length of true and predicted arrays to be larger than 1 but 
        they are {len(true)} and {len(predicted)}""")
    return


def acc(
        true, 
        predicted, 
        treat_arrays: bool = True,
        **treat_arrays_kws
        ) -> float:
    """Anomaly correction coefficient. See Langland_ et al., 2012; Miyakoda_ et al., 1972
    and Murphy_ et al., 1989.

    .. math::
        ACC = \\frac{\\sum_{i=1}^{N} \\left( (\\text{predicted}_i - \\overline{\\text{predicted}})(\\text{true}_i - \\overline{\\text{true}}) \\right)}{(N-1) \\cdot \\sigma_{\\text{true}} \\cdot \\sigma_{\\text{predicted}}}

    .. _Langland:
        https://doi.org/10.3402/tellusa.v64i0.17531

    .. _Miyakoda:

    .. _Murphy:
        https://doi.org/10.1080/02723646.1972.10642213

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import acc
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> acc(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    a = predicted - np.mean(predicted)
    b = true - np.mean(true)
    c = np.std(true, ddof=1) * np.std(predicted, ddof=1) * predicted.size
    return float(np.dot(a, b / c))


def agreement_index(true, predicted, treat_arrays: bool = True,
                    **treat_arrays_kws) -> float:
    """
    Agreement Index (d) developed by Willmott_, 1981.

    It detects additive and pro-portional differences in the observed and
    simulated means and vari-ances Moriasi_ et al., 2015. It is overly sensitive
    to extreme values due to the squared `differences <https://www.tandfonline.com/doi/abs/10.2747/0272-3646.26.6.467>`_.
    It can also be used
    as a substitute for R2 to identify the degree to which model predic-tions
    are error-free. Its value varies between 0 and 1 with 1 being the best.

    .. math::
        d = 1 - \\frac{\\sum_{i=1}^{N}(e_{i} - s_{i})^2}{\\sum_{i=1}^{N}(\\left | s_{i} - \\bar{e}
         \\right | + \\left | e_{i} - \\bar{e} \\right |)^2}

    .. _Willmott:
        https://doi.org/10.1080/02723646.1981.10642213

    .. _Moriasi:
        https://doi.org/10.13031/trans.58.10715

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import agreement_index
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> agreement_index(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    agreement_index_ = 1 - (np.sum((true - predicted) ** 2)) / (np.sum(
        (np.abs(predicted - np.mean(true)) + np.abs(true - np.mean(true))) ** 2))
    return float(agreement_index_)

def legates_coeff_eff(true, predicted, treat_arrays: bool = True,
                    **treat_arrays_kws) -> float:
    """
    Legates Coefficient of Efficiency. Its value varies between 0 and 1.
    It is not as sensitive to extreme values as agreement_index and coefficcient of
    determination because of the utilization of the absolute value of the difference
    instead of the squared difference. See Equaltion 23 in `Dodo et al., 2022 <https://doi.org/10.1016/j.nexus.2022.100157>`_

    .. math::
        LCE = 1 - \\frac{\\sum_{i=1}^{n} |true_i - predicted_i|}{\\sum_{i=1}^{n} |true_i - \\bar{true}|}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import legates_coeff_eff
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> agreement_index(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    agreement_index_ = 1 - (np.sum((true - predicted) ** 2)) / (np.sum(np.abs(predicted - np.mean(true))))
    return float(agreement_index_)


def aic(
        true, 
        predicted, 
        treat_arrays: bool = True, p=1,
        **treat_arrays_kws
        ) -> float:
    """
    Akaike_ Information Criterion. Modifying from this sourcee_

    .. math::
        AIC = n \\cdot \\ln\\left(\\frac{\\sum_{i=1}^{n} (\\text{true}_i - \\text{predicted}_i)^2}{n}\\right) + 2p

    .. _Akaike:
        https://doi.org/10.1007/978-1-4612-1694-0_15

    .. _sourcee:
        https://github.com/UBC-MDS/RegscorePy/blob/master/RegscorePy/aic.py
    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    p:

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import aic
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> aic(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    assert p > 0
    _assert_greater_than_one(true, predicted)  # noac

    n = len(true)
    resid = np.subtract(predicted, true)
    rss = np.sum(np.power(resid, 2))
    return float(n * np.log(rss / n) + 2 * p)


def cronbach_alpha(true, predicted, treat_arrays: bool = True,
                   **treat_arrays_kws) -> float:
    """
    It is a measure of internal consitency of data. See ucla_ and stackoverflow_
    pages for more info.

    .. math::
        alpha = \\frac{N}{N - 1} \\left(1 - \\frac{\\sum_{i=1}^{N} \\sigma^2_{i}}{\\sigma^2_{\\text{total}}}\\right)

    https://doi.org/10.1016/B0-12-369398-5/00396-0

    .. _ucla:
        https://stats.idre.ucla.edu/spss/faq/what-does-cronbachs-alpha-mean/

    .. _stackoverflow:
        https://stackoverflow.com/a/20799687/5982232

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import cronbach_alpha
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> cronbach_alpha(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    itemscores = np.stack([true, predicted])
    itemvars = itemscores.var(axis=1, ddof=1)
    tscores = itemscores.sum(axis=0)
    nitems = len(itemscores)
    return float(nitems / (nitems - 1.) * (1 - itemvars.sum() / tscores.var(ddof=1)))


def centered_rms_dev(true, predicted, treat_arrays: bool = True,
                     **treat_arrays_kws) -> float:
    """
    `Modified after SkillMetrics <https://doi.org/10.1016/B0-12-227410-5/00612-8>`_.
    Calculates the centered root-mean-square (RMS) difference between true and predicted
    using the formula:
    (E')^2 = sum_(n=1)^N [(p_n - mean(p))(r_n - mean(r))]^2/N
    where p is the predicted values, r is the true values, and
    N is the total number of values in p & r.

    .. math::
        CRMSD = \\sqrt{\\frac{1}{N} \\sum_{i=1}^{N} \\left( (p_i - \\text{mean}(p)) - (r_i - \\text{mean}(r)) \\right)^2}

    Output:
    CRMSDIFF : centered root-mean-square (RMS) difference (E')^2

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import centered_rms_dev
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> centered_rms_dev(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    # Calculate means
    pmean = np.mean(predicted)
    rmean = np.mean(true)

    # Calculate (E')^2
    crmsd = np.square((predicted - pmean) - (true - rmean))
    crmsd = np.sum(crmsd) / predicted.size
    crmsd = np.sqrt(crmsd)

    return float(crmsd)


def cosine_similarity(true, predicted, treat_arrays: bool = True,
                      **treat_arrays_kws) -> float:
    """
    It is a judgment of orientation and not magnitude: two vectors with
    the same orientation have a cosine similarity of 1, two vectors oriented
    at 90° relative to each other have a similarity of 0, and two vectors diametrically
    opposed have a similarity of -1, independent of their magnitude. `See <https://doi.org/10.1016/B978-0-12-804452-0.00002-6>`_

    .. math::
        \\text{Cosine Similarity} = \\frac{\\sum_{i=1}^{n} \\text{true}_i \\cdot \\text{predicted}_i}{\\sqrt{\\sum_{i=1}^{n} (\\text{true}_i)^2} \\cdot \\sqrt{\\sum_{i=1}^{n} (\\text{predicted}_i)^2}}


    References
    ----------
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import cosine_similarity
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> cosine_similarity(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.dot(true, predicted) /
                 (np.linalg.norm(true) * np.linalg.norm(predicted)))


def decomposed_mse(true, predicted, treat_arrays: bool = True,
                   **treat_arrays_kws) -> float:
    """
    Decomposed MSE developed by `Kobayashi and Salam (2000) <https://doi.org/10.2134/agronj2000.922345x>`_ Equation 24

    .. math ::
        dMSE = (\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i}))^2 + SDSD + LCS

    .. math::
        SDSD = (\\sigma(e) - \\sigma(s))^2

    .. math::
        LCS = 2 \\sigma(e) \\sigma(s) * (1 - \\frac{\\sum ^n _{i=1}(e_i - \\bar{e})(s_i - \\bar{s})}
        {\\sqrt{\\sum ^n _{i=1}(e_i - \\bar{e})^2} \\sqrt{\\sum ^n _{i=1}(s_i - \\bar{s})^2}})


    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import decomposed_mse
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> decomposed_mse(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    t_std = np.std(true)
    s_std = np.std(predicted)

    bias_squared = bias(true, predicted, treat_arrays=False) ** 2
    # sdsd is difference in the magnitude of fluctuation between predicted and true
    # bigger sdsd means the model fails to simulate the magnitude of the fluctuations in the true data
    sdsd = (s_std - t_std) ** 2
    # lack of positive correlation weighted by the std
    # bigger lcs means the models fails to simulate the patters of the fluctuations in the true data
    lcs = 2 * t_std * s_std * (1 - corr_coeff(true, predicted, treat_arrays=False))

    decomposed_mse_ = bias_squared + sdsd + lcs

    return float(decomposed_mse_)


def euclid_distance(true, predicted, treat_arrays: bool = True,
                    **treat_arrays_kws) -> float:
    """ `Euclidian distance <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html>`_
    taken from `this book <https://doi.org/10.1016/B978-0-12-088735-4.50006-7`_.

    .. math::
        D = \\sqrt{\\sum_{i=1}^{n} (\\text{true}_i - \\text{predicted}_i)^2}
    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import euclid_distance
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> euclid_distance(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.linalg.norm(true - predicted))


def exp_var_score(true, predicted, treat_arrays: bool = True, weights=None,
                  **treat_arrays_kws) -> Union[float, None]:
    """
    Explained variance `score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html>`_ .
    Best value is 1, lower values are less accurate.

    .. math::
        \\text{EVS} = 1 - \\frac{\\sum_{i=1}^{n} w_i \\left( (true_i - predicted_i) - \\frac{\\sum_{j=1}^{n} w_j (true_j - predicted_j)}{\\sum_{j=1}^{n} w_j} \\right)^2}{\\sum_{i=1}^{n} w_i (true_i - \\frac{\\sum_{j=1}^{n} w_j true_j}{\\sum_{j=1}^{n} w_j})^2}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    weights:

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import exp_var_score
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> exp_var_score(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    y_diff_avg = np.average(true - predicted, weights=weights, axis=0)
    numerator = np.average((true - predicted - y_diff_avg) ** 2,
                           weights=weights, axis=0)

    y_true_avg = np.average(true, weights=weights, axis=0)
    denominator = np.average((true - y_true_avg) ** 2,
                             weights=weights, axis=0)

    if numerator == 0.0:
        return None
    output_scores = _foo(denominator, numerator)

    return float(np.average(output_scores, weights=weights))


def expanded_uncertainty(true, predicted, treat_arrays: bool = True, cov_fact=1.96,
                         **treat_arrays_kws) -> float:
    """By default, it calculates uncertainty with 95% confidence interval.
    1.96 is the coverage factor corresponding 95% confidence level .This
    indicator is used in order to show more information about the model
    deviation. Using formula from by Behar_ et al., 2015 and Gueymard_ et al., 2014.

    .. math::
        U = \\text{cov_fact} \\times \\sqrt{\\frac{1}{n-1} \\sum_{i=1}^{n} \\left( \\left(\\text{true}_i - \\text{predicted}_i\\right) - \\overline{\\left(\\text{true} - \\text{predicted}\\right)} \\right)^2 + \\frac{1}{n} \\sum_{i=1}^{n} \\left(\\text{true}_i - \\text{predicted}_i\\right)^2}

    .. _Behar:
        https://doi.org/10.1016/j.enconman.2015.03.067

    .. _Gueymard:
        https://doi.org/10.1016/j.rser.2014.07.117
    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    cov_fact:

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import expanded_uncertainty
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> expanded_uncertainty(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    error = true - predicted
    sd = np.std(error)
    return float(cov_fact * np.sqrt(sd ** 2 + rmse(true, predicted, treat_arrays=False) ** 2))


def fdc_fhv(true, predicted, treat_arrays: bool = True, h: float = 0.02,
            **treat_arrays_kws) -> float:
    """
    modified `Kratzert2018 <https://github.com/kratzert/ealstm_regional_modeling/blob/64a446e9012ecd601e0a9680246d3bbf3f002f6d/papercode/metrics.py#L190>`_
    code. Peak flow bias of the flow duration curve `(Yilmaz 2008) <doi:10.1029/2007WR006716>`_
    used in `kratzert et al., 2019 <https://hess.copernicus.org/articles/23/5089/2019/hess-23-5089-2019.html>`_.

    .. math::
         FHV = \\frac{\\sum_{i=1}^{k} (predicted_i - true_i)}{\\sum_{i=1}^{k} true_i} \\times 100

    Parameters
    ----------
    h : float
        Must be between 0 and 1.
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Returns
    -------
        Bias of the peak flows

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import fdc_fhv
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> fdc_fhv(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)

    if (h <= 0) or (h >= 1):
        raise RuntimeError("h has to be in the range (0,1)")

    # sort both in descending order
    true = -np.sort(-true)
    predicted = -np.sort(-predicted)

    # subset data to only top h flow values
    true = true[:np.round(h * len(true)).astype(int)]
    predicted = predicted[:np.round(h * len(predicted)).astype(int)]

    fhv = np.sum(predicted - true) / (np.sum(true))

    return float(fhv * 100)


def fdc_flv(true, predicted, treat_arrays: bool = True, low_flow: float = 0.3,
            **treat_arrays_kws) -> float:
    """
    bias of the bottom 30 % low flows. modified Kratzert_ code
    used in `kratzert et al., 2019 <https://hess.copernicus.org/articles/23/5089/2019/hess-23-5089-2019.html>`_.

    .. math::
        \\text{FLV} = -1 \\times \\frac{\\sum (\\log(\\text{predicted}) - \\min(\\log(\\text{predicted}))) - \\sum (\\log(\\text{true}) - \\min(\\log(\\text{true})))}{\\sum (\\log(\\text{true}) - \\min(\\log(\\text{true}))) + 1 \\times 10^{-6}}

    Parameters
    ----------
    low_flow : float, optional
        Upper limit of the flow duration curve. E.g. 0.3 means the bottom 30% of the flows are
        considered as low flows, by default 0.3
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Returns
    -------
        float

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import fdc_flv
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> fdc_flv(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)

    low_flow = 1.0 - low_flow
    # make sure that metric is calculated over the same dimension
    obs = true.flatten()
    sim = predicted.flatten()

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
    obs = np.log(obs)
    sim = np.log(sim)

    # calculate flv part by part
    qsl = np.sum(sim - sim.min())
    qol = np.sum(obs - obs.min())

    flv = -1 * (qsl - qol) / (qol + 1e-6)

    return float(flv * 100)


def gmean_diff(true, predicted, treat_arrays: bool = True,
               **treat_arrays_kws) -> float:
    """
    `Geometric mean difference <https://www.sciencedirect.com/science/article/abs/pii/S0022316624002281>`_.
    First geometric mean is calculated for true and
    predicted arrays and their difference is calculated.

    .. math::
        \\text{gmean_diff} = \\left( \\prod_{i=1}^{n} \\text{true}_i \\right)^{\\frac{1}{n}} - \\left( \\prod_{i=1}^{n} \\text{predicted}_i \\right)^{\\frac{1}{n}}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import gmean_diff
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> gmean_diff(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(_geometric_mean(true) - _geometric_mean(predicted))


def gmrae(true, predicted, treat_arrays: bool = True, benchmark: np.ndarray = None,
          **treat_arrays_kws) -> float:
    """ Geometric Mean Relative Absolute Error

    .. math::
        GMRAE = \\left( \\prod_{i=1}^{n} \\frac{|true_i - predicted_i|}{|true_i - benchmark_i|} \\right)^{\\frac{1}{n}}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    benchmark:

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import gmrae
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> gmrae(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return _geometric_mean(np.abs(_relative_error(true, predicted, benchmark)))


def _relative_error(true, predicted, benchmark: np.ndarray = None):
    """
    Relative Error

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    """

    error = true - predicted
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        if not isinstance(benchmark, int):
            seasonality = 1
        else:
            seasonality = benchmark
        return (true[seasonality:] - predicted[seasonality:]) / \
            (true[seasonality:] - _naive_prognose(true, seasonality) + EPS)

    return error / (true - benchmark + EPS)


def _naive_prognose(true, seasonality: int = 1):
    """ Naive forecasting method which just repeats previous samples
    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
"""
    return true[:-seasonality]


def _minimal() -> list:
    """some minimal and basic metrics"""

    return ['r2', 'mape', 'nrmse', 'corr_coeff', 'rmse', 'mae', 'mse', 'mpe',
            'mase', 'r2_score']


def _hydro_metrics() -> list:
    """Names of metrics related to hydrology"""

    return _minimal() + [
        'fdc_flv', 'fdc_fhv',
        'kge', 'kge_np', 'kge_mod', 'kge_bound', 'kgeprime_bound', 'kgenp_bound',
        'nse', 'nse_alpha', 'nse_beta', 'nse_mod', 'nse_bound']


def calculate_hydro_metrics(true, predicted, treat_arrays: bool = True,
                            **treat_arrays_kws) -> dict:
    """
    Calculates the following performance metrics related to hydrology.
        - fdc_flv
        - fdc_fhv
        - kge
        - kge_np
        - kge_mod
        - kge_bound
        - kgeprime_bound
        - kgenp_bound
        - nse
        - nse_alpha
        - nse_beta
        - nse_mod
        - nse_bound
        - r2
        - mape
        - nrmse
        - corr_coeff
        - rmse
        - mae
        - mse
        - mpe
        - mase
        - r2_score

    Returns
    -------
    dict
        Dictionary with all metrics

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import calculate_hydro_metrics
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> calculate_hydro_metrics(t, p)
    """

    metrics = {}
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)

    for metric in [fdc_flv, fdc_fhv, kge, kge_np, kge_mod, kge_bound, kgeprime_bound, kgenp_bound,
                   nse, nse_alpha, nse_beta, nse_mod, nse_bound, r2, mape, nrmse, corr_coeff, rmse, mae, mse, mpe,
                   mase, r2_score]:
        metrics[metric.__name__] = metric(true, predicted, treat_arrays=False)

    return metrics


def JS(true, predicted, treat_arrays: bool = True,
       **treat_arrays_kws) -> float:
    """`Jensen-shannon divergence <https://datascientest.com/en/jensen-shannon-divergence-everything-you-need-to-know-about-this-ml-model#:~:text=Jensen%2DShannon%20divergence%20and%20Data,and%20expected%20or%20reference%20distributions>`_

    .. math::
        JS(P \parallel Q) = \\frac{1}{2} \\sum_{i} \\left( P(i) \\log_2 \\left( \\frac{2P(i)}{P(i) + Q(i)} \\right) + Q(i) \\log_2 \\left( \\frac{2Q(i)}{P(i) + Q(i)} \\right) \\right)

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import JS
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> JS(t, p)
    """
    if treat_arrays:
        true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    d1 = true * np.log2(2 * true / (true + predicted))
    d2 = predicted * np.log2(2 * predicted / (true + predicted))
    d1[np.isnan(d1)] = 0
    d2[np.isnan(d2)] = 0
    d = 0.5 * sum(d1 + d2)
    return float(d)


def kendall_tau(true, predicted, treat_arrays: bool = True, return_p=False,
                 **treat_arrays_kws) -> Union[float, tuple]:
    """
    Kendall's tau_ .used in Probst_ et al., 2019.
    It is a non-parameteric estimate of correlation between true and predicted arrays.
    It does not assume linearity of the relationship between true and predicted values.
    It compares the ranks of the values in the two arrays to estimate strength
    and direction of association between them. It ranges between -1 to 1 with 1 indicating
    strong association and -1 indicated strong disassociation.

    .. math::
        tau = \\frac{(C - D)}{\\sqrt{(C + D + T_{\\text{true}})(C + D + T_{\\text{predicted}})}}

    10.1088/1742-5468/aace08
    .. _tau:
        https://machinelearningmastery.com/how-to-calculate-nonparametric-rank-correlation-in-python/

    .. _Probst:
        https://www.jmlr.org/papers/volume20/18-444/18-444.pdf

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    return_p:

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import kendall_tau
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> kendall_tau(t, p)
    """
    if kendalltau is None:
        raise NotImplementedError("kendalltau function is not available. Please install scipy")

    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    coef, p = kendalltau(true, predicted)
    if return_p:
        return coef, p
    return float(coef)


def kgeprime_bound(true, predicted, treat_arrays: bool = True,
                 **treat_arrays_kws) -> float:
    """
    `Bounded Version of the Modified Kling-Gupta Efficiency <https://iahs.info/uploads/dms/13614.21--211-219-41-MATHEVET.pdf>`_

    .. math::
        KGE'_{\\text{bounded}} = \\frac{1 - \\sqrt{(r - 1)^2 + (\\gamma - 1)^2 + (\\beta - 1)^2}}{2 - (1 - \\sqrt{(r - 1)^2 + (\\gamma - 1)^2 + (\\beta - 1)^2})}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import kgeprime_bound
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> kgeprime_bound(t, p)

    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    kgeprime_ = kge_mod(true, predicted, return_all=True, treat_arrays=False)[0, :]
    kgeprime_c2m_ = kgeprime_ / (2 - kgeprime_)

    return float(kgeprime_c2m_)


def kgenp_bound(true, predicted, treat_arrays: bool = True,
                **treat_arrays_kws):
    """
    Bounded Version of the Non-Parametric Kling-Gupta Efficiency

    .. math::
        KGE_{np_{bound}} = \\frac{1 - \\sqrt{\\left(\\rho(t, p) - 1\\right)^2 + \\left(1 - 0.5 \\sum_{i=1}^{n} \\left| \\frac{\\text{sorted}(p_i)}{\\text{mean}(p) \\cdot n} - \\frac{\\text{sorted}(t_i)}{\\text{mean}(t) \\cdot n} \\right| - 1\\right)^2 + \\left(\\frac{\\text{mean}(p)}{\\text{mean}(t)} - 1\\right)^2}}{2 - \\left(1 - \\sqrt{\\left(\\rho(t, p) - 1\\right)^2 + \\left(1 - 0.5 \\sum_{i=1}^{n} \\left| \\frac{\\text{sorted}(p_i)}{\\text{mean}(p) \\cdot n} - \\frac{\\text{sorted}(t_i)}{\\text{mean}(t) \\cdot n} \\right| - 1\\right)^2 + \\left(\\frac{\\text{mean}(p)}{\\text{mean}(t)} - 1\\right)^2}\\right)}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import kgenp_bound
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> kgenp_bound(t, p)

    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    kgenp_ = kge_np(return_all=True, true=true, predicted=predicted, treat_arrays=False)[0, :]
    kgenp_c2m_ = kgenp_ / (2 - kgenp_)

    return float(kgenp_c2m_)


def kl_sym(true, predicted, treat_arrays: bool = True,
           **treat_arrays_kws) -> Union[float, None]:
    """
    `Symmetric kullback-leibler divergence <https://doi.org/10.1016/j.procs.2018.10.144>`_

    .. math::
        \\text{KL}_{\\text{sym}}(P || Q) = \\frac{1}{2} \\sum_{i=1}^{n} \\left( P_i - Q_i \\right) \\left( \\log_2 \\frac{P_i}{Q_i} \\right)

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import kl_sym
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> kl_sym(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)

    if not all((true == 0) == (predicted == 0)):
        return None  # ('KL divergence not defined when only one distribution is 0.')
    x, y = true, predicted
    # set values where both distributions are 0 to the same (positive) value.
    # This will not contribute to the final distance.
    x[x == 0] = 1
    y[y == 0] = 1
    d = 0.5 * np.sum((x - y) * (np.log2(x) - np.log2(y)))
    return float(d)


def lm_index(true, predicted, treat_arrays: bool = True, obs_bar_p=None,
             **treat_arrays_kws) -> float:
    """
    `Legate-McCabe Efficiency Index <https://doi.org/10.1016/j.cmpb.2023.107737>`_.
    Less sensitive to outliers in the data. The larger, the better

    .. math::
        a_i = |predicted_i - true_i|

    .. math::
        b_i = |true_i - \\text{obs\\_bar\\_p}| \\text{if } \\text{obs\\_bar\\_p} \\text{ is provided} \\|true_i - \\bar{true}| \\text{otherwise}

    .. math::
        \\text{LM Index} = 1 - \\frac{\\sum_{i=1}^{n} a_i}{\\sum_{i=1}^{n} b_i}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    obs_bar_p : float,
        Seasonal or other selected average. If None, the mean of the
        observed array will be used.

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import lm_index
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> lm_index(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    mean_obs = np.mean(true)
    a = np.abs(predicted - true)

    if obs_bar_p is not None:

        b = np.abs(true - obs_bar_p)
    else:
        b = np.abs(true - mean_obs)

    return float(1 - (np.sum(a) / np.sum(b)))


def maape(true, predicted, treat_arrays: bool = True,
          **treat_arrays_kws) -> float:
    """
    `Mean Arctangent Absolute Percentage Error <https://doi.org/10.1016/j.ijforecast.2015.12.003>`_
    Note: result is NOT multiplied by 100

    .. math::
        MAAPE = \\frac{1}{n} \\sum_{i=1}^{n} \\arctan \\left( \\frac{| \\text{true}_i - \\text{predicted}_i |}{| \\text{true}_i | + \\epsilon} \\right)

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import maape
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> maape(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.mean(np.arctan(np.abs((true - predicted) / (true + EPS)))))


def _percentage_error(true, predicted):
    """
    Percentage error. The value is multiplied by 100 to reflect percentage.

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    """

    error = true - predicted
    return error / (true + EPS) * 100


def _bounded_relative_error(
        true,
        predicted,
        benchmark: np.ndarray = None):
    """ Bounded Relative Error

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    """

    error = true - predicted
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        if not isinstance(benchmark, int):
            seasonality = 1
        else:
            seasonality = benchmark

        abs_err = np.abs(true[seasonality:] - predicted[seasonality:])
        abs_err_bench = np.abs(true[seasonality:] - _naive_prognose(true, seasonality))
    else:
        abs_err = np.abs(error)
        abs_err_bench = np.abs(error)

    return abs_err / (abs_err + abs_err_bench + EPS)


def mbrae(true, predicted, treat_arrays: bool = True, benchmark: np.ndarray = None,
          **treat_arrays_kws) -> float:
    """ `Mean Bounded Relative Absolute Error <https://doi.org/10.1371/journal.pone.0174202>`_

    .. math::
        MBRAE = \\frac{1}{n} \\sum_{i=1}^{n} \\frac{| \\text{true}_i - \\text{predicted}_i |}{| \\text{true}_i - \\text{benchmark}_i |}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    benchmark:

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import mbrae
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> mbrae(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.mean(_bounded_relative_error(true, predicted, benchmark=benchmark)))


def mapd(true, predicted, treat_arrays: bool = True, #ToDo equation not multiplied by 100
         **treat_arrays_kws) -> float:
    """ `Mean absolute percentage deviation <https://doi.org/10.1016/j.rinma.2022.100347>`_

    .. math::
        MAPD = \\frac{\\sum_{i=1}^{n} \\left| predicted_i - true_i \\right|}{\\sum_{i=1}^{n} \\left| true_i \\right|}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import mapd
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> mapd(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    a = np.sum(np.abs(predicted - true))
    b = np.sum(np.abs(true))
    return float(a / b)


def _ae(true, predicted):
    """Absolute error
    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    """

    return np.abs(true - predicted)


def max_error(true, predicted, treat_arrays: bool = True,
              **treat_arrays_kws) -> float:
    """
    `maximum absolute error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.max_error.html>`_
    In Sklearn, there is "absolute" in equation but not in name of metric.

    .. math::
        \\text{Max Error} = \\max_{i=1}^n \\left| \\text{true}_i - \\text{predicted}_i \\right|

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import max_error
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> max_error(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.max(_ae(true, predicted)))


def mb_r(true, predicted, treat_arrays: bool = True,
         **treat_arrays_kws) -> float:
    """
    `Mielke-Berry R value <https://psycnet.apa.org/buy/1988-15790-001>`_.

    .. math::
        R = 1 - \\frac{n^2 \\cdot \\frac{1}{n} \\sum_{i=1}^{n} \\left| \\text{predicted}_i - \\text{true}_i \\right|}{\\sum_{i=1}^{n} \\sum_{j=1}^{n} \\left| \\text{predicted}_j - \\text{true}_i \\right|}

    References
    ----------
    `Mielke, P. W., & Berry, K. J. (2007). Permutation methods: a distance function approach. Springer Science & Business Media <https://link.springer.com/book/10.1007/978-1-4757-3449-2>`_
    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import mb_r
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> mb_r(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    # Calculate metric
    n = predicted.size
    tot = 0.0
    for i in range(n):
        tot = tot + np.sum(np.abs(predicted - true[i]))
    mae_val = np.sum(np.abs(predicted - true)) / n
    mb = 1 - ((n ** 2) * mae_val / tot)

    return float(mb)


def mda(
        true,
        predicted,
        treat_arrays: bool = True,
        **treat_arrays_kws) -> float:
    """Mean Directional Accuracy

    .. math::
        \\text{MDA} = \\frac{1}{n-1} \\sum_{i=1}^{n-1} \\left( \\text{sign}( \\text{true}_{i+1} - \\text{true}_i) == \\text{sign}( \\text{predicted}_{i+1} - \\text{predicted}_i) \\right)

    modified `after <https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9>`_.

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import mda
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> mda(t, p)
     """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    dict_acc = np.sign(true[1:] - true[:-1]) == np.sign(predicted[1:] - true[:-1])
    return float(np.mean(dict_acc))


def mde(true, predicted, treat_arrays: bool = True,
        **treat_arrays_kws) -> float:
    """
    `Median Error <https://doi.org/10.1016/j.cma.2024.116842>`_

    .. math::
        MDE = \\text{median}(\\text{predicted}_i - \\text{true}_i)

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import mde
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> mde(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.median(predicted - true))


def mdape(true, predicted, treat_arrays: bool = True,
          **treat_arrays_kws) -> float:
    """
    `Median Absolute Percentage Error <https://doi.org/10.1016/j.petrol.2021.109265>`_. The value is multiplied by 100.

    .. math::
        \\text{MdAPE} = 100 \\times \\text{Median} \\left( \\left\\{ \\frac{|\\text{true}_i - \\text{predicted}_i|}{|\\text{true}_i|} \\right\\}_{i=1}^n \\right)

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import mdape
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> mdape(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.median(np.abs(_percentage_error(true, predicted))))


def mdrae(true, predicted, treat_arrays: bool = True, benchmark: np.ndarray = None,
          **treat_arrays_kws) -> float:
    """
    `Median Relative Absolute Error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html>`_
    In Sklearn, there is "absolute" in equation but not in name of metric.

    .. math::
        MdRAE = \\text{median} \\left( \\left| \\frac{true_i - predicted_i}{true_i - benchmark_i} \\right| \\right)

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    benchmark:

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import mdrae
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> mdrae(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.median(np.abs(_relative_error(true, predicted, benchmark))))


def me(true, predicted, treat_arrays: bool = True,
       **treat_arrays_kws):
    """ `Mean error <https://doi.org/10.1016/j.scitotenv.2024.174533>`_

    .. math::
        ME = \\frac{1}{n} \\sum_{i=1}^{n} (\\text{true}_i - \\text{predicted}_i)

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import me
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> me(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    error = true - predicted
    return float(np.mean(error))


def mean_bias_error(true, predicted, treat_arrays: bool = True,
                    **treat_arrays_kws) -> float:
    """
    Mean Bias Error
    It represents overall bias error or systematic error. It shows average interpolation bias; i.e. average over-
    or underestimation. [1][2].This indicator expresses a tendency of model to underestimate (negative value)
    or overestimate (positive value) global radiation, while the MBE values closest to zero are desirable.
    The drawback of this test is that it does not show the correct performance when the model presents
    overestimated and underestimated values at the same time, since overestimation and underestimation
    values cancel each `other <https://doi.org/10.1016/j.rser.2015.08.035>`_.

    .. math::
        \\text{MBE} = \\frac{1}{N} \\sum_{i=1}^{N} (true_i - predicted_i)

    References
    ----------

    - `Willmott, C. J., & Matsuura, K. (2006). On the use of dimensioned measures of error to evaluate the performance
        of spatial interpolators. International Journal of Geographical Information Science, 20(1), 89-102.
        <https://doi.org/10.1080/1365881050028697>`_

    - `Valipour, M. (2015). Retracted: Comparative Evaluation of Radiation-Based Methods for Estimation of Potential
        Evapotranspiration. Journal of Hydrologic Engineering, 20(5), 04014068.
        <https://dx.doi.org/10.1061/(ASCE)HE.1943-5584.0001066>`_

    -  `<https://doi.org/10.1016/j.rser.2015.08.035>`_

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import mean_bias_error
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> mean_bias_error(t, p)
     """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.sum(true - predicted) / len(true))


def mean_var(true, predicted, treat_arrays: bool = True,
             **treat_arrays_kws) -> float:
    """Mean variance

    .. math::
        \\text{mean_var} = \\text{Var} \\left( \\log(1 + \\text{true}) - \\log(1 + \\text{predicted}) \\right)

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import mean_var
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> mean_var(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.var(log1p(true) - log1p(predicted)))


def mean_poisson_deviance(true, predicted, treat_arrays: bool = True, weights=None,
                          **treat_arrays_kws) -> float:
    """
    mean poisson deviance

    .. math::
        \\text{MPD} = \\frac{1}{n} \\sum_{i=1}^{n} 2 \\left( \\text{true}_i \\log \\left( \\frac{\\text{true}_i}{\\text{predicted}_i} \\right) - (\\text{true}_i - \\text{predicted}_i) \\right)

    References
    ---------
    `<https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_poisson_deviance.html>`_

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    weights:

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import mean_poisson_deviance
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> mean_poisson_deviance(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression',
                                       **treat_arrays_kws)
    return _mean_tweedie_deviance(true, predicted, weights=weights, power=1)


def mean_gamma_deviance(true, predicted, treat_arrays: bool = True, weights=None,
                        **treat_arrays_kws) -> float:
    """
    `mean gamma deviance <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_gamma_deviance.html>`_

    .. math::
        \\text{Mean Gamma Deviance (Weighted)} = \\frac{1}{\\sum_{i=1}^{n} w_i} \\sum_{i=1}^{n} w_i \\frac{2}{\\text{true}_i} \\left( \\text{predicted}_i - \\text{true}_i - \\text{true}_i \\ln \\left( \\frac{\\text{predicted}_i}{\\text{true}_i} \\right) \\right)

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    weights:

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import mean_gamma_deviance
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> mean_gamma_deviance(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return _mean_tweedie_deviance(true, predicted, weights=weights, power=2)


def median_abs_error(true, predicted, treat_arrays: bool = True,
                     **treat_arrays_kws) -> float:
    """
    median absolute error

    .. math::
        \\text{MedAE} = \\text{median} \\left( \\left| \\text{true}_i - \\text{predicted}_i \\right| \\right)

    References
    ----------
    `<https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html>`_

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import median_abs_error
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> median_abs_error(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.median(np.abs(predicted - true), axis=0))


def med_seq_error(true, predicted, treat_arrays: bool = True,
                  **treat_arrays_kws) -> float:
    """ `Median Squared Error <https://www.sciencedirect.com/science/article/pii/S2468227620301757>`_
    Same as mse, but it takes median which reduces the impact of outliers.

    .. math::
        \\text{MedSE} = \\text{median} \\left( (\\text{predicted}_i - \\text{true}_i)^2 \\right)

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import med_seq_error
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> med_seq_error(t, p)
    """
    if treat_arrays:
        true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.median((predicted - true) ** 2))


def mle(true, predicted, treat_arrays=True,
        **treat_arrays_kws) -> float:
    """ `Mean log error <https://doi.org/10.1038/s41598-023-29871-8>`_

    .. math::
        \\text{MLE} = \\frac{1}{n} \\sum_{i=1}^{n} \\left( \\log(1 + \\text{predicted}_i) - \\log(1 + \\text{true}_i) \\right)

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import mle
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> mle(t, p)
    """
    if treat_arrays:
        true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.mean(log1p(predicted) - log1p(true)))


def mod_agreement_index(true, predicted, treat_arrays: bool = True, j=1,
                        **treat_arrays_kws) -> float:
    """Modified agreement of index.
    j: int, when j==1, this is same as agreement_index. Higher j means more impact of outliers.

    .. math::
        MAI = 1 - \\frac{\\sum_{i=1}^{n} \\left| \\text{predicted}_i - \\text{true}_i \\right|^j}{\\sum_{i=1}^{n} \\left( \\left| \\text{predicted}_i - \\overline{\\text{true}} \\right| + \\left| \\text{true}_i - \\overline{\\text{true}} \\right| \\right)^j}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    j:

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import mod_agreement_index
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> mod_agreement_index(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    a = (np.abs(predicted - true)) ** j
    b = np.abs(predicted - np.mean(true))
    c = np.abs(true - np.mean(true))
    e = (b + c) ** j
    return float(1 - (np.sum(a) / np.sum(e)))


def mpe(
        true, 
        predicted, 
        treat_arrays: bool = True,
        **treat_arrays_kws
        ) -> float:
    """
    `Mean Percentage Error <https://doi.org/10.1016/j.molliq.2023.123378>`_.
    The value is multiplied by 100 to reflect percentage.

    .. math::
        MPE = \\frac{1}{n} \\sum_{i=1}^{n} \\left( \\frac{true_i - predicted_i}{true_i} \\right) \\times 100

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import mpe
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> mpe(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.mean(_percentage_error(true, predicted)))


def mrae(true, predicted, treat_arrays: bool = True, benchmark: np.ndarray = None,
         **treat_arrays_kws):
    """ `Mean Relative Absolute Error <https://doi.org/10.1016/j.comnet.2024.110237>`_

    .. math::
        MRAE = \\frac{1}{n} \\sum_{i=1}^{n} \\left| \\frac{\\text{true}_i - \\text{predicted}_i}{\\text{benchmark}_i} \\right|

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    benchmark:

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import mrae
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> mrae(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.mean(np.abs(_relative_error(true, predicted, benchmark))))


def norm_euclid_distance(true, predicted, treat_arrays: bool = True,
                         **treat_arrays_kws) -> float:
    """ `Normalized Euclidian distance <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html>`_

    .. math::
        D_{norm} = \\sqrt{\\sum_{i=1}^{n} \\left( \\frac{\\text{true}_i}{\\bar{\\text{true}}} - \\frac{\\text{predicted}_i}{\\bar{\\text{predicted}}} \\right)^2}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import norm_euclid_distance
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> norm_euclid_distance(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)

    a = true / np.mean(true)
    b = predicted / np.mean(predicted)
    return float(np.linalg.norm(a - b))


def nrmse_range(true, predicted, treat_arrays: bool = True,
                **treat_arrays_kws) -> float:
    """Range Normalized Root Mean Squared Error.
    RMSE normalized by true values. This allows comparison between data sets
    with different scales. It is more sensitive to outliers.

    Reference: `Pontius et al., 2008 <https://link.springer.com/article/10.1007/s10651-007-0043-y>`_

    .. math::
        \\text{NRMSE} = \\frac{\\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (\\text{predicted}_i - \\text{true}_i)^2}}{\\max(\\text{true}) - \\min(\\text{true})}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import nrmse_range
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> nrmse_range(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)

    return float(rmse(true, predicted, treat_arrays=False) / (np.max(true) - np.min(true)))


def nrmse_ipercentile(
        true, 
        predicted, 
        treat_arrays: bool = True, 
        q1=25, 
        q2=75,
        **treat_arrays_kws
        ) -> float:
    """
    RMSE normalized by inter percentile range of true. This is the least sensitive to outliers.
    q1: any interger between 1 and 99
    q2: any integer between 2 and 100. Should be greater than q1.
    Reference: `Pontius et al., 2008. <https://link.springer.com/article/10.1007/s10651-007-0043-y>`_

    .. math::
        \\text{NRMSE}_{\\text{IP}} = \\frac{\\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (\\text{true}_i - \\text{predicted}_i)^2}}{Q_{q2} - Q_{q1}}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    q1:
    q2:

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import nrmse_ipercentile
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> nrmse_ipercentile(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)

    q1 = np.percentile(true, q1)
    q3 = np.percentile(true, q2)
    iqr = q3 - q1

    return float(rmse(true, predicted, treat_arrays=False) / iqr)


def nrmse_mean(true, predicted, treat_arrays: bool = True,
               **treat_arrays_kws) -> float:
    """Mean Normalized RMSE

    RMSE normalized by mean of true values.This allows comparison between datasets with different scales.

    .. math::
        NRMSE_{mean} = \\frac{\\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (\\text{true}_i - \\text{predicted}_i)^2}}{\\bar{\\text{true}}}

    Reference: `Pontius et al., 2008 <https://link.springer.com/article/10.1007/s10651-007-0043-y>`_
    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import nrmse_mean
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> nrmse_mean(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(rmse(true, predicted, treat_arrays=False) / np.mean(true))


def norm_ae(true, predicted, treat_arrays: bool = True,
            **treat_arrays_kws) -> float:
    """ `Normalized Absolute Error <https://doi.org/10.1016/j.apor.2024.104042>`_

    .. math::
        norm\\_ae = \\sqrt{\\frac{\\sum_{i=1}^{n} (error_i - MAE)^2}{n - 1}}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import norm_ae
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> norm_ae(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    error = true - predicted
    return float(np.sqrt(np.sum(np.square(error - mae(true, predicted, False))) / (len(true) - 1)))


def log_prob(true, predicted, treat_arrays: bool = True,
             **treat_arrays_kws) -> float:
    """
    Logarithmic probability distribution

    .. math::
        \\text{log_prob} = \\frac{1}{N} \\sum_{i=1}^{N} \\left( -\\frac{\\left( \\frac{\\text{true}_i - \\text{predicted}_i}{\\text{scale}} \\right)^2}{2} - \\log(\\sqrt{2\\pi}) \\right)

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import log_prob
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> log_prob(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    scale = np.mean(true) / 10
    if scale < .01:
        scale = .01
    y = (true - predicted) / scale
    normpdf = -y ** 2 / 2 - np.log(np.sqrt(2 * np.pi))
    return float(np.mean(normpdf))


def rmdspe(true, predicted, treat_arrays: bool = True,
           **treat_arrays_kws) -> float:
    """
    `Root Median Squared Percentage Error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html>`_.
    The value is multiplied by 100 to reflect percentage.

    .. math::
        \\text{RMDSPE} = \\sqrt{\\text{median}\\left(\\left(\\frac{\\text{true}_i - \\text{predicted}_i}{\\text{true}_i} \\times 100\\right)^2\\right)}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import rmdspe
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> rmdspe(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.sqrt(np.median(np.square(_percentage_error(true, predicted)))))


def rse(
        true, 
        predicted, 
        treat_arrays: bool = True,
        **treat_arrays_kws
        ) -> float:
    """Relative Squared Error

    .. math::
        \\text{RSE} = \\frac{\\sum_{i=1}^{n} (\\text{true}_i - \\text{predicted}_i)^2}{\\sum_{i=1}^{n} (\\text{true}_i - \\bar{\\text{true}})^2}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import rse
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> rse(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.sum(np.square(true - predicted)) / np.sum(np.square(true - np.mean(true))))


def rrse(true, predicted, treat_arrays: bool = True,
         **treat_arrays_kws) -> float:
    """ `Root Relative Squared Error <https://www.sciencedirect.com/science/article/pii/S0360319923031798>`_

    .. math::
        RRSE = \\sqrt{\\frac{\\sum_{i=1}^{n} (\\text{true}_i - \\text{predicted}_i)^2}{\\sum_{i=1}^{n} (\\text{true}_i - \\bar{\\text{true}})^2}}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import rrse
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> rrse(t, p)"""
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.sqrt(rse(true, predicted, treat_arrays=False)))


def rae(true, predicted, treat_arrays: bool = True,
        **treat_arrays_kws) -> float:
    """ `Relative Absolute Error <https://doi.org/10.1016/j.compbiomed.2017.02.010>`_ (aka Approximation Error)

    .. math::
        \\text{RAE} = \\frac{\\sum_{i=1}^{n} \\left| \\text{true}_i - \\text{predicted}_i \\right|}{\\sum_{i=1}^{n} \\left| \\text{true}_i - \\overline{\\text{true}} \\right|}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import rae
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> rae(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.sum(_ae(true, predicted)) / (np.sum(np.abs(true - np.mean(true))) + EPS))


def ref_agreement_index(true, predicted, treat_arrays: bool = True,
                        **treat_arrays_kws) -> float:
    """Refined Index of Agreement. From -1 to 1. Larger the better.

    .. math::
        a = \\sum_{i=1}^{n} \\left| \\text{predicted}_i - \\text{true}_i \\right|

    .. math::
        b = 2 \\sum_{i=1}^{n} \\left| \\text{true}_i - \\overline{\\text{true}} \\right|

    .. math::
        d_{\\text{ref}} =
        \\begin{cases}
        1 - \\frac{a}{b} & \\text{if } a \\leq b \\
        \\frac{b}{a} - 1 & \\text{if } a > b
        \\end{cases}

    Refrence: `Willmott et al., 2012 <https://www.researchgate.net/profile/Scott-Robeson/publication/235961403_A_refined_index_of_model_performance/links/5a6114254585158bca49f8e4/A-refined-index-of-model-performance.pdf>`_
    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import ref_agreement_index
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> ref_agreement_index(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    a = np.sum(np.abs(predicted - true))
    b = 2 * np.sum(np.abs(true - true.mean()))
    if a <= b:
        return float(1 - (a / b))
    else:
        return float((b / a) - 1)


def rel_agreement_index(true, predicted, treat_arrays: bool = True,
                        **treat_arrays_kws) -> float:
    """ `Relative index of agreement <https://doi.org/10.1007/s10661-022-10844-9>`_. from 0 to 1. larger the better.

    .. math::
        \\text{rel_agreement_index} = 1 - \\frac{\\sum_{i=1}^{n} \\left( \\frac{\\text{predicted}_i - \\text{true}_i}{\\text{true}_i} \\right)^2}{\\sum_{i=1}^{n} \\left( \\frac{|\\text{predicted}_i - \\bar{\\text{true}}| + |\\text{true}_i - \\bar{\\text{true}}|}{\\bar{\\text{true}}} \\right)^2}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import rel_agreement_index
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> rel_agreement_index(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    a = ((predicted - true) / true) ** 2
    b = np.abs(predicted - np.mean(true))
    c = np.abs(true - np.mean(true))
    e = ((b + c) / np.mean(true)) ** 2
    return float(1 - (np.sum(a) / np.sum(e)))


def relative_rmse(
        true, predicted, treat_arrays: bool = True,
                  **treat_arrays_kws) -> float:
    """
    `Relative Root Mean Squared Error <https://search.r-project.org/CRAN/refmans/metrica/html/RRMSE.html>`_. It normalizes teh rmse by mean of true values.

    .. math::
        RRMSE=\\frac{\\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})^2}}{\\bar{e}}



    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import relative_rmse
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> relative_rmse(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    rrmse = rmse(true, predicted, treat_arrays=False) / np.mean(true)
    return float(rrmse)


def rmspe(true, predicted, treat_arrays: bool = True,
          **treat_arrays_kws) -> float:
    """
    `Root Mean Square Percentage Error <https://stackoverflow.com/a/53166790/5982232>`_.

    .. math::
        RMSPE = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} \\left(PE_i\\right)^2} = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} \\left(\\frac{\\text{true}_i - \\text{predicted}_i}{\\text{true}_i}\\right)^2}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import rmspe
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> rmspe(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.sqrt(np.mean(np.square(_percentage_error(true, predicted)), axis=0)))


def rsr(
        true, 
        predicted, 
        treat_arrays: bool = True,
        **treat_arrays_kws
        ) -> float:
    """
    It is MSE normalized by standard deviation of true values. 
    Following `Moriasi et al., 2007. <https://swat.tamu.edu/media/1312/moriasimodeleval.pdf>`_.

    It incorporates the benefits of error index statistics andincludes a
    scaling/normalization factor, so that the resulting statistic and reported
    values can apply to various constitu-ents. It ranges from 0 to infinity, with
    0-0.5 indicating very good model performance, 0.5-0.8 indicating good model
    performance. 

    Standard deviation is calculated using np.ntd(true, ddof=1) to match the results of `this <https://rdrr.io/cran/hydroGOF/man/rsr.html>`_.

    .. math::
        \\text{RSR} = \\frac{\\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (\\text{true}_i - \\text{predicted}_i)^2}}{\\sqrt{\\frac{1}{n-1} \\sum_{i=1}^{n} (\\text{true}_i - \\bar{\\text{true}})^2}}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import rsr
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> rsr(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(rmse(predicted=predicted, true=true, treat_arrays=False) / np.std(true, ddof=1))


def rmsse(true, predicted, treat_arrays: bool = True, seasonality: int = 1,
          **treat_arrays_kws) -> float:
    """ Root Mean Squared Scaled Error

    .. math::
        \\text{RMSSE} = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} \\left( \\frac{\\left| \\text{true}_i - \\text{predicted}_i \\right|}{\\frac{1}{n-s} \\sum_{j=s+1}^{n} \\left| \\text{true}_j - \\text{true}_{j-s} \\right|} \\right)^2}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    seasonality:

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import rmsse
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> rmsse(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    error = true - predicted
    q = np.abs(error) / mae(true[seasonality:], _naive_prognose(true, seasonality), treat_arrays=False)
    return float(np.sqrt(np.mean(np.square(q))))


def sa(true, predicted, treat_arrays: bool = True,
       **treat_arrays_kws) -> float:
    """Spectral angle. From -pi/2 to pi/2. Closer to 0 is better.
    It measures angle between two vectors in hyperspace indicating
    how well the shape of two arrays match instead of their magnitude.
    Reference: `Robila and Gershman, 2005 <https://ieeexplore.ieee.org/abstract/document/1509878>`_.

    .. math::
        SA = \\arccos \\left( \\frac{\\sum_{i=1}^{n} (\\text{true}_i \\cdot \\text{predicted}_i)}{\\sqrt{\\sum_{i=1}^{n} (\\text{true}_i)^2} \\cdot \\sqrt{\\sum_{i=1}^{n} (\\text{predicted}_i)^2}} \\right)

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import sa
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> sa(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    a = np.dot(predicted, true)
    b = np.linalg.norm(predicted) * np.linalg.norm(true)
    return float(np.arccos(a / b))


def sc(true, predicted, treat_arrays: bool = True,
       **treat_arrays_kws) -> float:
    """Spectral correlation.
    It varies from -pi/2 to pi/2. Closer to 0 is better.

    .. math::
        sc = \\arccos \\left( \\frac{ \\sum_{i=1}^{n} (t_i - \\bar{t}) \\cdot (p_i - \\bar{p}) }{ \\sqrt{\\sum_{i=1}^{n} (t_i - \\bar{t})^2} \\cdot \\sqrt{\\sum_{i=1}^{n} (p_i - \\bar{p})^2} } \\right)

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import sc
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> sc(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    a = np.dot(true - np.mean(true), predicted - np.mean(predicted))
    b = np.linalg.norm(true - np.mean(true))
    c = np.linalg.norm(predicted - np.mean(predicted))
    e = b * c
    return float(np.arccos(a / e))


def sga(true, predicted, treat_arrays: bool = True,
        **treat_arrays_kws) -> float:
    """Spectral gradient angle.
    It varies from -pi/2 to pi/2. Closer to 0 is better.

    .. math::
        \\text{SGA} = \\arccos \\left( \\frac{\\sum_{i=1}^{n-1} \\left( (true_{i+1} - true_i) \\cdot (predicted_{i+1} - predicted_i) \\right)}{\\sqrt{\\sum_{i=1}^{n-1} (true_{i+1} - true_i)^2} \\times \\sqrt{\\sum_{i=1}^{n-1} (predicted_{i+1} - predicted_i)^2}} \\right)

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import sga
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> sga(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    sgx = true[1:] - true[:true.size - 1]
    sgy = predicted[1:] - predicted[:predicted.size - 1]
    a = np.dot(sgx, sgy)
    b = np.linalg.norm(sgx) * np.linalg.norm(sgy)
    return float(np.arccos(a / b))


def smape(true, predicted, treat_arrays: bool = True,
          **treat_arrays_kws) -> float:
    """
    `Symmetric Mean Absolute Percentage Error <https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error>`_.
    Adoption from `this <https://stackoverflow.com/a/51440114/5982232>`_.

    .. math::
        SMAPE = \\frac{100}{n} \\sum_{i=1}^{n} \\frac{2 \\left| \\text{predicted}_i - \\text{true}_i \\right|}{\\left| \\text{true}_i \\right| + \\left| \\text{predicted}_i \\right|}

    Goodwin and Lawton, 1999 : https://doi.org/10.1016/S0169-2070(99)00007-2
    Flores et al., 1986 : https://doi.org/10.1016/0305-0483(86)90013-7
        
    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import smape
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> smape(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    _temp = np.sum(2 * np.abs(predicted - true) / (np.abs(true) + np.abs(predicted)))
    return float(100 / len(true) * _temp)


def smdape(true, predicted, treat_arrays: bool = True,
           **treat_arrays_kws) -> float:
    """
    Symmetric Median Absolute Percentage Error
    Note: result is NOT multiplied by 100

    .. math::
        \\text{smdape} = \\text{median} \\left( \\frac{2 \\cdot | \\text{predicted} - \\text{true} |}{| \\text{true} | + | \\text{predicted} | + \\epsilon} \\right)

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import smdape
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> smdape(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.median(2.0 * _ae(predicted=predicted, true=true) / ((np.abs(true) + np.abs(predicted)) + EPS)))


def sid(true, predicted, treat_arrays: bool = True,
        **treat_arrays_kws) -> float:
    """Spectral Information Divergence.
    From -pi/2 to pi/2. Closer to 0 is better.

    .. math::
        \\text{SID} = \\left( \\frac{\\text{t}}{\\text{mean(t)}} - \\frac{\\text{p}}{\\text{mean(p)}} \\right) \\cdot \\left( \\log_{10}(\\text{t}) - \\log_{10}(\\text{mean(t)}) - \\log_{10}(\\text{p}) + \\log_{10}(\\text{mean(p)}) \\right)

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import sid
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> sid(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    first = (true / np.mean(true)) - (
            predicted / np.mean(predicted))
    second1 = np.log10(true) - np.log10(np.mean(true))
    second2 = np.log10(predicted) - np.log10(np.mean(predicted))
    return float(np.dot(first, second1 - second2))


def skill_score_murphy(true, predicted, treat_arrays: bool = True,
                       **treat_arrays_kws) -> float:
    """
    Adopted from here_ .
    Calculate non-dimensional skill score (SS) between two variables using
    definition of Murphy (1988) using the formula:

    .. math::
        SS = 1 - RMSE^2/SDEV^2

    .. math::
        SDEV is the standard deviation of the true values

    .. math::
        SDEV^2 = sum_(n=1)^N [r_n - mean(r)]^2/(N-1)

    where p is the predicted values, r is the reference values, and N is the total number of values in p & r.
    Note that p & r must have the same number of values. A positive skill score can be interpreted as the percentage
    of improvement of the new model forecast in comparison to the reference. On the other hand, a negative skill
    score denotes that the forecast of interest is worse than the referencing forecast. Consequently, a value of
    zero denotes that both forecasts perform equally [MLAir, 2020].

    Returns:
        flaot

    References
    ---------
        `Allan H. Murphy, 1988: Skill Scores Based on the Mean Square Error
        and Their Relationships to the Correlation Coefficient. Mon. Wea.
        Rev., 116, 2417-2424
        <doi: http//dx.doi.org/10.1175/1520-0493(1988)<2417:SSBOTM>2.0.CO;2>`_.

    .. _here:
        https://github.com/PeterRochford/SkillMetrics/blob/278b2f58c7d73566f25f10c9c16a15dc204f5869/skill_metrics/skill_score_murphy.py

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import skill_score_murphy
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> skill_score_murphy(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    # Calculate RMSE
    rmse2 = rmse(true, predicted, treat_arrays=False) ** 2

    # Calculate standard deviation
    sdev2 = np.std(true, ddof=1) ** 2

    # Calculate skill score
    ss = 1 - rmse2 / sdev2

    return float(ss)


def std_ratio(true,
              predicted,
              treat_arrays: bool = True,
              std_kwargs: dict = None,
              **treat_arrays_kws
              ) -> float:
    """
    `Ratio of standard deviations of predictions and trues <https://doi.org/10.1016/j.engfracmech.2024.110057>`_.
    Also known as standard ratio, it varies from 0.0 to infinity while
    1.0 being the perfect value.

    .. math::
        \\text{std_ratio} = \\frac{\\sigma_{\\text{predicted}}}{\\sigma_{\\text{true}}}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import std_ratio
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> std_ratio(t, p)
    """

    std_kwargs = std_kwargs or {}

    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.std(predicted, **std_kwargs) / np.std(true, **std_kwargs))


def umbrae(
        true, predicted, treat_arrays: bool = True, benchmark: np.ndarray = None,
           **treat_arrays_kws):
    """ `Unscaled Mean Bounded Relative Absolute Error <https://doi.org/10.1016/j.jclepro.2022.135414>`_

    .. math::
        UMBRAE = \\frac{\\frac{1}{n} \\sum_{i=1}^{n} \\frac{|t_i - p_i|}{|t_i - b_i|}}{1 - \\frac{1}{n} \\sum_{i=1}^{n} \\frac{|t_i - p_i|}{|t_i - b_i|}}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    benchmark :

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import umbrae
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> umbrae(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return mbrae(true, predicted, False, benchmark) / (1 - mbrae(true, predicted, False, benchmark))


def ve(true, predicted, treat_arrays: bool = True,
       **treat_arrays_kws) -> float:
    """
    `Volumetric efficiency <https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007WR006415>`_. Ranges from 0 to 1. Smaller the better.

    .. math::
        VE = 1 - \\frac{\\sum_{i=1}^{n} \\left| \\text{predicted}_i - \\text{true}_i \\right|}{\\sum_{i=1}^{n} \\text{true}_i}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import ve
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> ve(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    a = np.sum(np.abs(predicted - true))
    b = np.sum(true)
    return float(1 - (a / b))


def volume_error(true, predicted, treat_arrays: bool = True,
                 **treat_arrays_kws) -> float:
    """
    Returns the Volume Error (Ve).
    It is an indicator of the agreement between the averages of the simulated
    and observed runoff (i.e. long-term water balance).
    used in `Reynolds <https://doi.org/10.1016/j.jhydrol.2017.05.012>`_ paper:

    .. math::
        \\text{volume_error}= Sum(self.predicted- true)/sum(self.predicted)

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import volume_error
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> volume_error(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    # TODO written formula and executed formula are different.
    ve_ = np.sum(predicted - true) / np.sum(true)
    return float(ve_)


def wape(
        true,
        predicted,
        treat_arrays: bool = True,
         **treat_arrays_kws
) -> float:
    """
    `weighted absolute percentage error <https://mattdyor.wordpress.com/2018/05/23/calculating-wape/>`_. The lower the better.

    It is a variation of mape but more suitable for intermittent and low-volume `data <https://arxiv.org/pdf/2103.12057v1.pdf>`_.

    .. math::
        \\text{WAPE} = \\frac{\\sum_{i=1}^{n} \\left| \\text{true}_i - \\text{predicted}_i \\right|}{\\sum_{i=1}^{n} \\text{true}_i}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import wape
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> wape(t, p)


    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.sum(_ae(true, predicted) / np.sum(true)))


def watt_m(true, predicted, treat_arrays: bool = True,
           **treat_arrays_kws) -> float:
    """
    `Watterson's M. <https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1097-0088(199604)16:4%3C379::AID-JOC18%3E3.0.CO;2-U>`_

    .. math::
        M = \\frac{2}{\\pi} \\cdot \\arcsin \\left( 1 - \\frac{\\frac{1}{n} \\sum_{i=1}^{n} ( \\text{true}_i - \\text{predicted}_i )^2}{\\sigma_{\\text{true}}^2 + \\sigma_{\\text{predicted}}^2 + (\\mu_{\\text{predicted}} - \\mu_{\\text{true}})^2} \\right)

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import watt_m
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> watt_m(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    a = 2 / np.pi
    c = np.std(true, ddof=1) ** 2 + np.std(predicted, ddof=1) ** 2
    e = (np.mean(predicted) - np.mean(true)) ** 2
    f = c + e
    return float(a * np.arcsin(1 - (mse(true, predicted, treat_arrays=False) / f)))


def wmape(
        true, predicted, treat_arrays: bool = True,
          **treat_arrays_kws) -> float:
    """
    `Weighted Mean Absolute Percent Error <https://stackoverflow.com/a/54833202/5982232>`_.

    .. math::
        \\text{WMAPE} = \\frac{\\sum_{i=1}^{n} \\left| \\text{true}_i - \\text{predicted}_i \\right|}{\\sum_{i=1}^{n} \\text{true}_i}


    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import wmape
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> wmape(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    # Take a series (actual) and a dataframe (forecast) and calculate wmape
    # for each forecast. Output shape is (1, num_forecasts)

    # Make an array of mape (same shape as forecast)
    se_mape = abs(true - predicted) / true

    # Calculate sum of actual values
    ft_actual_sum = true.sum(axis=0)

    # Multiply the actual values by the mape
    se_actual_prod_mape = true * se_mape

    # Take the sum of the product of actual values and mape
    # Make sure to sum down the rows (1 for each column)
    ft_actual_prod_mape_sum = se_actual_prod_mape.sum(axis=0)

    # Calculate wmape for each forecast and return as a dictionary
    ft_wmape_forecast = ft_actual_prod_mape_sum / ft_actual_sum
    return float(ft_wmape_forecast)


def norm_ape(true, predicted, treat_arrays: bool = True,
             **treat_arrays_kws) -> float:
    """ Normalized Absolute Percentage Error

    .. math::
        \\text{norm_APE} = \\sqrt{ \\frac{1}{n-1} \\sum_{i=1}^{n} \\left( \\left| \\frac{\\text{true}_i - \\text{predicted}_i}{\\text{true}_i} \\right| - \\frac{1}{n} \\sum_{j=1}^{n} \\left| \\frac{\\text{true}_j - \\text{predicted}_j}{\\text{true}_j} \\right| \\right)^2 }

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import norm_ape
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> norm_ape(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(
        np.sqrt(np.sum(np.square(_percentage_error(true, predicted) - mape(true, predicted, treat_arrays=False))) / (
                    len(true) - 1)))


def mse(true, predicted, treat_arrays: bool = True, weights=None,
        **treat_arrays_kws) -> float:
    """
    `Mean Square Error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html>`_

    .. math::
        MSE = \\frac{\\sum_{i=1}^{N} w_i (true_i - predicted_i)^2}{\\sum_{i=1}^{N} w_i}

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
         simulated values
    treat_arrays :
        process the true and predicted arrays using maybe_treat_arrays function
    weights :

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import mse
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)treat_arrays
    >>> mse(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(np.average((true - predicted) ** 2, axis=0, weights=weights))


def variability_ratio(true, predicted, treat_arrays: bool = True,
                      **treat_arrays_kws) -> float:
    """
    Variability Ratio
    It is the ratio of the variance of the predicted values to the variance of the true values.
    It is used to measure the variability of the predicted values relative to the true values.

    .. math::
        VR = 1 - \\left| \\frac{\\frac{\\sigma_{\\text{predicted}}}{\\mu_{\\text{predicted}}}}{\\frac{\\sigma_{\\text{true}}}{\\mu_{\\text{true}}}} - 1 \\right|

    Parameters
    ----------
    true :
         true/observed/actual/target values. It must be a numpy array,
         or pandas series/DataFrame or a list.
    predicted :
        simulated/predicted values
    treat_arrays :
        treat_arrays the true and predicted array

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import variability_ratio
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> variability_ratio(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    return float(1 - abs((np.std(predicted) / np.mean(predicted)) / (np.std(true) / np.mean(true)) - 1))

def concordance_corr_coef(
        true, predicted, treat_arrays: bool = True, **treat_arrays_kws
) -> float:
    """
    `Concordance Correlation Coefficient (CCC) <https://en.wikipedia.org/wiki/Concordance_correlation_coefficient>`_
    taken from this `paper <https://doi.org/10.2307/2532051>`_.

    .. math::
        CCC = \\frac{2 \\rho \\sigma_{true} \\sigma_{predicted}}{\\sigma_{true}^2 + \\sigma_{predicted}^2 + (\\bar{true} - \\bar{predicted})^2}

    Parameters
    ----------
    true :
        true/observed/actual/target values. It must be a numpy array,
        or pandas series/DataFrame or a list.
    predicted :
        simulated values
    treat_arrays :
        treat_arrays the true and predicted array

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import concordance_corr_coef
    >>> t = np.random.random(10)
    >>> p = np.random.random(10)
    >>> concordance_corr_coef(t, p)
    """

    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)

    # taken from https://nirpyresearch.com/concordance-correlation-coefficient/
    mean_true = np.mean(true)
    mean_predicted = np.mean(predicted)

    var_true = np.var(true)
    var_predicted = np.var(predicted)

    true_mean = np.mean(true)
    pred_mean = np.mean(predicted)

    numerator = np.sum((true - true_mean) * (predicted - pred_mean))
    denominator = np.sqrt(np.sum((true - true_mean) ** 2)) * np.sqrt(np.sum((predicted - pred_mean) ** 2))

    pearson = numerator / denominator

    ccc = (2 * pearson * np.sqrt(var_true) * np.sqrt(var_predicted)) / (var_true + var_predicted + (mean_true - mean_predicted) ** 2)

    return float(ccc)

def critical_success_index(
        true, predicted,treat_arrays: bool = True, threshold=0.5, **treat_arrays_kws
)->float:
    """
    `Critical Success Index (CSI) <https://doi.org/10.1016/j.heliyon.2024.e26371>`_

    .. math::
        CSI = \\frac{TP}{TP + FN + FP}

    Parameters
    ----------
    true :
        True/observed/actual/target values. It should be a binary array (0s and 1s),
        or a continuous array where values are binarized using a threshold.
    predicted :
        Predicted values, same format as 'true'.
    treat_arrays :
        treat_arrays the true and predicted array
    threshold :
        Threshold for binarizing continuous values (if applicable).

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import critical_success_index
    >>> t = np.array([0.4, 0.1, 0.1, 0.3, 0.7, 0.1])
    >>> p = np.array([0.8, 0.11, 0.5, 0.1, 0.1, 0.1])
    >>> critical_success_index(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)

    # Binarize if the arrays are not binary
    true = np.array(true) >= threshold
    predicted = np.array(predicted) >= threshold

    TP = np.sum((true == 1) & (predicted == 1))
    FN = np.sum((true == 1) & (predicted == 0))
    FP = np.sum((true == 0) & (predicted == 1))

    csi = TP / float(TP + FN + FP) if (TP + FN + FP) > 0 else 0
    return float(csi)

def kl_divergence(
        true, predicted, treat_arrays: bool = True, **treat_arrays_kws
)->float:
    """
    `Kullback-Leibler Divergence <https://doi.org/10.1016/j.imu.2024.101510>`_

    .. math::
        D_{KL}(P \\parallel Q) = \\sum_{i} P(i) \\log \\left( \\frac{P(i)}{Q(i)} \\right)

    Parameters
    ----------
    true :
        True/observed/actual/target probability distribution. It must be a numpy array,
        pandas series/DataFrame, or a list.
    predicted :
        Predicted probability distribution, same format as 'true'.
    treat_arrays :
        treat_arrays the true and predicted array

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import kl_divergence
    >>> t = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
    >>> p = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    >>> divergence = kl_divergence(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)

    # taken from https://lightning.ai/docs/torchmetrics/stable/regression/kl_divergence.html
    predicted = predicted / predicted.sum(axis=-1, keepdims=True)
    true = true / true.sum(axis=-1, keepdims=True)

    res = predicted * np.log(predicted / true)
    res[predicted == 0] = 0.0
    kl = res.sum(axis=-1)

    return float(kl)


def log_cosh_error(
        true, predicted, treat_arrays: bool = True, **treat_arrays_kws
)->float:
    """
    `Log-Cosh Error <https://doi.org/10.1016/j.compchemeng.2022.107933>`_

    .. math::
        \\text{Log-Cosh Error} = \\frac{1}{n} \\sum_{i=1}^{n} \\log \\left( \\cosh(\\text{predicted}_i - \\text{true}_i) \\right)

    Parameters
    ----------
    true :
        True/observed/actual/target values. It must be a numpy array,
        pandas series/DataFrame, or a list.
    predicted :
        Predicted values, same format as 'true'.
    treat_arrays :
        treat_arrays the true and predicted array

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import log_cosh_error
    >>> t = np.array([1, 2, 3, 4, 5])
    >>> p = np.array([1.1, 1.9, 3.1, 4.2, 4.8])
    >>> error = log_cosh_error(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)

    true = np.asarray(true)
    predicted = np.asarray(predicted)

    # Calculation of Log-Cosh Error
    error = np.log(np.cosh(predicted - true))
    return float(np.mean(error))

def minkowski_distance(
        true, predicted, order =1, treat_arrays: bool = True, **treat_arrays_kws
)->float:
    """
    `Minkowski Distance <https://doi.org/10.1016/j.imu.2024.101492>`_

    .. math::
        D_{Minkowski} = \\left( \\sum_{i=1}^{n} \\left| \\text{true}_i - \\text{predicted}_i \\right|^p \\right)^{\\frac{1}{p}}

    Parameters
    ----------
    true :
        True/observed/actual/target values. It must be a numpy array,
        pandas series/DataFrame, or a list.
    predicted :
        Predicted values, same format as 'true'.
    order :
        The order of the norm of the difference. `order=2` is equivalent to the Euclidean distance,
        `order=1` is the Manhattan distance.
    treat_arrays :
        treat_arrays the true and predicted array

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import minkowski_distance
    >>> t = np.array([1, 2, 3, 4, 5])
    >>> p = np.array([1.1, 1.9, 3.1, 4.2, 4.8])
    >>> order = 2  # Euclidean distance
    >>> distance = minkowski_distance(t, p, order)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)

    true = np.asarray(true)
    predicted = np.asarray(predicted)

    # Calculation of Minkowski Distance
    return float(np.sum(np.abs(true - predicted) ** order) ** (1 / order))

def tweedie_deviance_score(
        true, predicted, power=0, treat_arrays: bool = True, **treat_arrays_kws
)->float:
    """
    `Tweedie Deviance Score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_tweedie_deviance.html>`_

    .. math::
        D(\\text{true}, \\text{predicted}) = \\frac{1}{n} \\sum_{i=1}^{n} (\\text{true}_i - \\text{predicted}_i)^2

    .. math::
        D(\\text{true}, \\text{predicted}) = 2 \\sum_{i=1}^{n} \\left( \\text{true}_i \\log\\left(\\frac{\\text{true}_i + (\\text{true}_i = 0)}{\\text{predicted}_i}\\right) - \\text{true}_i + \\text{predicted}_i \\right)

    .. math::
        D(\\text{true}, \\text{predicted}) = 2 \\sum_{i=1}^{n} \\left( \\frac{\\text{true}_i}{\\text{predicted}_i} - \\log\\left(\\frac{\\text{true}_i}{\\text{predicted}_i}\\right) - 1 \\right)

    .. math::
        D(\\text{true}, \\text{predicted}) = 2 \\sum_{i=1}^{n} \\left( \\frac{(\\text{true}_i - \\text{predicted}_i)^2}{\\text{true}_i^2 \\text{predicted}_i} \\right)

    Parameters
    ----------
    true :
        True/observed/actual/target values. It must be a numpy array,
        pandas series/DataFrame, or a list.
    predicted :
        Predicted values, same format as 'true'.
    power :
        The power determines the underlying target distribution.
        `power=0` for Normal, `power=1` for Poisson, `power=2` for Gamma,
        and `power=3` for Inverse Gaussian.
    treat_arrays :
        treat_arrays the true and predicted array

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import tweedie_deviance_score
    >>> t = np.array([1, 2, 3, 4, 5])
    >>> p = np.array([1.1, 1.9, 3.1, 4.2, 4.8])
    >>> power = 2  # Gamma distribution
    >>> score = tweedie_deviance_score(t, p, power)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)

    true = np.asarray(true)
    predicted = np.asarray(predicted)

    # Tweedie Deviance Score calculation
    if power == 0:
        # Normal distribution
        return np.mean((true - predicted) ** 2).item()
    elif power == 1:
        # Poisson distribution
        return 2 * np.sum(true * np.log((true + (true == 0)) / predicted) - true + predicted)
    elif power == 2:
        # Gamma distribution
        return 2 * np.sum((true / predicted) - np.log(true / predicted) - 1)
    elif power == 3:
        # Inverse Gaussian distribution
        return 2 * np.sum((true - predicted) ** 2 / (true ** 2 * predicted))
    else:
        raise ValueError("Invalid power value. Power must be 0 (Normal), 1 (Poisson), 2 (Gamma), or 3 (Inverse Gaussian).")


def mre(
        true,
        predicted,
        benchmark:np.ndarray = None,
        treat_arrays: bool = True,
        **treat_arrays_kws
)->float:
    """
    `mean relative error <https://doi.org/10.1016/j.trd.2022.103505>`_

    .. math::
        \\text{MRE} = \\frac{1}{n} \\sum_{i=1}^{n} \\left| \\frac{\\text{true}_i - \\text{predicted}_i}{\\text{true}_i} \\right|

    Parameters
    ----------
    true :
        True/observed/actual/target values. It must be a numpy array,
        pandas series/DataFrame, or a list.
    predicted :
        Predicted values, same format as 'true'.
    benchmark :
    treat_arrays :
        treat_arrays the true and predicted array

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import mre
    >>> t = np.array([1, 2, 3, 4, 5])
    >>> p = np.array([1.1, 1.9, 3.1, 4.2, 4.8])
    >>> score = mre(t, p)
    """

    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
    re = _relative_error(true, predicted, benchmark)
    return float(np.mean(re))


# def peak_flow_ratio(
#         true,
#         predicted,
#         treat_arrays: bool = True,
#         **treat_arrays_kws
# )->float:
#     """
#     Peak flow ratio is defined the ratio of the highest simulated to the highest
#     observed flow rates (Broekhuizen et al., 2020).
#
#     https://doi.org/10.5194/hess-24-869-2020
#
#     Parameters
#     ----------
#     true :
#         True/observed/actual/target values. It must be a numpy array,
#         pandas series/DataFrame, or a list.
#     predicted :
#         Predicted values, same format as 'true'.
#     treat_arrays :
#         treat_arrays the true and predicted array
#
#     Examples
#     ---------
#     >>> import numpy as np
#     >>> from SeqMetrics import peak_flow_ratio
#     >>> t = np.array([1, 2, 3, 4, 5])
#     >>> p = np.array([1.1, 1.9, 3.1, 4.2, 4.8])
#     >>> score = mre(t, p)
#     """
#
#     true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
#     o_max = np.max(true)
#     s_max = np.max(predicted)
#     return float(s_max/o_max)

def mape_for_peaks(
        true,
        predicted,
        treat_arrays: bool = True,
        **treat_arrays_kws
) -> float:
    """
    Mean Absolute Percentage Error for peaks which are found using
    `scipy.singnal.find_peaks <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`_

    .. math::
        \\text{MAPE}_\\text{peak} = \\frac{1}{P}\\sum_{p=1}^{P} \\left |\\frac{Q_{s,p} - Q_{o,p}}{Q_{o,p}} \\right | \\times 100,

    Parameters
    ----------
    true :
        True/observed/actual/target values. It must be a numpy array,
        pandas series/DataFrame, or a list.
    predicted :
        Predicted values, same format as 'true'.
    treat_arrays :
        treat_arrays the true and predicted array

    https://github.com/neuralhydrology/neuralhydrology/blob/master/neuralhydrology/evaluation/metrics.py#L707

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import mape_for_peaks
    >>> t = np.array([1, 2, 3, 4, 5])
    >>> p = np.array([1.1, 1.9, 3.1, 4.2, 4.8])
    >>> score = mre(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)

    peaks, _ = find_peaks(true, prominence=np.std(true))

    if peaks.size == 0:
        return np.nan

    true = true[peaks]
    predicted = predicted[peaks]

    return mape(true, predicted, treat_arrays=False)


# todo, relative_error following equation from Moriasi does not return scaler value
# it taken mean, then what will be the difference between this and mre?
# def relative_error(
#         true,
#         predicted,
#         treat_arrays: bool = True,
#         **treat_arrays_kws
# ):
#     """
#     Relative Error. It indicates the mismatch that
#     occurs between the observed and modeled values, expressed
#     in terms of percentages.
#     It quantifies the relative deviations between observed/true
#     and predicted values. This significantly reduces the influence of absolute
#     differences at peaks. The absolute lower differences during low flow
#     periods are enhanced because they are significant if looked at in a
#     relative sense. As a result, there might be a systematic over- or underprediction during low flow periods.
#     It used along with other statistics to quantify low
#     flow simulations Moriasi et al., 2007.
#
#     Parameters
#     ----------
#     true :
#          true/observed/actual/target values. It must be a numpy array,
#          or pandas series/DataFrame or a list.
#     predicted :
#          simulated/predicted values
#     treat_arrays :
#         treat_arrays the true and predicted array
#
#     Examples
#     ---------
#     >>> import numpy as np
#     >>> from SeqMetrics import relative_error
#     >>> t = np.array([1, 2, 3, 4, 5])
#     >>> p = np.array([1.1, 1.9, 3.1, 4.2, 4.8])
#     >>> score = mre(t, p)
#     """
#     true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)
#
#     error = true - predicted
#
#     return np.abs(error / (true + EPS)) * 100

def manhattan_distance(
        true,
        predicted,
        treat_arrays: bool = True,
        **treat_arrays_kws
) -> float:
    """
    Manhattan distance, also known as cityblock distance or taxicab norm.

    See `Blanco-Mallo et al., 2023 <https://doi.org/10.1016/j.patcog.2023.109646>`_
    and `Alexei Botchkarev 2019 <https://www.ijikm.org/Volume14/IJIKMv14p045-076Botchkarev5064.pdf>`_
    on the use of distances in performance measures.

    .. math::
        D_{\\text{manhattan}} = \\sum_{i=1}^{n} \\left| \\text{true}_i - \\text{predicted}_i \\right|

    Parameters
    ----------
    true :
        True/observed/actual/target values. It must be a numpy array,
        pandas series/DataFrame, or a list.
    predicted :
        Predicted values, same format as 'true'.
    treat_arrays :
        treat_arrays the true and predicted array

    Examples
    ---------
    >>> import numpy as np
    >>> from SeqMetrics import manhattan_distance
    >>> t = np.random.random(100)
    >>> p = np.random.random(100)
    >>> manhattan_distance(t, p)
    """
    true, predicted = maybe_treat_arrays(treat_arrays, true, predicted, 'regression', **treat_arrays_kws)

    return float(np.sum(np.abs(true - predicted)))


def drv():
    # https://rstudio-pubs-static.s3.amazonaws.com/433152_56d00c1e29724829bad5fc4fd8c8ebff.html
    raise NotImplementedError

def gini():
    # https://github.com/benhamner/Metrics/blob/master/MATLAB/metrics/gini.m
    raise NotImplementedError

def log_likelihood():
    # https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/elementwise.py#L202
    raise NotImplementedError

def cross_entropy():
    # https://datascience.stackexchange.com/q/20296
    raise NotImplementedError

def vaf():
    # https://www.dcsc.tudelft.nl/~jwvanwingerden/lti/doc/html/vaf.html
    raise NotImplementedError

def resid_std_error():
    # https://www.statology.org/residual-standard-error-r/
    raise NotImplementedError

def eff_coeff():
    # https://doi.org/10.1016/j.csite.2022.101797
    raise NotImplementedError

def overall_index():
    # https://doi.org/10.1016/j.csite.2022.101797
    raise NotImplementedError

def resid_mass_coeff():
    # https://doi.org/10.1016/j.csite.2022.101797
    raise NotImplementedError

def log_euclid_dist():
    #
    raise NotImplementedError