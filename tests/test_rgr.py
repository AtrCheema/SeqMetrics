import os
import unittest
import site  # so that SeqMetrics directory is in path

seqmet_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(seqmet_dir)

import scipy
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, kendalltau

from sklearn.metrics.pairwise import cosine_similarity as sklearn_cos_sim
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance
from sklearn.metrics import max_error, explained_variance_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
from sklearn.metrics import median_absolute_error, mean_absolute_percentage_error

from SeqMetrics import mse as sm_mse
from SeqMetrics import RegressionMetrics
from SeqMetrics import nse
from SeqMetrics import nse_alpha
from SeqMetrics import nse_beta
from SeqMetrics import nse_mod
from SeqMetrics import nse_rel
from SeqMetrics import nse_bound
from SeqMetrics import r2_score as sm_r2_score
from SeqMetrics import adjusted_r2
from SeqMetrics import kge
from SeqMetrics import kge_bound
from SeqMetrics import kge_mod
from SeqMetrics import kge_np
from SeqMetrics import log_nse
from SeqMetrics import corr_coeff
from SeqMetrics import rmse as sm_rmse
from SeqMetrics import rmsle as sm_rmsle
from SeqMetrics import mape as sm_mape
from SeqMetrics import nrmse
from SeqMetrics import pbias
from SeqMetrics import bias
from SeqMetrics import med_seq_error
from SeqMetrics import mae as sm_mae
from SeqMetrics import abs_pbias
from SeqMetrics import gmae
from SeqMetrics import inrse
from SeqMetrics import irmse
from SeqMetrics import mase
from SeqMetrics import mare
from SeqMetrics import msle as sm_msle
from SeqMetrics import covariance
from SeqMetrics import bic
from SeqMetrics import sse
from SeqMetrics import amemiya_pred_criterion
from SeqMetrics import amemiya_adj_r2
from SeqMetrics import aitchison
from SeqMetrics import aic
from SeqMetrics import acc
from SeqMetrics import euclid_distance
from SeqMetrics import cosine_similarity
from SeqMetrics import decomposed_mse
from SeqMetrics import cronbach_alpha
from SeqMetrics import exp_var_score as sm_exp_var_score
from SeqMetrics import expanded_uncertainty
from SeqMetrics import fdc_fhv
from SeqMetrics import fdc_flv
from SeqMetrics import gmean_diff
from SeqMetrics import gmrae
from SeqMetrics import calculate_hydro_metrics
from SeqMetrics import JS
from SeqMetrics import kendaull_tau
from SeqMetrics import kgeprime_bound
from SeqMetrics import kgenp_bound
from SeqMetrics import kl_sym
from SeqMetrics import lm_index
from SeqMetrics import maape
from SeqMetrics import mbrae
from SeqMetrics import max_error as sm_max_error
from SeqMetrics import mb_r
from SeqMetrics import mda
from SeqMetrics import mde
from SeqMetrics import mdape
from SeqMetrics import mdrae
from SeqMetrics import me
from SeqMetrics import mean_bias_error
from SeqMetrics import mean_var
from SeqMetrics import mean_poisson_deviance as sm_mean_poisson_deviance
from SeqMetrics import mean_gamma_deviance as sm_mean_gamma_deviance
from SeqMetrics import median_abs_error as sm_median_abs_error
from SeqMetrics import mle
from SeqMetrics import mod_agreement_index
from SeqMetrics import mpe
from SeqMetrics import mrae
from SeqMetrics import norm_euclid_distance
from SeqMetrics import nrmse_range
from SeqMetrics import nrmse_ipercentile
from SeqMetrics import nrmse_mean
from SeqMetrics import norm_ae
from SeqMetrics import norm_ape
from SeqMetrics import log_prob
from SeqMetrics import rmdspe
from SeqMetrics import rse
from SeqMetrics import rrse
from SeqMetrics import rae
from SeqMetrics import ref_agreement_index
from SeqMetrics import rel_agreement_index
from SeqMetrics import relative_rmse
from SeqMetrics import rmspe
from SeqMetrics import rsr
from SeqMetrics import rmsse
from SeqMetrics import sa
from SeqMetrics import sc
from SeqMetrics import smape
from SeqMetrics import smdape
from SeqMetrics import sid
from SeqMetrics import skill_score_murphy
from SeqMetrics import std_ratio
from SeqMetrics import umbrae
from SeqMetrics import ve
from SeqMetrics import volume_error
from SeqMetrics import wape
from SeqMetrics import watt_m
from SeqMetrics import wmape
from SeqMetrics import variability_ratio
from SeqMetrics import concordance_corr_coef as sm_concordance_corr_coef
from SeqMetrics import critical_success_index as sm_critical_success_index
from SeqMetrics import kl_divergence as sm_kl_divergence
from SeqMetrics import log_cosh_error as sm_log_cosh_error
from SeqMetrics import minkowski_distance as sm_minkowski_distance
from SeqMetrics import tweedie_deviance_score as sm_tweedie_deviance_score

from SeqMetrics.utils import maybe_treat_arrays

not_metrics = ['calculate_all',
               "treat_arrays",
               "scale_free_metrics",
               "scale_dependent_metrics",
               "composite_metrics",
               "relative_metrics",
               "percentage_metrics"]

random_state = np.random.RandomState(seed=313)

t11 = random_state.random(100)
p11 = random_state.random(100)
metrics = RegressionMetrics(t11, p11)

# check with very large values say > 1e7
t_large = random_state.random(100) * 1e7
p_large = random_state.random(100) * 1e7
metrics_large = RegressionMetrics(t_large, p_large)

# check by inserting NaN values at random places in t11 and p11
t_nan = t11.copy()
p_nan = p11.copy()
nan_indices = random_state.choice(range(100), 10, replace=False)
t_nan[nan_indices] = np.nan
nan_indices = random_state.choice(range(100), 10, replace=False)
p_nan[nan_indices] = np.nan
metrics_nan = RegressionMetrics(t_nan, p_nan)

# check where some values are negative and some values are positive
t_neg = random_state.randint(-100, 100, 100)
p_neg = random_state.randint(-100, 100, 100)
metrics_neg = RegressionMetrics(t_neg, p_neg)


class test_errors(unittest.TestCase):
    
    def test_attrs(self):
        for _attr in not_metrics:
            assert _attr not in metrics.all_methods
    
    def test_calculate_all(self):
        
        all_errors = metrics.calculate_all()
        assert len(all_errors) > 100
        assert all([isinstance(val, float) for val in all_errors.values()])

        all_errors = metrics_large.calculate_all()
        assert len(all_errors) > 100
        assert all([isinstance(val, float) for val in all_errors.values()])

        all_errors = metrics_nan.calculate_all()
        assert len(all_errors) > 100
        assert all([isinstance(val, float) for val in all_errors.values()])

        all_errors = metrics_neg.calculate_all()
        assert len(all_errors) > 100
        assert all([isinstance(val, float) for val in all_errors.values()])

        return

    def test_mrae(self):
        # https://support.numxl.com/hc/en-us/articles/115001223363-MRAE-Mean-Relative-Absolute-Error
        data = np.array(
            [[-2.9, -2.95],
             [-2.83, -2.7],
             [-0.95, -1.00],
             [-0.88, -0.68],
             [1.21, 1.50],
             [-1.67, -1.00],
             [0.83, 0.90],
             [-0.27, -0.37],
             [1.36, 1.26],
             [-0.34, -0.54],
             [0.48, 0.58],
             [-2.83, -2.13],
             [-0.95, -0.75],
             [-0.88, -0.89],
             [1.21, 1.25],
             [-1.67, -1.65],
             [-2.99, -3.20],
             [1.24, 1.29],
             [0.64, 0.60]]
        )
        errs = RegressionMetrics(data[:, 0], data[:, 1])
        np.testing.assert_almost_equal(0.348, errs.mrae(), 2)
        return

    def test_hydro_metrics(self):
        hydr_metrics = metrics.calculate_hydro_metrics()
        assert len(hydr_metrics) == len(metrics._hydro_metrics())

        hydr_metrics = metrics_large.calculate_hydro_metrics()
        assert len(hydr_metrics) == len(metrics._hydro_metrics())

        hydr_metrics = metrics_nan.calculate_hydro_metrics()
        assert len(hydr_metrics) == len(metrics._hydro_metrics())

        hydr_metrics = metrics_neg.calculate_hydro_metrics()
        assert len(hydr_metrics) == len(metrics._hydro_metrics())

        return

    def test_minimal(self):
        minimal_metrics = metrics.calculate_minimal()
        assert len(minimal_metrics) == len(metrics._minimal())

        minimal_metrics = metrics_large.calculate_minimal()
        assert len(minimal_metrics) == len(metrics._minimal())

        minimal_metrics = metrics_nan.calculate_minimal()
        assert len(minimal_metrics) == len(metrics._minimal())

        minimal_metrics = metrics_neg.calculate_minimal()
        assert len(minimal_metrics) == len(metrics._minimal())
        return

    def test_scale_dependent(self):
        minimal_metrics = metrics.calculate_scale_dependent_metrics()
        assert len(minimal_metrics) == len(metrics._scale_dependent_metrics())

        minimal_metrics = metrics_large.calculate_scale_dependent_metrics()
        assert len(minimal_metrics) == len(metrics._scale_dependent_metrics())

        minimal_metrics = metrics_nan.calculate_scale_dependent_metrics()
        assert len(minimal_metrics) == len(metrics._scale_dependent_metrics())

        minimal_metrics = metrics_neg.calculate_scale_dependent_metrics()
        assert len(minimal_metrics) == len(metrics._scale_dependent_metrics())

        return

    def test_scale_independent(self):
        minimal_metrics = metrics.calculate_scale_independent_metrics()
        assert len(minimal_metrics) == len(metrics._scale_independent_metrics())

        minimal_metrics = metrics_large.calculate_scale_independent_metrics()
        assert len(minimal_metrics) == len(metrics._scale_independent_metrics())

        minimal_metrics = metrics_nan.calculate_scale_independent_metrics()
        assert len(minimal_metrics) == len(metrics._scale_independent_metrics())

        minimal_metrics = metrics_neg.calculate_scale_independent_metrics()
        assert len(minimal_metrics) == len(metrics._scale_independent_metrics())
        return

    def test_r2(self):
        new_r2 = metrics.r2()
        _, _, rvalue, _, _ = scipy.stats.linregress(t11, p11)
        assert np.allclose(new_r2, rvalue**2)

        new_r2 = metrics_large.r2()
        _, _, rvalue, _, _ = scipy.stats.linregress(t_large, p_large)
        assert np.allclose(new_r2, rvalue**2)

        new_r2 = metrics_nan.r2()
        t_nan_, p_nan_ = maybe_treat_arrays(True, t_nan, p_nan, 'regression', remove_nan=True)
        _, _, rvalue, _, _ = scipy.stats.linregress(t_nan_, p_nan_)
        assert np.allclose(new_r2, rvalue**2)

        new_r2 = metrics_neg.r2()
        _, _, rvalue, _, _ = scipy.stats.linregress(t_neg, p_neg)
        assert np.allclose(new_r2, rvalue**2)
        return

    def test_mse_cls(self):
        new_mse = metrics.mse()
        sk_mse= mean_squared_error(t11, p11)
        assert np.allclose(new_mse, sk_mse)

        new_mse = metrics_large.mse()
        sk_mse= mean_squared_error(t_large, p_large)
        assert np.allclose(new_mse, sk_mse)

        new_mse = metrics_nan.mse()
        t_nan_, p_nan_ = maybe_treat_arrays(True, t_nan, p_nan, 'regression', remove_nan=True)
        sk_mse= mean_squared_error(t_nan_, p_nan_)
        assert np.allclose(new_mse, sk_mse)

        new_mse = metrics_neg.mse()
        sk_mse= mean_squared_error(t_neg, p_neg)
        assert np.allclose(new_mse, sk_mse)
        return

    def test_mse_func(self):
        new_mse = sm_mse(t11, p11)
        sk_mse= mean_squared_error(t11, p11)
        assert np.allclose(new_mse, sk_mse)

        new_mse = sm_mse(t_large, p_large)
        sk_mse= mean_squared_error(t_large, p_large)
        assert np.allclose(new_mse, sk_mse)

        new_mse = sm_mse(t_nan, p_nan)
        t_nan_, p_nan_ = maybe_treat_arrays(True, t_nan, p_nan, 'regression', remove_nan=True)
        sk_mse= mean_squared_error(t_nan_, p_nan_)
        assert np.allclose(new_mse, sk_mse)

        new_mse = sm_mse(t_neg, p_neg)
        sk_mse= mean_squared_error(t_neg, p_neg)
        assert np.allclose(new_mse, sk_mse)
        return

    def test_nse(self):
        # verified against https://agrimetsoft.com/calculators/Nash%20Sutcliffe%20model%20Efficiency%20coefficient
        new_nse = nse(t11, p11)
        assert np.allclose(new_nse, -1.068372251749874)
        return

    def test_nse_alpha(self):
        new_nse_alpha = nse_alpha(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_nse_alpha, 1.0235046034233621)
        return

    def test_nse_beta(self):
        new_nse_beta = nse_beta(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_nse_beta, 0.2405519617999516)
        return

    def test_nse_mod(self):
        new_nse_mod = nse_mod(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_nse_mod, -0.32879454094431804)
        return

    def test_nse_rel(self):
        new_nse_rel = nse_rel(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_nse_rel, -517670.8159599439)
        return

    def test_nse_bound(self):
        new_nse_bound = nse_bound(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_nse_bound, -0.34818860428052295)
        return

    def test_r2_score_cls(self):
        new_r2_score = metrics.r2_score()
        sk_r2_score= r2_score(t11, p11)
        assert np.allclose(new_r2_score, sk_r2_score)

        new_r2_score = metrics_large.r2_score()
        sk_r2_score= r2_score(t_large, p_large)
        assert np.allclose(new_r2_score, sk_r2_score)

        new_r2_score = metrics_nan.r2_score()
        t_nan_, p_nan_ = maybe_treat_arrays(True, t_nan, p_nan, 'regression', remove_nan=True)
        sk_r2_score= r2_score(t_nan_, p_nan_)
        assert np.allclose(new_r2_score, sk_r2_score)

        new_r2_score = metrics_neg.r2_score()
        sk_r2_score= r2_score(t_neg, p_neg)
        assert np.allclose(new_r2_score, sk_r2_score)
        return

    def test_r2_score_func(self):
        new_r2_score = sm_r2_score(t11, p11)
        sk_r2_score = r2_score(t11, p11)
        assert np.allclose(new_r2_score, sk_r2_score)

        new_r2_score = sm_r2_score(t_large, p_large)
        sk_r2_score = r2_score(t_large, p_large)
        assert np.allclose(new_r2_score, sk_r2_score)

        new_r2_score = sm_r2_score(t_nan, p_nan)
        t_nan_, p_nan_ = maybe_treat_arrays(True, t_nan, p_nan, 'regression', remove_nan=True)
        sk_r2_score = r2_score(t_nan_, p_nan_)
        assert np.allclose(new_r2_score, sk_r2_score)

        new_r2_score = sm_r2_score(t_neg, p_neg)
        sk_r2_score = r2_score(t_neg, p_neg)
        assert np.allclose(new_r2_score, sk_r2_score)
        return

    def test_adjusted_r2(self):
        new_adjusted_r2 = adjusted_r2(t11, p11)
        # Equation taken from https://people.duke.edu/~rnau/rsquared.htm
        assert np.allclose(new_adjusted_r2, -0.009873060763049724)
        return

    def test_kge(self):
        new_kge = kge(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_kge, 0.008970625237195717)
        return

    def test_kge_bound(self):
        new_kge_bound = kge_bound(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_kge_bound, 0.0045055212900936776)
        return

    def test_kge_mod(self):
        new_kge_mod = kge_mod(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_kge_mod, 0.004612979178136856)
        return

    def test_kge_np(self):
        new_kge_np = kge_np(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_kge_np, -0.006718774884987511)
        return

    def test_log_nse(self):
        new_log_nse = log_nse(t11, p11)
        assert np.allclose(new_log_nse, 1.0)
        return

    def test_corr_coeff(self):
        new_corr_coeff = corr_coeff(t11, p11)
        assert np.allclose(new_corr_coeff, pearsonr(t11, p11)[0])

        new_corr_coeff = corr_coeff(t_large, p_large)
        assert np.allclose(new_corr_coeff, pearsonr(t_large, p_large)[0])

        new_corr_coeff = corr_coeff(t_nan, p_nan)
        t_nan_, p_nan_ = maybe_treat_arrays(True, t_nan, p_nan, 'regression', remove_nan=True)
        assert np.allclose(new_corr_coeff, pearsonr(t_nan_, p_nan_)[0])

        new_corr_coeff = corr_coeff(t_neg, p_neg)
        assert np.allclose(new_corr_coeff, pearsonr(t_neg, p_neg)[0])
        return

    def test_rmse(self):
        new_rmse = sm_rmse(t11, p11)
        assert np.allclose(new_rmse, np.sqrt(mean_squared_error(t11, p11)))

        new_rmse = sm_rmse(t_large, p_large)
        assert np.allclose(new_rmse, np.sqrt(mean_squared_error(t_large, p_large)))

        new_rmse = sm_rmse(t_nan, p_nan)
        t_nan_, p_nan_ = maybe_treat_arrays(True, t_nan, p_nan, 'regression', remove_nan=True)
        assert np.allclose(new_rmse, np.sqrt(mean_squared_error(t_nan_, p_nan_)))

        new_rmse = sm_rmse(t_neg, p_neg)
        assert np.allclose(new_rmse, np.sqrt(mean_squared_error(t_neg, p_neg)))
        return

    def test_rmsle(self):
        new_rmsle = sm_rmsle(t11, p11)
        assert np.allclose(new_rmsle, np.sqrt(mean_squared_log_error(t11, p11)))

        new_rmsle = sm_rmsle(t_large, p_large)
        assert np.allclose(new_rmsle, np.sqrt(mean_squared_log_error(t_large, p_large)))

        new_rmsle = sm_rmsle(t_nan, p_nan)
        t_nan_, p_nan_ = maybe_treat_arrays(True, t_nan, p_nan, 'regression', remove_nan=True)
        assert np.allclose(new_rmsle, np.sqrt(mean_squared_log_error(t_nan_, p_nan_)))

        new_rmsle = sm_rmsle(t_neg, p_neg)
        # assert np.allclose(new_rmsle, np.sqrt(mean_squared_log_error(t_neg, p_neg)))
        return

    def test_mape_cls(self):
        new_mape = metrics.mape()
        sk_mape= mean_absolute_percentage_error(t11, p11) * 100.0
        self.assertAlmostEqual(new_mape, sk_mape)

        new_mape = metrics_large.mape()
        sk_mape= mean_absolute_percentage_error(t_large, p_large) * 100.0
        self.assertAlmostEqual(new_mape, sk_mape)

        new_mape = metrics_nan.mape()
        t_nan_, p_nan_ = maybe_treat_arrays(True, t_nan, p_nan, 'regression', remove_nan=True)
        sk_mape= mean_absolute_percentage_error(t_nan_, p_nan_) * 100.0
        self.assertAlmostEqual(new_mape, sk_mape)

        new_mape = metrics_neg.mape()
        sk_mape= mean_absolute_percentage_error(t_neg, p_neg) * 100.0
        self.assertAlmostEqual(new_mape, sk_mape)
        return

    def test_mape_func(self):
        new_mape = sm_mape(t11, p11)
        sk_mape= mean_absolute_percentage_error(t11, p11) * 100.0
        self.assertAlmostEqual(new_mape, sk_mape)

        new_mape = sm_mape(t_large, p_large)
        sk_mape= mean_absolute_percentage_error(t_large, p_large) * 100.0
        self.assertAlmostEqual(new_mape, sk_mape)

        new_mape = sm_mape(t_nan, p_nan)
        t_nan_, p_nan_ = maybe_treat_arrays(True, t_nan, p_nan, 'regression', remove_nan=True)
        sk_mape= mean_absolute_percentage_error(t_nan_, p_nan_) * 100.0
        self.assertAlmostEqual(new_mape, sk_mape)

        new_mape = sm_mape(t_neg, p_neg)
        sk_mape= mean_absolute_percentage_error(t_neg, p_neg) * 100.0
        self.assertAlmostEqual(new_mape, sk_mape)
        return

    def test_nrmse(self):
        new_nrmse = nrmse(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_nrmse, 0.4081874143525102)
        return

    def test_pbias(self):
        new_pbias = pbias(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_pbias, -13.214685733697532)
        return

    def test_bias(self):
        new_bias = bias(t11, p11)
        # Equation taken from https://doi.org/10.1029/97WR03495
        assert np.allclose(new_bias, -0.06738857779448111)
        return

    def test_med_seq_error(self):
        new_med_seq_error = med_seq_error(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_med_seq_error, 0.06731204476856545)
        return

    def test_mae_cls(self):
        new_mae = metrics.mae()
        sk_mae = mean_absolute_error(t11, p11)
        assert np.allclose(new_mae, sk_mae)

        new_mae = metrics_large.mae()
        sk_mae = mean_absolute_error(t_large, p_large)
        assert np.allclose(new_mae, sk_mae)

        new_mae = metrics_nan.mae()
        t_nan_, p_nan_ = maybe_treat_arrays(True, t_nan, p_nan, 'regression', remove_nan=True)
        sk_mae = mean_absolute_error(t_nan_, p_nan_)
        assert np.allclose(new_mae, sk_mae)

        new_mae = metrics_neg.mae()
        sk_mae = mean_absolute_error(t_neg, p_neg)
        assert np.allclose(new_mae, sk_mae)
        return

    def test_mae_func(self):
        new_mae = sm_mae(t11, p11)
        sk_mae = mean_absolute_error(t11, p11)
        assert np.allclose(new_mae, sk_mae)

        new_mae = sm_mae(t_large, p_large)
        sk_mae = mean_absolute_error(t_large, p_large)
        assert np.allclose(new_mae, sk_mae)

        new_mae = sm_mae(t_nan, p_nan)
        t_nan_, p_nan_ = maybe_treat_arrays(True, t_nan, p_nan, 'regression', remove_nan=True)
        sk_mae = mean_absolute_error(t_nan_, p_nan_)
        assert np.allclose(new_mae, sk_mae)

        new_mae = sm_mae(t_neg, p_neg)
        sk_mae = mean_absolute_error(t_neg, p_neg)
        assert np.allclose(new_mae, sk_mae)
        return

    def test_abs_pbias(self):
        new_abs_pbias = abs_pbias(t11, p11)
        assert np.allclose(new_abs_pbias, 62.05374050378925)
        return

    def test_gmae(self):
        new_gmae = gmae(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_gmae, 0.19423992928498718)
        return

    def test_inrse(self):
        new_inrse = inrse(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_inrse, 1.4381836641228662)
        return

    def test_irmse(self):
        new_irmse = irmse(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_irmse, 0.9954807723243245)
        return

    def test_mase(self):
        new_mase = mase(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_mase, 0.9609397361653512)
        return

    def test_mare_new(self):
        sm_mare = mare(t11, p11)
        assert sm_mare * 100.0 == sm_mape(t11, p11)
        assert np.allclose(sm_mare, mean_absolute_percentage_error(t11, p11))

        sm_mare = mare(t_large, p_large)
        assert sm_mare * 100.0 == sm_mape(t_large, p_large)
        assert np.allclose(sm_mare, mean_absolute_percentage_error(t_large, p_large))

        sm_mare = mare(t_nan, p_nan)
        assert sm_mare * 100.0 == sm_mape(t_nan, p_nan)
        # assert np.allclose(sm_mare, mean_absolute_percentage_error(t_nan, p_nan))

        sm_mare = mare(t_neg, p_neg)
        assert sm_mare * 100.0 == sm_mape(t_neg, p_neg)
        assert np.allclose(sm_mare, mean_absolute_percentage_error(t_neg, p_neg))
        return

    def test_msle_cls(self):
        new_msle = metrics.msle()
        sk_msle= mean_squared_log_error(t11, p11)
        assert np.allclose(new_msle, sk_msle)

        new_msle = metrics_large.msle()
        sk_msle= mean_squared_log_error(t_large, p_large)
        assert np.allclose(new_msle, sk_msle)

        new_msle = metrics_nan.msle()
        t_nan_, p_nan_ = maybe_treat_arrays(True, t_nan, p_nan, 'regression', remove_nan=True)
        sk_msle= mean_squared_log_error(t_nan_, p_nan_)
        assert np.allclose(new_msle, sk_msle)

        new_msle = metrics_neg.msle()
        # sk_msle= mean_squared_log_error(t_neg, p_neg)
        # assert np.allclose(new_msle, sk_msle)
        return

    def test_msle_func(self):
        new_msle = sm_msle(t11, p11)
        sk_msle= mean_squared_log_error(t11, p11)
        assert np.allclose(new_msle, sk_msle)

        new_msle = sm_msle(t_large, p_large)
        sk_msle= mean_squared_log_error(t_large, p_large)
        assert np.allclose(new_msle, sk_msle)

        new_msle = sm_msle(t_nan, p_nan)
        t_nan_, p_nan_ = maybe_treat_arrays(True, t_nan, p_nan, 'regression', remove_nan=True)
        sk_msle= mean_squared_log_error(t_nan_, p_nan_)
        assert np.allclose(new_msle, sk_msle)

        new_msle = sm_msle(t_neg, p_neg)
        # sk_msle= mean_squared_log_error(t_neg, p_neg)
        # assert np.allclose(new_msle, sk_msle)
        return

    def test_covariance(self):
        new_covariance = covariance(t11, p11)
        np_cov = np.cov(t11, p11, bias=True)[0][1]  # https://stackoverflow.com/a/39098306
        assert np.allclose(new_covariance, np_cov)

        new_covariance = covariance(t_large, p_large)
        np_cov = np.cov(t_large, p_large, bias=True)[0][1]  # https://stackoverflow.com/a/39098306
        assert np.allclose(new_covariance, np_cov)

        new_covariance = covariance(t_nan, p_nan)
        t_nan_, p_nan_ = maybe_treat_arrays(True, t_nan, p_nan, 'regression', remove_nan=True)
        np_cov = np.cov(t_nan_, p_nan_, bias=True)[0][1]  # https://stackoverflow.com/a/39098306
        assert np.allclose(new_covariance, np_cov)

        new_covariance = covariance(t_neg, p_neg)
        np_cov = np.cov(t_neg, p_neg, bias=True)[0][1]  # https://stackoverflow.com/a/39098306
        assert np.allclose(new_covariance, np_cov)
        return

    def test_brier_score(self):
        # new_brier_score = brier_score(t11, p11)
        # assert np.allclose(new_brier_score, 0.0014540110400519878)
        return

    def test_bic(self):
        new_bic = bic(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_bic, -177.2107529924996)
        return

    def test_sse(self):
        new_sse = sse(t11, p11)
        np_sse = ((t11-p11)**2).sum()  # https://stackoverflow.com/a/2284634
        assert np.allclose(new_sse, np_sse)

        new_sse = sse(t_large, p_large)
        np_sse = ((t_large-p_large)**2).sum()  # https://stackoverflow.com/a/2284634
        assert np.allclose(new_sse, np_sse)

        new_sse = sse(t_nan, p_nan)
        t_nan_, p_nan_ = maybe_treat_arrays(True, t_nan, p_nan, 'regression', remove_nan=True)
        np_sse = ((t_nan_ - p_nan_)**2).sum()  # https://stackoverflow.com/a/2284634
        assert np.allclose(new_sse, np_sse)

        new_sse = sse(t_neg, p_neg)
        np_sse = ((t_neg - p_neg)**2).sum()  # https://stackoverflow.com/a/2284634
        assert np.allclose(new_sse, np_sse)
        return

    def test_amemiya_pred_criterion(self):
        new_amemiya_pred_criterion = amemiya_pred_criterion(t11, p11)
        # Equation taken from https://www.sfu.ca/sasdoc/sashtml/ets/chap30/sect19.htm#:~:text=Amemiya
        assert np.allclose(new_amemiya_pred_criterion, 0.16560355579355351)
        return

    def test_amemiya_adj_r2(self):
        new_amemiya_adj_r2 = amemiya_adj_r2(t11, p11)
        # equation taken from https://www.sfu.ca/sasdoc/sashtml/ets/chap30/sect19.htm#:~:text=Amemiya
        assert np.allclose(new_amemiya_adj_r2, -0.030274536738060798)
        return

    def test_aitchison(self):
        new_aitchison = aitchison(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_aitchison, 16.326288844358846)
        return

    def test_aic(self):
        new_aic = aic(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_aic, -179.8159231784877)
        return

    def test_acc(self):
        new_acc = acc(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_acc, 0.0179208383645756)
        return

    def test_cronbach_alpha(self):
        new_cronbach_alpha = cronbach_alpha(t11, p11)
        # https://stackoverflow.com/a/20799687
        assert np.allclose(new_cronbach_alpha, 0.03555058748735895)
        return

    def test_cosine_similarity(self):
        new_cosine_similarity = cosine_similarity(t11, p11)
        sklearn_cos_sim_val = sklearn_cos_sim(t11.reshape(1,-1), p11.reshape(1,-1))[0].item()
        assert np.allclose(new_cosine_similarity, sklearn_cos_sim_val)

        new_cosine_similarity = cosine_similarity(t_large, p_large)
        sklearn_cos_sim_val = sklearn_cos_sim(t_large.reshape(1,-1), p_large.reshape(1,-1))[0].item()
        assert np.allclose(new_cosine_similarity, sklearn_cos_sim_val)

        new_cosine_similarity = cosine_similarity(t_nan, p_nan)
        # todo sklearn_cos_sim does not allow nan
        #sklearn_cos_sim_val = sklearn_cos_sim(t_nan.reshape(1,-1), p_nan.reshape(1,-1))[0].item()
        #assert np.allclose(new_cosine_similarity, sklearn_cos_sim_val)

        new_cosine_similarity = cosine_similarity(t_neg, p_neg)
        sklearn_cos_sim_val = sklearn_cos_sim(t_neg.reshape(1,-1), p_neg.reshape(1,-1))[0].item()
        assert np.allclose(new_cosine_similarity, sklearn_cos_sim_val)
        return

    def test_decomposed_mse(self):
        new_decomposed_mse = decomposed_mse(t11, p11)
        # equation 24 in https://doi.org/10.2134/agronj2000.922345x
        assert np.allclose(new_decomposed_mse, 0.1623242774610079)
        return

    def test_euclid_distance(self):
        new_euclid_distance = euclid_distance(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_euclid_distance, 4.028948714751875)
        return

    def test_exp_var_score_cls(self):
        new_exp_var_score = metrics.exp_var_score()
        sk_exp_var_scr= explained_variance_score(t11, p11)
        assert np.allclose(new_exp_var_score, sk_exp_var_scr)

        new_exp_var_score = metrics_large.exp_var_score()
        sk_exp_var_scr= explained_variance_score(t_large, p_large)
        assert np.allclose(new_exp_var_score, sk_exp_var_scr)

        new_exp_var_score = metrics_nan.exp_var_score()
        # sk_exp_var_scr= explained_variance_score(t_nan, p_nan)
        # assert np.allclose(new_exp_var_score, sk_exp_var_scr)

        new_exp_var_score = metrics_neg.exp_var_score()
        sk_exp_var_scr= explained_variance_score(t_neg, p_neg)
        assert np.allclose(new_exp_var_score, sk_exp_var_scr)
        return

    def test_exp_var_score_func(self):
        new_exp_var_score = sm_exp_var_score(t11, p11)
        sk_exp_var_scr= explained_variance_score(t11, p11)
        assert np.allclose(new_exp_var_score, sk_exp_var_scr)

        new_exp_var_score = sm_exp_var_score(t_large, p_large)
        sk_exp_var_scr= explained_variance_score(t_large, p_large)
        assert np.allclose(new_exp_var_score, sk_exp_var_scr)

        new_exp_var_score = sm_exp_var_score(t_nan, p_nan)
        # sk_exp_var_scr= explained_variance_score(t_nan, p_nan)
        # assert np.allclose(new_exp_var_score, sk_exp_var_scr)

        new_exp_var_score = sm_exp_var_score(t_neg, p_neg)
        sk_exp_var_scr= explained_variance_score(t_neg, p_neg)
        assert np.allclose(new_exp_var_score, sk_exp_var_scr)
        return

    def test_expanded_uncertainty(self):
        new_expanded_uncertainty = expanded_uncertainty(t11, p11)
        # the value is calculated by using equations from following two references
        # https://doi.org/10.1016/j.rser.2014.07.117
        # https://doi.org/10.1016/j.enconman.2015.03.067
        assert np.allclose(new_expanded_uncertainty, 1.1089293648532548)
        return

    def test_fdc_fhv(self):
        new_fdc_fhv = fdc_fhv(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_fdc_fhv, 1.5933766138626262)
        return

    def test_fdc_flv(self):
        new_fdc_flv = fdc_flv(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_fdc_flv, 32.64817835760714)
        return

    def test_gmean_diff(self):
        new_gmean_diff = gmean_diff(t11, p11)
        # todo : find reference for this
        # assert np.allclose(new_gmean_diff, 1.0537636718549144)
        return

    def test_gmrae(self):
        new_gmrae = gmrae(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_gmrae, 0.79938390310645)
        return

    def test_calculate_hydro_metrics(self):
        out = calculate_hydro_metrics(t11, p11)
        assert isinstance(out, dict)
        assert len(out) > 1
        return

    def test_JS(self):
        new_JS = JS(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_JS, 7.275875413762115)
        return

    def test_kendaull_tau(self):
        new_kendaull_tau = kendaull_tau(t11, p11)
        assert np.allclose(new_kendaull_tau, kendalltau(t11, p11)[0])

        new_kendaull_tau = kendaull_tau(t_large, p_large)
        assert np.allclose(new_kendaull_tau, kendalltau(t_large, p_large)[0])

        new_kendaull_tau = kendaull_tau(t_nan, p_nan)
        t_nan_, p_nan_ = maybe_treat_arrays(True, t_nan, p_nan, 'regression', remove_nan=True)
        assert np.allclose(new_kendaull_tau, kendalltau(t_nan_, p_nan_)[0])

        new_kendaull_tau = kendaull_tau(t_neg, p_neg)
        assert np.allclose(new_kendaull_tau, kendalltau(t_neg, p_neg)[0])
        return

    def test_kgeprime_c2m(self):
        new_kgeprime_c2m = kgeprime_bound(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_kgeprime_c2m, 0.0023118217819402547)
        return

    def test_kgenp_bound(self):
        new_kgenp_bound = kgenp_bound(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_kgenp_bound, -0.00334814)
        return

    def test_kl_sym(self):
        new_kl_sym = kl_sym(t11, p11)
        assert np.allclose(new_kl_sym, 40.219282596783955)
        return

    def test_lm_index(self):
        new_lm_index = lm_index(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_lm_index, -0.32879454094431804)
        return

    def test_maape(self):
        new_maape = maape(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_maape, 0.5828454707567975)
        return

    def test_mbrae(self):
        new_mbrae = mbrae(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_mbrae, 0.46659593775205116)
        return

    def test_max_error_cls(self):
        new_max_error = metrics.max_error()
        sk_max_error= max_error(t11, p11)
        assert np.allclose(new_max_error, sk_max_error)

        new_max_error = metrics_large.max_error()
        sk_max_error= max_error(t_large, p_large)
        assert np.allclose(new_max_error, sk_max_error)

        new_max_error = metrics_nan.max_error()
        # sk_max_error= max_error(t_nan, p_nan)
        # assert np.allclose(new_max_error, sk_max_error)

        new_max_error = metrics_neg.max_error()
        sk_max_error= max_error(t_neg, p_neg)
        assert np.allclose(new_max_error, sk_max_error)
        return

    def test_max_error_func(self):
        new_max_error = sm_max_error(t11, p11)
        sk_max_error= max_error(t11, p11)
        assert np.allclose(new_max_error, sk_max_error)

        new_max_error = sm_max_error(t_large, p_large)
        sk_max_error= max_error(t_large, p_large)
        assert np.allclose(new_max_error, sk_max_error)

        new_max_error = sm_max_error(t_nan, p_nan)
        # sk_max_error= max_error(t_nan, p_nan)
        # assert np.allclose(new_max_error, sk_max_error)

        new_max_error = sm_max_error(t_neg, p_neg)
        sk_max_error= max_error(t_neg, p_neg)
        assert np.allclose(new_max_error, sk_max_error)
        return

    def test_mb_r(self):
        new_mb_r = mb_r(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_mb_r, 0.04444743269492335)
        return

    def test_mda(self):
        new_mda = mda(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_mda, 0.696969696969697)
        return

    def test_mde(self):
        new_mde = mde(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_mde, 0.0313854202641316)
        return

    def test_mdape(self):
        new_mdape = mdape(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_mdape, 51.3246349701827)
        return

    def test_mdrae(self):
        new_mdrae = mdrae(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_mdrae, 0.9086455067666214)
        return

    def test_me(self):
        new_me = me(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_me, -0.06738857779448111)
        return

    def test_mean_bias_error(self):
        sm_mbe = mean_bias_error(t11, p11)
        np_mbe = np.mean(t11 - p11) # https://stackoverflow.com/q/59935155
        assert np.allclose(sm_mbe, np_mbe)

        sm_mbe = mean_bias_error(t_large, p_large)
        np_mbe = np.mean(t_large - p_large) # https://stackoverflow.com/q/59935155
        assert np.allclose(sm_mbe, np_mbe)

        sm_mbe = mean_bias_error(t_nan, p_nan)
        t_nan_, p_nan_ = maybe_treat_arrays(True, t_nan, p_nan, 'regression', remove_nan=True)
        np_mbe = np.mean(t_nan_ - p_nan_) # https://stackoverflow.com/q/59935155
        assert np.allclose(sm_mbe, np_mbe)

        sm_mbe = mean_bias_error(t_neg, p_neg)
        np_mbe = np.mean(t_neg - p_neg) # https://stackoverflow.com/q/59935155
        assert np.allclose(sm_mbe, np_mbe)
        return

    def test_mean_var(self):
        new_mean_var = mean_var(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_mean_var, 0.07449144510570738)
        return

    def test_mean_poisson_deviance(self):
        new_mean_poisson_deviance = sm_mean_poisson_deviance(t11, p11)
        assert np.allclose(new_mean_poisson_deviance, mean_poisson_deviance(t11, p11))

        new_mean_poisson_deviance = sm_mean_poisson_deviance(t_large, p_large)
        assert np.allclose(new_mean_poisson_deviance, mean_poisson_deviance(t_large, p_large))

        new_mean_poisson_deviance = sm_mean_poisson_deviance(t_nan, p_nan)
        t_nan_, p_nan_ = maybe_treat_arrays(True, t_nan, p_nan, 'regression', remove_nan=True)
        assert np.allclose(new_mean_poisson_deviance, mean_poisson_deviance(t_nan_, p_nan_))

        # new_mean_poisson_deviance = sm_mean_poisson_deviance(t_neg, p_neg)
        # assert np.allclose(new_mean_poisson_deviance, mean_poisson_deviance(t_neg, p_neg))
        return

    def test_mean_gamma_deviance(self):
        new_mean_gamma_deviance = sm_mean_gamma_deviance(t11, p11)
        assert np.allclose(new_mean_gamma_deviance, mean_gamma_deviance(t11, p11))

        new_mean_gamma_deviance = sm_mean_gamma_deviance(t_large, p_large)
        assert np.allclose(new_mean_gamma_deviance, mean_gamma_deviance(t_large, p_large))

        new_mean_gamma_deviance = sm_mean_gamma_deviance(t_nan, p_nan)
        t_nan_, p_nan_ = maybe_treat_arrays(True, t_nan, p_nan, 'regression', remove_nan=True)
        assert np.allclose(new_mean_gamma_deviance, mean_gamma_deviance(t_nan_, p_nan_))

        # new_mean_gamma_deviance = sm_mean_gamma_deviance(t_neg, p_neg)
        # assert np.allclose(new_mean_gamma_deviance, mean_gamma_deviance(t_neg, p_neg))
        return

    def test_median_abs_error_cls(self):
        new_median_abs_error = metrics.median_abs_error()
        sk_median_abs_error= median_absolute_error(t11, p11)
        assert np.allclose(new_median_abs_error, sk_median_abs_error)

        new_median_abs_error = metrics_large.median_abs_error()
        sk_median_abs_error= median_absolute_error(t_large, p_large)
        assert np.allclose(new_median_abs_error, sk_median_abs_error)

        new_median_abs_error = metrics_nan.median_abs_error()
        t_nan_, p_nan_ = maybe_treat_arrays(True, t_nan, p_nan, 'regression', remove_nan=True)
        sk_median_abs_error = median_absolute_error(t_nan_, p_nan_)
        assert np.allclose(new_median_abs_error, sk_median_abs_error)

        new_median_abs_error = metrics_neg.median_abs_error()
        sk_median_abs_error= median_absolute_error(t_neg, p_neg)
        assert np.allclose(new_median_abs_error, sk_median_abs_error)
        return

    def test_median_abs_error_func(self):
        new_median_abs_error = sm_median_abs_error(t11, p11)
        sk_median_abs_error= median_absolute_error(t11, p11)
        assert np.allclose(new_median_abs_error, sk_median_abs_error)

        new_median_abs_error = sm_median_abs_error(t_large, p_large)
        sk_median_abs_error= median_absolute_error(t_large, p_large)
        assert np.allclose(new_median_abs_error, sk_median_abs_error)

        new_median_abs_error = sm_median_abs_error(t_nan, p_nan)
        t_nan_, p_nan_ = maybe_treat_arrays(True, t_nan, p_nan, 'regression', remove_nan=True)
        sk_median_abs_error = median_absolute_error(t_nan_, p_nan_)
        assert np.allclose(new_median_abs_error, sk_median_abs_error)

        new_median_abs_error = sm_median_abs_error(t_neg, p_neg)
        sk_median_abs_error= median_absolute_error(t_neg, p_neg)
        assert np.allclose(new_median_abs_error, sk_median_abs_error)
        return

    def test_mle(self):
        new_mle = mle(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_mle, 0.0438958324374804)
        return

    def test_mod_agreement_index(self):
        new_mod_agreement_index = mod_agreement_index(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_mod_agreement_index, 0.36018092524466827)
        return

    def test_mpe(self):
        new_mpe = mpe(t11, p11)
        assert np.allclose(new_mpe, -4220.843064537674)
        return

    def test_mrae_new(self):
        new_mrae = mrae(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_mrae, 2.5711621568850163)
        return

    def test_norm_euclid_distance(self):
        new_norm_euclid_distance = norm_euclid_distance(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_norm_euclid_distance, 7.338597737626875)
        return

    def test_nrmse_range(self):
        new_nrmse_range = nrmse_range(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_nrmse_range, 0.4081874143525102)
        return

    def test_nrmse_ipercentile(self):
        new_nrmse_ipercentile = nrmse_ipercentile(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_nrmse_ipercentile, 0.8187123709758822)
        return

    def test_nrmse_mean(self):
        new_nrmse_mean = nrmse_mean(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_nrmse_mean, 0.790064026354788)
        return

    def test_norm_ae(self):
        new_norm_ae = norm_ae(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_norm_ae, 0.5551510970200795)
        return

    def test_norm_ape(self):
        new_norm_ape = norm_ape(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_norm_ape, 40406.322788120626)
        return

    def test_log_prob(self):
        new_log_prob = log_prob(t11, p11)
        assert np.allclose(new_log_prob, -32.128996820201635)
        return

    def test_rmdspe(self):
        new_rmdspe = rmdspe(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_rmdspe, 51.33222853161395)
        return

    def test_rse(self):
        new_rse = rse(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_rse, 2.0683722517498735)
        return

    def test_rrse(self):
        new_rrse = rrse(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_rrse, 1.4381836641228662)
        return

    def test_rae(self):
        new_rae = rae(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_rae, 1.3287945409387383)
        return

    def test_ref_agreement_index(self):
        new_ref_agreement_index = ref_agreement_index(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_ref_agreement_index, 0.335602729527841)
        return

    def test_rel_agreement_index(self):
        new_rel_agreement_index = rel_agreement_index(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_rel_agreement_index, -139396.49261170527)
        return

    def test_relative_rmse(self):
        new_relative_rmse = relative_rmse(t11, p11)
        # todo
        assert np.allclose(new_relative_rmse, 0.790064026354788)
        return

    def test_rmspe(self):
        new_rmspe = rmspe(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_rmspe, 39525.28325496917)
        return

    def test_rsr(self):
        # https://rdrr.io/cran/hydroGOF/man/rsr.html
        new_rsr = rsr(np.arange(1, 10), np.arange(1, 10))
        assert np.allclose(new_rsr, 0.0)

        obs = np.arange(1, 11)
        sim = np.arange(2, 12)
        new_rsr = rsr(obs, sim)
        assert np.allclose(new_rsr, 0.3302891)

        # copy paste these values to R and run rsr(sim, obs)
        t = np.array([160, 112, 129, 116, 100,  68, 103, 87, 70, 69])
        p = np.array([169.43952, 121.76982, 140.55871, 126.07051, 110.12929, 79.71506, 113.46092, 95.73494, 79.31315, 78.55434])
        assert np.allclose(rsr(t, p), 0.3417515)
        return

    def test_rmsse(self):
        new_rmsse = rmsse(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_rmsse, 1.2234619716320643)
        return

    def test_sa(self):
        new_sa = sa(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_sa, 0.6618474080345743)
        return

    def test_sc(self):
        new_sc = sc(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_sc, 1.5526934811208075)
        return

    def test_smape(self):
        new_smape = smape(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_smape, 70.28826490215243)
        return

    def test_smdape(self):
        new_smdape = smdape(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_smdape, 0.5999121382638821)
        return

    def test_sid(self):
        new_sid = sid(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_sid, 43.71192101756139)
        return

    def test_skill_score_murphy(self):
        new_skill_score_murphy = skill_score_murphy(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_skill_score_murphy, -1.0476885292323743)
        return

    def test_std_ratio(self):
        new_std_ratio = std_ratio(t11, p11)
        assert np.allclose(new_std_ratio, 1.0235046034233621)
        return

    def test_umbrae(self):
        new_umbrae = umbrae(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_umbrae, 0.8747513766311694)
        return

    def test_ve(self):
        new_ve = ve(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_ve, 0.3794625949621073)
        return

    def test_volume_error(self):
        new_volume_error = volume_error(t11, p11)
        assert np.allclose(new_volume_error, 0.13214685733697532)
        return

    def test_wape(self):
        new_wape = wape(t11, p11)
        np_wape = np.abs(t11 - p11).sum() / t11.sum() # https://stackoverflow.com/a/68531393
        assert np.allclose(new_wape, np_wape)

        new_wape = wape(t_large, p_large)
        np_wape = np.abs(t_large - p_large).sum() / t_large.sum() # https://stackoverflow.com/a/68531393
        assert np.allclose(new_wape, np_wape)

        new_wape = wape(t_nan, p_nan)
        t_nan_, p_nan_ = maybe_treat_arrays(True, t_nan, p_nan, 'regression', remove_nan=True)
        np_wape = np.abs(t_nan_ - p_nan_).sum() / t_nan_.sum() # https://stackoverflow.com/a/68531393
        assert np.allclose(new_wape, np_wape)

        new_wape = wape(t_neg, p_neg)
        np_wape = np.abs(t_neg - p_neg).sum() / t_neg.sum() # https://stackoverflow.com/a/68531393
        assert np.allclose(new_wape, np_wape)
        return

    def test_watt_m(self):
        new_watt_m = watt_m(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_watt_m, 0.017290316806567577)
        return

    def test_wmape(self):
        new_wmape = wmape(t11, p11)
        # ref : reference_for_seqmetric_tests.ipynb
        assert np.allclose(new_wmape, 0.6205374050378927)
        return

    def test_vr(self):
        new_vr = variability_ratio(t11, p11)
        assert np.allclose(new_vr, 0.9040387267698112)
        return

    def test_concordance_corr_coef_cls(self):
        # taken from https://nirpyresearch.com/concordance-correlation-coefficient/
        new_concordance_corr_coef = metrics.concordance_corr_coef()
        self.assertAlmostEqual(new_concordance_corr_coef, 0.017599598191033003)
        return

    def test_concordance_corr_coef_func(self):
        # taken from https://nirpyresearch.com/concordance-correlation-coefficient/
        new_concordance_corr_coef = sm_concordance_corr_coef(t11, p11)
        self.assertAlmostEqual(new_concordance_corr_coef, 0.017599598191033003)
        return

    def test_mre(self):
        # ref : reference_for_seqmetric_tests.ipynb
        self.assertAlmostEqual(metrics.mre(), 0.5101139564068491)
        return


# class test_torch_metrics(unittest.TestCase):

#     def test_critical_success_index_cls(self):
#         try:
#             import torch
#             from torchmetrics.regression import CriticalSuccessIndex
#         except (ModuleNotFoundError, ImportError):
#             print('Cant run test_torch_tensor')
#             torch = None

#         if torch is not None:
#             new_critical_success_index = metrics.critical_success_index()
#             csi = CriticalSuccessIndex(0.5)
#             torch_csi = csi(torch.tensor(p11), torch.tensor(t11))
#             self.assertAlmostEqual(new_critical_success_index, torch_csi)
#         return
#     def test_critical_success_index_func(self):
#         try:
#             import torch
#             from torchmetrics.regression import CriticalSuccessIndex
#         except (ModuleNotFoundError, ImportError):
#             print('Cant run test_torch_tensor')
#             torch = None
#         if torch is not None:
#             new_critical_success_index = sm_critical_success_index(t11, p11)
#             csi = CriticalSuccessIndex(0.5)
#             torch_csi = csi(torch.tensor(p11), torch.tensor(t11))
#             self.assertAlmostEqual(new_critical_success_index, torch_csi)
#         return

#     def test_kl_divergence_cls(self):
#         try:
#             import torch
#             from torchmetrics.regression import KLDivergence
#         except (ModuleNotFoundError, ImportError):
#             print('Cant run test_torch_tensor')
#             torch = None
#         if torch is not None:
#             new_kl_divergence = metrics.kl_divergence()
#             kl_div = KLDivergence()
#             torch_kl_div = kl_div(torch.tensor(p11).reshape(1,-1), torch.tensor(t11).reshape(1,-1))
#             self.assertAlmostEqual(new_kl_divergence, torch_kl_div.numpy().item())
#         return

#     def test_kl_divergence_func(self):
#         try:
#             import torch
#             from torchmetrics.regression import KLDivergence
#         except (ModuleNotFoundError, ImportError):
#             print('Cant run test_torch_tensor')
#             torch = None
#         if torch is not None:
#             new_kl_divergence = sm_kl_divergence(t11, p11)
#             kl_div = KLDivergence()
#             torch_kl_div = kl_div(torch.tensor(p11).reshape(1,-1), torch.tensor(t11).reshape(1,-1))
#             self.assertAlmostEqual(new_kl_divergence, torch_kl_div.numpy().item())
#         return

#     def test_log_cosh_error_cls(self):
#         try:
#             import torch
#             from torchmetrics.regression import LogCoshError
#         except (ModuleNotFoundError, ImportError):
#             print('Cant run test_torch_tensor')
#             torch = None
#         if torch is not None:
#             new_log_cosh_error = metrics.log_cosh_error()
#             lg_cosh_err = LogCoshError()
#             torch_lg_cosh_err = lg_cosh_err(torch.tensor(p11), torch.tensor(t11))
#             self.assertAlmostEqual(new_log_cosh_error, torch_lg_cosh_err)
#         return

#     def test_log_cosh_error_func(self):
#         try:
#             import torch
#             from torchmetrics.regression import LogCoshError
#         except (ModuleNotFoundError, ImportError):
#             print('Cant run test_torch_tensor')
#             torch = None
#         if torch is not None:
#             new_log_cosh_error = sm_log_cosh_error(t11, p11)
#             lg_cosh_err = LogCoshError()
#             torch_lg_cosh_err = lg_cosh_err(torch.tensor(p11), torch.tensor(t11))
#             self.assertAlmostEqual(new_log_cosh_error, torch_lg_cosh_err)
#         return


#     def test_minkowski_distance_cls(self):
#         try:
#             import torch
#             from torchmetrics.regression import MinkowskiDistance
#         except (ModuleNotFoundError, ImportError):
#             print('Cant run test_torch_tensor')
#             torch = None
#         if torch is not None:
#             new_minkowski_distance = metrics.minkowski_distance()
#             mink_dist = MinkowskiDistance(1)
#             torch_mink_dist = mink_dist(torch.tensor(p11), torch.tensor(t11))
#             self.assertAlmostEqual(new_minkowski_distance, torch_mink_dist)
#         return

#     def test_minkowski_distance_func(self):
#         try:
#             import torch
#             from torchmetrics.regression import MinkowskiDistance
#         except (ModuleNotFoundError, ImportError):
#             print('Cant run test_torch_tensor')
#             torch = None
#         if torch is not None:
#             new_minkowski_distance = sm_minkowski_distance(t11, p11)
#             mink_dist = MinkowskiDistance(1)
#             torch_mink_dist = mink_dist(torch.tensor(p11), torch.tensor(t11))
#             self.assertAlmostEqual(new_minkowski_distance, torch_mink_dist)
#         return

#     def test_tweedie_deviance_score_cls(self):
#         try:
#             import torch
#             from torchmetrics.regression import TweedieDevianceScore
#         except (ModuleNotFoundError, ImportError):
#             print('Cant run test_torch_tensor')
#             torch = None
#         if torch is not None:
#             new_tweedie_deviance_score = metrics.tweedie_deviance_score()
#             tw_dev_score = TweedieDevianceScore(0)
#             torch_tw_dev_score = tw_dev_score(torch.tensor(p11), torch.tensor(t11))
#             self.assertAlmostEqual(new_tweedie_deviance_score, torch_tw_dev_score)
#         return

#     def test_tweedie_deviance_score_func(self):
#         try:
#             import torch
#             from torchmetrics.regression import TweedieDevianceScore
#         except (ModuleNotFoundError, ImportError):
#             print('Cant run test_torch_tensor')
#             torch = None
#         if torch is not None:
#             new_tweedie_deviance_score = sm_tweedie_deviance_score(t11, p11)
#             tw_dev_score = TweedieDevianceScore(0)
#             torch_tw_dev_score = tw_dev_score(torch.tensor(p11), torch.tensor(t11))
#             self.assertAlmostEqual(new_tweedie_deviance_score, torch_tw_dev_score)
#         return


# class TestTreatment(unittest.TestCase):
#     random_state = np.random.RandomState(seed=92)

#     t = random_state.random(20)
#     p = random_state.random(20)
    
#     def test_nan_in_true(self):
#         t = self.t.copy()
#         t[0] = np.nan
#         assert not np.isnan(kge(t, self.p))
#         assert np.isnan(kge(t, self.p, remove_nan=False))
#         return
    
#     def test_nan_in_pred(self):
#         p = self.p.copy()
#         p[0] = np.nan
#         assert not np.isnan(kge(self.t, p))
#         assert np.isnan(kge(self.t, p, remove_nan=False))
#         return
    
#     def test_nan_in_true_and_pred(self):
#         t_ = self.t.copy()
#         t_[0] = np.nan
#         p_ = self.p.copy()
#         p_[1] = np.nan
#         assert not np.isnan(kge(t_, p_))
#         assert np.isnan(kge(t_, p_, remove_nan=False))
#         return
    
#     def test_inf_in_true(self):
#         t = self.t.copy()
#         t[0] = np.inf
#         assert not np.isnan(kge(t, self.p))
#         assert np.isnan(kge(t, self.p, remove_inf=False))
#         return
    
#     def test_inf_in_pred(self):
#         p = self.p.copy()
#         p[0] = np.inf
#         assert not np.isnan(kge(self.t, p))
#         assert np.isnan(kge(self.t, p, remove_inf=False))
#         return
    
#     def test_inf_in_true_and_pred(self):
#         t_ = self.t.copy()
#         t_[0] = np.inf
#         p_ = self.p.copy()
#         p_[1] = np.inf
#         assert not np.isnan(kge(t_, p_))
#         assert np.isnan(kge(t_, p_, remove_inf=False))
#         return
    
#     def test_replace_nan_in_true(self):
#         t_ = self.t.copy()
#         t_[0] = np.nan
#         assert not np.isnan(kge(t_, self.p, replace_nan=1.0))
#         return

if __name__ == "__main__":
    unittest.main()
