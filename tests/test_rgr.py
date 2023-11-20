import os
import unittest
import site   # so that ai4water directory is in path

ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(ai4_dir)

import numpy as np
import pandas as pd

from SeqMetrics import r2
from SeqMetrics import RegressionMetrics
from SeqMetrics import nse
from SeqMetrics import nse_alpha
from SeqMetrics import nse_beta
from SeqMetrics import nse_mod
from SeqMetrics import nse_rel
from SeqMetrics import nse_bound
from SeqMetrics import r2_score
from SeqMetrics import adjusted_r2
from SeqMetrics import kge
from SeqMetrics import kge_bound
from SeqMetrics import kge_mod
from SeqMetrics import kge_np
from SeqMetrics import log_nse
from SeqMetrics import corr_coeff
from SeqMetrics import rmse
from SeqMetrics import rmsle
from SeqMetrics import mape
from SeqMetrics import nrmse
from SeqMetrics import pbias
from SeqMetrics import bias
from SeqMetrics import med_seq_error
from SeqMetrics import mae
from SeqMetrics import abs_pbias
from SeqMetrics import gmae
from SeqMetrics import inrse
from SeqMetrics import irmse
from SeqMetrics import mase
from SeqMetrics import mare
from SeqMetrics import msle
from SeqMetrics import covariance
from SeqMetrics import brier_score
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
from SeqMetrics import exp_var_score
from SeqMetrics import expanded_uncertainty
from SeqMetrics import fdc_fhv
from SeqMetrics import fdc_flv
from SeqMetrics import gmean_diff
from SeqMetrics import gmrae
from SeqMetrics import calculate_hydro_metrics
from SeqMetrics import JS
from SeqMetrics import kendaull_tau
from SeqMetrics import kgeprime_c2m
from SeqMetrics import kgenp_bound
from SeqMetrics import kl_sym
from SeqMetrics import lm_index
from SeqMetrics import maape
from SeqMetrics import mbe
from SeqMetrics import mbrae
from SeqMetrics import max_error
from SeqMetrics import mb_r
from SeqMetrics import mda
from SeqMetrics import mde
from SeqMetrics import mdape
from SeqMetrics import mdrae
from SeqMetrics import me
from SeqMetrics import mean_bias_error
from SeqMetrics import mean_var
from SeqMetrics import mean_poisson_deviance
from SeqMetrics import mean_gamma_deviance
from SeqMetrics import median_abs_error
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

t = np.random.random((20, 1))
p = np.random.random((20, 1))

ts = pd.Series(np.random.random((20, )))
ps = pd.Series(np.random.random((20, )))
tdf = pd.DataFrame(np.random.random((20, 1)))
pdf = pd.DataFrame(np.random.random((20, 1)))

er = RegressionMetrics(t, p)
ers = RegressionMetrics(ts, ps)
erdf = RegressionMetrics(tdf, pdf)

all_errors = er.calculate_all()

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

class test_errors(unittest.TestCase):

    def test_attrs(self):
        for _attr in not_metrics:
            assert _attr not in er.all_methods

    def test_calculate_all(self):
        assert len(all_errors) > 100
        for er_name, er_val in all_errors.items():
            if er_val is not None:
                er_val = getattr(er, er_name)()
                self.assertEqual(er_val.__class__.__name__, 'float', f'{er_name} is {er_val}')
        return

    def test_calculate_all_series(self):
        series_errors = ers.calculate_all()
        assert len(series_errors) > 100
        for er_name, er_val in series_errors.items():
            if er_val is not None:
                er_val = getattr(ers, er_name)()
                self.assertEqual(er_val.__class__.__name__, 'float', f'{er_name} is {er_val}')
        return

    def test_calculate_all_df(self):
        df_errors = ers.calculate_all()
        assert len(df_errors) > 100
        for er_name, er_val in df_errors.items():
            if er_val is not None:
                er_val = getattr(erdf, er_name)()
                self.assertEqual(er_val.__class__.__name__, 'float', f'{er_name} is {er_val}')
        return

    def test_mrae(self):
        assert er.mare() * 100.0 == er.mape()
        return

    def test_mare(self):
        # https://support.numxl.com/hc/en-us/articles/115001223363-MRAE-Mean-Relative-Absolute-Error
        data = np.array(
            [[-2.9, 	-2.95],
             [-2.83, 	-2.7],
             [-0.95, 	-1.00],
             [-0.88, 	-0.68],
             [1.21,	1.50],
             [-1.67, 	-1.00],
             [0.83, 	0.90],
             [-0.27, 	-0.37],
             [1.36, 	1.26],
             [-0.34, 	-0.54],
             [0.48, 	0.58],
             [-2.83, 	-2.13],
             [-0.95, 	-0.75],
             [-0.88, 	-0.89],
             [1.21, 	1.25],
             [-1.67, 	-1.65],
             [-2.99, 	-3.20],
             [1.24, 	1.29],
             [0.64, 	0.60]]
        )
        errs = RegressionMetrics(data[:, 0], data[:, 1])
        np.testing.assert_almost_equal(0.348, errs.mrae(), 2)
        assert errs.mare() * 100.0 == errs.mape()
        return

    def test_hydro_metrics(self):
        hydr_metrics = er.calculate_hydro_metrics()
        assert len(hydr_metrics) == len(er._hydro_metrics())
        return

    def test_minimal(self):
        minimal_metrics = er.calculate_minimal()
        assert len(minimal_metrics) == len(er._minimal())
        return

    def test_scale_dependent(self):
        minimal_metrics = er.calculate_scale_dependent_metrics()
        assert len(minimal_metrics) == len(er._scale_dependent_metrics())
        return

    def test_scale_independent(self):
        minimal_metrics = er.calculate_scale_independent_metrics()
        assert len(minimal_metrics) == len(er._scale_independent_metrics())
        return

    def test_r2(self):
        new_r2 = r2(t11, p11)
        assert np.allclose(new_r2, 0.0003276772244559177)
        return
    def test_nse(self):
        new_nse = nse(t11, p11)
        assert np.allclose(new_nse, -1.068372251749874)
        return
    def test_nse_alpha(self):
        new_nse_alpha = nse_alpha(t11, p11)
        assert np.allclose(new_nse_alpha, 1.0235046034233621)
        return
    def test_nse_beta(self):
        new_nse_beta = nse_beta(t11, p11)
        assert np.allclose(new_nse_beta, 0.2405519617999516)
        return
    def test_nse_mod(self):
        new_nse_mod = nse_mod(t11, p11)
        assert np.allclose(new_nse_mod, -0.32879454094431804)
        return
    def test_nse_rel(self):
        new_nse_rel = nse_rel(t11, p11)
        assert np.allclose(new_nse_rel, -517670.8159599439)
        return
    def test_nse_bound(self):
        new_nse_bound = nse_bound(t11, p11)
        assert np.allclose(new_nse_bound, -0.34818860428052295)
        return
    def test_r2_score(self):
        new_r2_score = r2_score(t11, p11)
        assert np.allclose(new_r2_score, -1.0683722517498735)
        return
    def test_adjusted_r2(self):
        new_adjusted_r2 = adjusted_r2(t11, p11)
        assert np.allclose(new_adjusted_r2, -0.009873060763049724)
        return
    def test_kge(self):
        new_kge = kge(t11, p11)
        assert np.allclose(new_kge, 0.008970625237195717)
        return
    def test_kge_bound(self):
        new_kge_bound = kge_bound(t11, p11)
        assert np.allclose(new_kge_bound, 0.0045055212900936776)
        return
    def test_kge_mod(self):
        new_kge_mod = kge_mod(t11, p11)
        assert np.allclose(new_kge_mod, 0.004612979178136856)
        return
    def test_kge_np(self):
        new_kge_np = kge_np(t11, p11)
        assert np.allclose(new_kge_np, -0.006266558719581594)
        return
    def test_log_nse(self):
        new_log_nse = log_nse(t11, p11)
        assert np.allclose(new_log_nse, 1.0)
        return
    def test_corr_coeff(self):
        new_corr_coeff = corr_coeff(t11, p11)
        assert np.allclose(new_corr_coeff, 0.018101856933914716)
        return
    def test_rmse(self):
        new_rmse = rmse(t11, p11)
        assert np.allclose(new_rmse, 0.40289487147518754)
        return
    def test_rmsle(self):
        new_rmsle = rmsle(t11, p11)
        assert np.allclose(new_rmsle, 0.276438581263699)
        return
    def test_mape(self):
        new_mape = mape(t11, p11)
        assert np.allclose(new_mape, 4259.236161487332)
        return
    def test_nrmse(self):
        new_nrmse = nrmse(t11, p11)
        assert np.allclose(new_nrmse, 0.4081874143525102)
        return
    def test_pbias(self):
        new_pbias = pbias(t11, p11)
        assert np.allclose(new_pbias, 13.214685733697532)
        return
    def test_bias(self):
        new_bias = bias(t11, p11)
        assert np.allclose(new_bias, -0.06738857779448111)
        return
    def test_med_seq_error(self):
        new_med_seq_error = med_seq_error(t11, p11)
        assert np.allclose(new_med_seq_error, 0.06731204476856545)
        return
    def test_mae(self):
        new_mae = mae(t11, p11)
        assert np.allclose(new_mae, 0.31644440160349424)
        return
    def test_abs_pbias(self):
        new_abs_pbias = abs_pbias(t11, p11)
        assert np.allclose(new_abs_pbias, 62.05374050378925)
        return
    def test_gmae(self):
        new_gmae = gmae(t11, p11)
        assert np.allclose(new_gmae, 0.19423992928498718)
        return
    def test_inrse(self):
        new_inrse = inrse(t11, p11)
        assert np.allclose(new_inrse, 1.4381836641228662)
        return
    def test_irmse(self):
        new_irmse = irmse(t11, p11)
        assert np.allclose(new_irmse, 0.9954807723243245)
        return
    def test_mase(self):
        new_mase = mase(t11, p11)
        assert np.allclose(new_mase, 0.9609397361653512)
        return
    def test_mare_new(self):
        new_mare = mare(t11, p11)
        assert np.allclose(new_mare, 42.59236161487332)
        return
    def test_msle(self):
        new_msle = msle(t11, p11)
        assert np.allclose(new_msle, 0.07641828921108672)
        return
    def test_covariance(self):
        new_covariance = covariance(t11, p11)
        assert np.allclose(new_covariance, 0.0014540110400519878)
        return
    def test_brier_score(self):
        # new_brier_score = brier_score(t11, p11)
        # assert np.allclose(new_brier_score, 0.0014540110400519878)
        return
    def test_bic(self):
        new_bic = bic(t11, p11)
        assert np.allclose(new_bic, -177.2107529924996)
        return
    def test_sse(self):
        new_sse = sse(t11, p11)
        assert np.allclose(new_sse, 16.23242774610079)
        return
    def test_amemiya_pred_criterion(self):
        new_amemiya_pred_criterion = amemiya_pred_criterion(t11, p11)
        assert np.allclose(new_amemiya_pred_criterion, 0.16560355579355351)
        return
    def test_amemiya_adj_r2(self):
        new_amemiya_adj_r2 = amemiya_adj_r2(t11, p11)
        assert np.allclose(new_amemiya_adj_r2, -0.030274536738060798)
        return
    def test_aitchison(self):
        new_aitchison = aitchison(t11, p11)
        assert np.allclose(new_aitchison, 16.326288844358846)
        return
    def test_aic(self):
        new_aic = aic(t11, p11)
        assert np.allclose(new_aic, -179.8159231784877)
        return
    def test_acc(self):
        new_acc = acc(t11, p11)
        assert np.allclose(new_acc, 0.0179208383645756)
        return
    def test_cronbach_alpha(self):
        new_cronbach_alpha = cronbach_alpha(t11, p11)
        assert np.allclose(new_cronbach_alpha, 0.03555058748735895)
        return
    def test_cosine_similarity(self):
        new_cosine_similarity = cosine_similarity(t11, p11)
        assert np.allclose(new_cosine_similarity, 0.7888582070548288)
        return
    def test_decomposed_mse(self):
        new_decomposed_mse = decomposed_mse(t11, p11)
        assert np.allclose(new_decomposed_mse, 0.1623242774610079)
        return
    def test_euclid_distance(self):
        new_euclid_distance = euclid_distance(t11, p11)
        assert np.allclose(new_euclid_distance, 4.028948714751875)
        return
    def test_exp_var_score(self):
        new_exp_var_score = exp_var_score(t11, p11)
        assert np.allclose(new_exp_var_score, -1.0105070054240683)
        return
    def test_expanded_uncertainty(self):
        new_expanded_uncertainty = expanded_uncertainty(t11, p11)
        assert np.allclose(new_expanded_uncertainty, 1.1089293648532548)
        return
    def test_fdc_fhv(self):
        new_fdc_fhv = fdc_fhv(t11, p11)
        assert np.allclose(new_fdc_fhv, 1.5933757966893547)
        return
    def test_fdc_flv(self):
        new_fdc_flv = fdc_flv(t11, p11)
        assert np.allclose(new_fdc_flv, 32.605250716215686)
        return
    def test_gmean_diff(self):
        new_gmean_diff = gmean_diff(t11, p11)
        assert np.allclose(new_gmean_diff, 1.0537636718549144)
        return
    def test_gmrae(self):
        new_gmrae = gmrae(t11, p11)
        assert np.allclose(new_gmrae, 0.79938390310645)
        return
    def test_calculate_hydro_metrics(self):
        out = calculate_hydro_metrics(t, p)
        assert isinstance(out, dict)
        assert len(out)> 1
        return
    def test_JS(self):
        new_JS = JS(t11, p11)
        assert np.allclose(new_JS, 7.275875413762115)
        return
    def test_kendaull_tau(self):
        new_kendaull_tau = kendaull_tau(t11, p11)
        assert np.allclose(new_kendaull_tau, 0.9952476412894299)
        return
    def test_kgeprime_c2m(self):
        new_kgeprime_c2m = kgeprime_c2m(t11, p11)
        assert np.allclose(new_kgeprime_c2m, 0.0023118217819402547)
        return
    def test_kgenp_bound(self):
        new_kgenp_bound = kgenp_bound(t11, p11)
        assert np.allclose(new_kgenp_bound, -0.003123492584943932)
        return
    def test_kl_sym(self):
        new_kl_sym = kl_sym(t11, p11)
        assert np.allclose(new_kl_sym, 40.219282596783955)
        return
    def test_lm_index(self):
        new_lm_index = lm_index(t11, p11)
        assert np.allclose(new_lm_index, -0.32879454094431804)
        return
    def test_maape(self):
        new_maape = maape(t11, p11)
        assert np.allclose(new_maape, 0.5828454707567975)
        return
    def test_mbe(self):
        new_mbe = mbe(t11, p11)
        assert np.allclose(new_mbe, -0.06738857779448111)
        return
    def test_mbrae(self):
        new_mbrae = mbrae(t11, p11)
        assert np.allclose(new_mbrae, 0.46659593775205116)
        return
    def test_max_error(self):
        new_max_error = max_error(t11, p11)
        assert np.allclose(new_max_error, 0.9192299717467063)
        return
    def test_mb_r(self):
        new_mb_r = mb_r(t11, p11)
        assert np.allclose(new_mb_r, 0.04444743269492335)
        return
    def test_mda(self):
        new_mda = mda(t11, p11)
        assert np.allclose(new_mda, 0.5252525252525253)
        return
    def test_mde(self):
        new_mde = mde(t11, p11)
        assert np.allclose(new_mde, 0.0313854202641316)
        return
    def test_mdape(self):
        new_mdape = mdape(t11, p11)
        assert np.allclose(new_mdape, 5132.46349701827)
        return
    def test_mdrae(self):
        new_mdrae = mdrae(t11, p11)
        assert np.allclose(new_mdrae, 0.9086455067666214)
        return
    def test_me(self):
        new_me = me(t11, p11)
        assert np.allclose(new_me, -0.06738857779448111)
        return
    def test_mean_bias_error(self):
        new_mean_bias_error = mean_bias_error(t11, p11)
        assert np.allclose(new_mean_bias_error, -0.06738857779448111)
        return
    def test_mean_var(self):
        new_mean_var = mean_var(t11, p11)
        assert np.allclose(new_mean_var, 0.07449144510570738)
        return
    def test_mean_poisson_deviance(self):
        new_mean_poisson_deviance = mean_poisson_deviance(t11, p11)
        assert np.allclose(new_mean_poisson_deviance, 0.4910207582066133)
        return
    def test_mean_gamma_deviance(self):
        new_mean_gamma_deviance = mean_gamma_deviance(t11, p11)
        assert np.allclose(new_mean_gamma_deviance, 11.533824019539743)
        return
    def test_median_abs_error(self):
        new_median_abs_error = median_abs_error(t11, p11)
        assert np.allclose(new_median_abs_error, 0.2594229386964548)
        return
    def test_mle(self):
        new_mle = mle(t11, p11)
        assert np.allclose(new_mle, 0.0438958324374804)
        return
    def test_mod_agreement_index(self):
        new_mod_agreement_index = mod_agreement_index(t11, p11)
        assert np.allclose(new_mod_agreement_index, 0.36018092524466827)
        return
    def test_mpe(self):
        new_mpe = mpe(t11, p11)
        assert np.allclose(new_mpe, -4220.843064537674)
        return
    def test_mrae_new(self):
        new_mrae = mrae(t11, p11)
        assert np.allclose(new_mrae, 2.5711621568850163)
        return
    def test_norm_euclid_distance(self):
        new_norm_euclid_distance = norm_euclid_distance(t11, p11)
        assert np.allclose(new_norm_euclid_distance, 7.338597737626875)
        return
    def test_nrmse_range(self):
        new_nrmse_range = nrmse_range(t11, p11)
        assert np.allclose(new_nrmse_range, 0.4081874143525102)
        return
    def test_nrmse_ipercentile(self):
        new_nrmse_ipercentile = nrmse_ipercentile(t11, p11)
        assert np.allclose(new_nrmse_ipercentile, 0.8187123709758822)
        return
    def test_nrmse_mean(self):
        new_nrmse_mean = nrmse_mean(t11, p11)
        assert np.allclose(new_nrmse_mean, 0.790064026354788)
        return
    def test_norm_ae(self):
        new_norm_ae = norm_ae(t11, p11)
        assert np.allclose(new_norm_ae, 0.5551510970200795)
        return
    def test_norm_ape(self):
        new_norm_ape = norm_ape(t11, p11)
        assert np.allclose(new_norm_ape, 40406.323186154805)
        return
    def test_log_prob(self):
        new_log_prob = log_prob(t11, p11)
        assert np.allclose(new_log_prob, -32.128996820201635)
        return
    def test_rmdspe(self):
        new_rmdspe = rmdspe(t11, p11)
        assert np.allclose(new_rmdspe, 5133.222853161395)
        return
    def test_rse(self):
        new_rse = rse(t11, p11)
        assert np.allclose(new_rse, 2.0683722517498735)
        return
    def test_rrse(self):
        new_rrse = rrse(t11, p11)
        assert np.allclose(new_rrse, 1.4381836641228662)
        return
    def test_rae(self):
        new_rae = rae(t11, p11)
        assert np.allclose(new_rae, 1.3287945409387383)
        return
    def test_ref_agreement_index(self):
        new_ref_agreement_index = ref_agreement_index(t11, p11)
        assert np.allclose(new_ref_agreement_index, 0.335602729527841)
        return
    def test_rel_agreement_index(self):
        new_rel_agreement_index = rel_agreement_index(t11, p11)
        assert np.allclose(new_rel_agreement_index, -139396.49261170527)
        return
    def test_relative_rmse(self):
        new_relative_rmse = relative_rmse(t11, p11)
        assert np.allclose(new_relative_rmse, 0.790064026354788)
        return
    def test_rmspe(self):
        new_rmspe = rmspe(t11, p11)
        assert np.allclose(new_rmspe, 395.25301985078426)
        return
    def test_rsr(self):
        new_rsr = rsr(t11, p11)
        assert np.allclose(new_rsr, 1.4381836641228662)
        return
    def test_rmsse(self):
        new_rmsse = rmsse(t11, p11)
        assert np.allclose(new_rmsse, 1.2234619716320643)
        return
    def test_sa(self):
        new_sa = sa(t11, p11)
        assert np.allclose(new_sa, 0.6618474080345743)
        return
    def test_sc(self):
        new_sc = sc(t11, p11)
        assert np.allclose(new_sc, 1.5526934811208075)
        return
    def test_smape(self):
        new_smape = smape(t11, p11)
        assert np.allclose(new_smape, 70.28826490215243)
        return
    def test_smdape(self):
        new_smdape = smdape(t11, p11)
        assert np.allclose(new_smdape, 0.5999121382638821)
        return
    def test_sid(self):
        new_sid = sid(t11, p11)
        assert np.allclose(new_sid, 43.71192101756139)
        return
    def test_skill_score_murphy(self):
        new_skill_score_murphy = skill_score_murphy(t11, p11)
        assert np.allclose(new_skill_score_murphy, -1.0476885292323743)
        return
    def test_std_ratio(self):
        new_std_ratio = std_ratio(t11, p11)
        assert np.allclose(new_std_ratio, 1.0235046034233621)
        return
    def test_umbrae(self):
        new_umbrae = umbrae(t11, p11)
        assert np.allclose(new_umbrae, 0.8747513766311694)
        return
    def test_ve(self):
        new_ve = ve(t11, p11)
        assert np.allclose(new_ve, 0.3794625949621073)
        return
    def test_volume_error(self):
        new_volume_error = volume_error(t11, p11)
        assert np.allclose(new_volume_error, 0.13214685733697532)
        return
    def test_wape(self):
        new_wape = wape(t11, p11)
        assert np.allclose(new_wape, 0.6205374050378927)
        return
    def test_watt_m(self):
        new_watt_m = watt_m(t11, p11)
        assert np.allclose(new_watt_m, 0.017290316806567577)
        return
    def test_wmape(self):
        new_wmape = wmape(t11, p11)
        assert np.allclose(new_wmape, 0.6205374050378927)
        return












if __name__ == "__main__":
    unittest.main()
