"""
SeqMetrics, the module to calculate performance related to tabular/structured and
sequential data. The values in a sequence are not necessarily related.
"""

from ._main import Metrics


from ._rgr import r2
from ._rgr import nse
from ._rgr import nse_alpha
from ._rgr import nse_beta
from ._rgr import nse_mod
from ._rgr import nse_rel
from ._rgr import nse_bound
from ._rgr import r2_score
from ._rgr import adjusted_r2
from ._rgr import kge
from ._rgr import kge_bound
from ._rgr import kge_mod
from ._rgr import kge_np
# from ._rgr import log_nse todo, find reference for the code
from ._rgr import corr_coeff
from ._rgr import rmse
from ._rgr import rmsle
from ._rgr import mape
from ._rgr import nrmse
from ._rgr import pbias
from ._rgr import bias
from ._rgr import med_seq_error
from ._rgr import mae
# from ._rgr import abs_pbias #  todo, find reference for the code
from ._rgr import gmae
from ._rgr import inrse
from ._rgr import irmse
from ._rgr import mase
from ._rgr import mare
from ._rgr import msle
from ._rgr import covariance
from ._rgr import brier_score
from ._rgr import bic
from ._rgr import sse
from ._rgr import amemiya_pred_criterion
from ._rgr import amemiya_adj_r2
from ._rgr import aitchison
from ._rgr import aic
from ._rgr import acc
from ._rgr import cronbach_alpha
from ._rgr import cosine_similarity
from ._rgr import decomposed_mse
from ._rgr import euclid_distance
from ._rgr import exp_var_score
from ._rgr import expanded_uncertainty
from ._rgr import fdc_fhv
from ._rgr import fdc_flv
from ._rgr import gmean_diff
from ._rgr import gmrae
from ._rgr import calculate_hydro_metrics
from ._rgr import JS
from ._rgr import kendall_tau
from ._rgr import kgeprime_bound
from ._rgr import kgenp_bound
from ._rgr import kl_sym
from ._rgr import lm_index
from ._rgr import maape
from ._rgr import mbrae
from ._rgr import max_error
from ._rgr import mb_r
from ._rgr import mda
from ._rgr import mde
from ._rgr import mdape
from ._rgr import mdrae
from ._rgr import me
from ._rgr import mean_bias_error
from ._rgr import mean_var
from ._rgr import mean_poisson_deviance
from ._rgr import mean_gamma_deviance
from ._rgr import median_abs_error
from ._rgr import mle
from ._rgr import mod_agreement_index
from ._rgr import mpe
from ._rgr import mrae
from ._rgr import norm_euclid_distance
from ._rgr import nrmse_range
from ._rgr import nrmse_ipercentile
from ._rgr import nrmse_mean
from ._rgr import norm_ae
from ._rgr import norm_ape
from ._rgr import log_prob
from ._rgr import rmdspe
from ._rgr import rse
from ._rgr import rrse
from ._rgr import rae
from ._rgr import ref_agreement_index
from ._rgr import rel_agreement_index
from ._rgr import relative_rmse
from ._rgr import rmspe
from ._rgr import rsr
from ._rgr import rmsse
from ._rgr import sa
from ._rgr import sc
from ._rgr import smape
from ._rgr import smdape
from ._rgr import sid
from ._rgr import skill_score_murphy
from ._rgr import std_ratio
from ._rgr import umbrae
from ._rgr import ve
from ._rgr import volume_error
from ._rgr import wape
from ._rgr import watt_m
from ._rgr import wmape
from ._rgr import spearmann_corr
from ._rgr import agreement_index
from ._rgr import centered_rms_dev
from ._rgr import mapd
from ._rgr import sga
from ._rgr import mse
from ._rgr import variability_ratio
from ._rgr import RegressionMetrics
from ._rgr import concordance_corr_coef
from ._rgr import critical_success_index
from ._rgr import kl_divergence
from ._rgr import log_cosh_error
from ._rgr import minkowski_distance
from ._rgr import tweedie_deviance_score
from ._rgr import mre
# from ._rgr import spearmann_rank_corr todo, find reference for the code
from ._rgr import mape_for_peaks
from ._rgr import legates_coeff_eff
from ._rgr import relative_error


from ._cls import f1_score
from ._cls import accuracy
from ._cls import precision
from ._cls import recall
from ._cls import balanced_accuracy
from ._cls import confusion_matrix
from ._cls import cross_entropy
from ._cls import error_rate
from ._cls import false_positive_rate
from ._cls import false_negative_rate
from ._cls import false_discovery_rate
from ._cls import false_omission_rate
from ._cls import f2_score
from ._cls import fowlkes_mallows_index
from ._cls import mathews_corr_coeff
from ._cls import negative_likelihood_ratio
from ._cls import negative_predictive_value
from ._cls import positive_likelihood_ratio
from ._cls import prevalence_threshold
from ._cls import specificity
from ._cls import youden_index
from ._cls import ClassificationMetrics


from .utils import plot_metrics

__version__ = '2.0.0'