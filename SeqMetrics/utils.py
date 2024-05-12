import os
import warnings
import itertools
from typing import Union
from types import FunctionType
from collections import OrderedDict

import scipy
import numpy as np
from scipy.special import xlogy
from scipy.stats import skew, kurtosis, variation, gmean

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    go = None


def take(st, en, d):
    keys = list(d.keys())[st:en]
    values = list(d.values())[st:en]

    return {k: v for k, v in zip(keys, values)}


def plot_metrics(
        metrics: dict,
        ranges: tuple = ((0.0, 1.0), (1.0, 10), (10, 1000)),
        exclude: list = None,
        plot_type: str = 'bar',
        max_metrics_per_fig: int = 15,
        show: bool = True,
        save: bool = False,
        save_path: str = '',
        **kwargs):
    """
    Plots the metrics given as dictionary as radial or bar plot between specified ranges.

    Arguments:
        metrics:
            dictionary whose keys are names are erros and values are error values.
        ranges:
            tuple of tuples defining range of errors to plot in one plot
        exclude:
            List of metrics to be excluded from plotting.
        max_metrics_per_fig:
            maximum number of metrics to show in one figure.
        plot_type:
            either of ``radial`` or ``bar``.
        show : If, then figure will be shown/drawn
        save:
            if True, the figure will be saved.
        save_path:
            if given, the figure will the saved at this location.
        kwargs:
            keyword arguments for plotting

    Examples:
        >>> import numpy as np
        >>> from SeqMetrics import RegressionMetrics
        >>> from SeqMetrics import plot_metrics
        >>> t = np.random.random((20, 1))
        >>> p = np.random.random((20, 1))
        >>> er = RegressionMetrics(t, p)
        >>> all_errors = er.calculate_all()
        >>> plot_metrics(all_errors, plot_type='bar', max_metrics_per_fig=50)
        >>> # or draw the radial plot
        >>> plot_metrics(all_errors, plot_type='radial', max_metrics_per_fig=50)
    ```
    """
    for idx, rng in enumerate(ranges):
        assert rng[1] > rng[0], f'For range {idx}, second value: {rng[1]} is not greater than first value: {rng[0]}. '
        assert len(rng) == 2, f"Range number {idx} has length {len(rng)}. It must be a tuple of length 2."

    if exclude is None:
        exclude = []

    _metrics = metrics.copy()
    for k in metrics.keys():
        if k in exclude:
            _metrics.pop(k)

    assert plot_type in ['bar', 'radial'], f'plot_type must be either `bar` or `radial`.'

    for _range in ranges:
        plot_metrics_between(
            _metrics,
            *_range,
            plot_type=plot_type,
            max_metrics_per_fig=max_metrics_per_fig,
            show=show,
            save=save,
            save_path=save_path, **kwargs)
    return


def plot_metrics_between(
        errors: dict,
        lower: int,
        upper: int,
        plot_type: str = 'bar',
        max_metrics_per_fig: int = 15,
        save=False,
        show=True,
        save_path='',
        **kwargs):
    import matplotlib.pyplot as plt

    if plot_type == "bar":
        from easy_mpl import circular_bar_plot

    selected_metrics = {}
    for k, v in errors.items():
        if v is not None:
            if lower < v < upper:
                selected_metrics[k] = v
    st = 0
    n = len(selected_metrics)
    sequence = np.linspace(0, n, int(n / max_metrics_per_fig) + 1)
    if len(sequence) == 1 and n > 0:
        sequence = np.array([0, len(selected_metrics)])
    for i in np.array(sequence, dtype=np.int32):
        if i == 0:
            pass
        else:
            en = i
            d = take(st, en, selected_metrics)
            if plot_type == 'radial':
                plot_radial(d,
                            lower,
                            upper,
                            save=save,
                            show=show,
                            save_path=save_path,
                            **kwargs)
            elif len(d) < 10:
                pass
            else:
                plt.close('all')
                _ = circular_bar_plot(d, show=False, **kwargs)
                if save:
                    plt.savefig(
                        os.path.join(save_path, f"errors_{lower}_{upper}_{st}_{en}.png"),
                        bbox_inches="tight")
                if show:
                    plt.show()
            st = i
    return


def plot_radial(errors: dict, low: int, up: int, save=True, save_path=None, **kwargs):
    """Plots all the errors in errors dictionary. low and up are used to draw the limits of radial plot."""
    if go is None:
        print("can not plot radial plot because plotly is not installed.")
        return

    fill = kwargs.get('fill', None)
    fillcolor = kwargs.get('fillcolor', None)
    line = kwargs.get('line', None)
    marker = kwargs.get('marker', None)

    OrderedDict(sorted(errors.items(), key=lambda kv: kv[1]))

    lower = round(np.min(list(errors.values())), 4)
    upper = round(np.max(list(errors.values())), 4)

    fig = go.Figure()
    categories = list(errors.keys())

    fig.add_trace(go.Scatterpolar(
        r=list(errors.values()),
        theta=categories,  # angular coordinates
        fill=fill,
        fillcolor=fillcolor,
        line=line,
        marker=marker,
        name='errors'
    ))

    fig.update_layout(
        title_text=f"Errors from {lower} to {upper}",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[low, up]
            )),
        showlegend=False
    )

    fig.show()
    if save:
        fname = f"radial_errors_from_{lower}_to_{upper}.png"
        if save_path is not None:
            fname = os.path.join(save_path, fname)
        fig.write_image(fname)
    return


def plot1d(true, predicted, save=True, name="plot", show=False):
    import matplotlib.pyplot as plt

    _, axis = plt.subplots()

    axis.plot(np.arange(len(true)), true, label="True")
    axis.plot(np.arange(len(predicted)), predicted, label="Predicted")
    axis.legend(loc="best")

    if save:
        plt.savefig(name, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    plt.close('all')
    return


def _foo(denominator, numerator):
    nonzero_numerator = numerator != 0
    nonzero_denominator = denominator != 0
    valid_score = nonzero_numerator & nonzero_denominator
    output_scores = np.ones(1)

    output_scores[valid_score] = 1 - (numerator[valid_score] /
                                      denominator[valid_score])
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.
    return output_scores


def _mean_tweedie_deviance(y_true, y_pred, power=0, weights=None):
    # copying from
    # https://github.com/scikit-learn/scikit-learn/blob/95d4f0841d57e8b5f6b2a570312e9d832e69debc/sklearn/metrics/_regression.py#L659

    message = ("Mean Tweedie deviance error with power={} can only be used on "
               .format(power))
    if power < 0:
        # 'Extreme stable', y_true any real number, y_pred > 0
        if (y_pred <= 0).any():
            raise ValueError(message + "strictly positive y_pred.")
        dev = 2 * (np.power(np.maximum(y_true, 0), 2 - power)
                   / ((1 - power) * (2 - power))
                   - y_true * np.power(y_pred, 1 - power) / (1 - power)
                   + np.power(y_pred, 2 - power) / (2 - power))
    elif power == 0:
        # Normal distribution, y_true and y_pred any real number
        dev = (y_true - y_pred) ** 2
    elif power < 1:
        raise ValueError("Tweedie deviance is only defined for power<=0 and "
                         "power>=1.")
    elif power == 1:
        # Poisson distribution, y_true >= 0, y_pred > 0
        if (y_true < 0).any() or (y_pred <= 0).any():
            raise ValueError(message + "non-negative y_true and strictly "
                                       "positive y_pred.")
        dev = 2 * (xlogy(y_true, y_true / y_pred) - y_true + y_pred)
    elif power == 2:
        # Gamma distribution, y_true and y_pred > 0
        if (y_true <= 0).any() or (y_pred <= 0).any():
            raise ValueError(message + "strictly positive y_true and y_pred.")
        dev = 2 * (np.log(y_pred / y_true) + y_true / y_pred - 1)
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

        dev = 2 * (np.power(y_true, 2 - power) / ((1 - power) * (2 - power))
                   - y_true * np.power(y_pred, 1 - power) / (1 - power)
                   + np.power(y_pred, 2 - power) / (2 - power))

    return float(np.average(dev, weights=weights))


def _geometric_mean(a, axis=0):
    """ Geometric mean """
    return float(np.exp(np.log(a).mean(axis=axis)))


def listMethods(cls):
    return set(x for x, y in cls.__dict__.items()
               if isinstance(y, (FunctionType, classmethod, staticmethod)))


def listParentMethods(cls):
    return set(itertools.chain.from_iterable(
        listMethods(c).union(listParentMethods(c)) for c in cls.__bases__))


def list_subclass_methods(cls, is_narrow, ignore_underscore=True, additional_ignores=None):
    """Finds all methods of a child class"""
    methods = listMethods(cls)

    if is_narrow:
        parent_methods = listParentMethods(cls)
        methods = set(cls for cls in methods if not (cls in parent_methods))

    if additional_ignores is not None:
        methods = methods - set(additional_ignores)

    if ignore_underscore:
        methods = set(cls for cls in methods if not cls.startswith('_'))

    return methods


def features(data: Union[np.ndarray, list],
             precision: int = 3,
             name: str = '',
             st: int = 0,
             en: int = None,
             features: Union[list, str] = None
             ) -> dict:
    """
    Extracts features from 1d time series data. Features can be
        * point, one integer or float point value for example mean
        * 1D, 1D array for example sin(data)
        * 2D, 2D array for example wavelent transform

    Arguments
    ---------
        data: array like
        precision: number of significant figures
        name: str, only for erro or warning messages
        st: str/int, starting index of data to be considered.
        en: str/int, end index of data to be considered.
        features: name/names of features to extract from data.

    # information holding degree
    """
    point_features = {
        'Skew': skew,
        'Kurtosis': kurtosis,
        'Mean': np.nanmean,
        'Geometric Mean': gmean,
        'Standard error of mean': scipy.stats.sem,
        'Median': np.nanmedian,
        'Variance': np.nanvar,
        'Coefficient of Variation': variation,
        'Std': np.nanstd,
        'Non Zeros': np.count_nonzero,
        'Min': np.nanmin,
        'Max': np.nanmax,
        'Sum': np.nansum,
        'Counts': np.size
    }

    point_features_lambda = {
        # 'Shannon entropy': lambda x: np.round(scipy.stats.entropy(pd.Series(x).value_counts()), precision),
        'Negative counts': lambda x: int(np.sum(x < 0.0)),
        '90th percentile': lambda x: round(np.nanpercentile(x, 90), precision),
        '75th percentile': lambda x: round(np.nanpercentile(x, 75), precision),
        '50th percentile': lambda x: round(np.nanpercentile(x, 50), precision),
        '25th percentile': lambda x: round(np.nanpercentile(x, 25), precision),
        '10th percentile': lambda x: round(np.nanpercentile(x, 10), precision),
    }

    if not isinstance(data, np.ndarray):
        if hasattr(data, '__len__'):
            data = np.array(data)
        else:
            raise TypeError(f"{name} must be array like but it is of type {data.__class__.__name__}")

    if np.array(data).dtype.type is np.str_:
        warnings.warn(f"{name} contains string values")
        return {}

    if 'int' not in data.dtype.name:
        if 'float' not in data.dtype.name:
            warnings.warn(f"changing the dtype of {name} from {data.dtype.name} to float")
            data = data.astype(np.float64)

    assert data.size == len(data), f"""
data must be 1 dimensional array but it has shape {np.shape(data)}
"""
    data = data[st:en]
    stats = dict()

    if features is None:
        features = list(point_features.keys()) + list(point_features_lambda.keys())
    elif isinstance(features, str):
        features = [features]

    for feat in features:
        if feat in point_features:
            stats[feat] = np.round(point_features[feat](data), precision)
        elif feat in point_features_lambda:
            stats[feat] = point_features_lambda[feat](data)

    for k, v in stats.items():
        if 'int' in v.__class__.__name__:
            stats[k] = int(v)
        else:
            stats[k] = round(float(v), precision)

    return stats


def maybe_to_oneD_array(array_like):
    """converts x to 1D array if possible otherwise returns as it is
    """
    if array_like.__class__.__name__ in ['list', 'tuple', 'Series', 'int', 'float']:
        return np.array(array_like)

    if isinstance(array_like, np.ndarray):
        if len(array_like) == array_like.size:
            return array_like.reshape(-1, )

    return array_like


def to_oneD_array(array_like):
    """converts x to 1D array and if not possible, raises ValueError
    Returned array will have shape (n,)
    """
    if array_like.__class__.__name__ in ['list', 'tuple', 'Series', 'int', 'float']:
        return np.array(array_like)

    elif array_like.__class__.__name__ == 'ndarray':
        if array_like.ndim == 1:
            return array_like
        else:
            if array_like.size != len(array_like):
                raise ValueError(f'cannot convert multidim array of shape {array_like.shape} to 1d')

            return array_like.reshape(-1, )

    elif array_like.__class__.__name__ == 'DataFrame' and array_like.ndim == 2:
        assert len(array_like) == array_like.size
        return array_like.values.reshape(-1, )
    elif array_like.__class__.__name__ == "Series":
        return array_like.values.reshape(-1, )
    elif isinstance(array_like, float) or isinstance(array_like, int):
        return np.array([array_like])
    else:
        raise ValueError(f'cannot convert object {array_like.__class__.__name__}  to 1d ')


METRIC_TYPES = {
    "r2": "max",
    "nse": "max",
    "nse_alpha": "max",
    "nse_beta": "max",
    "nse_mod": "max",
    "nse_rel": "max",
    "nse_bound": "max",
    "r2_score": "max",
    "adjusted_r2": "max",
    "kge": "max",
    "kge_bound": "max",
    "kge_mod": "max",
    "kge_np": "max",
    'log_nse': 'max',
    "corr_coeff": "max",
    'accuracy': "max",
    'f1_score': 'max',
    "mse": "min",
    "rmse": "min",
    "rmsle": "min",
    "mape": "min",
    "nrmse": "min",
    "pbias": "min",
    "bias": "min",
    "med_seq_error": "min",
    "mae": "min",
    "abs_pbias": "min",
    "gmae": "min",
    "inrse": "min",
    "irmse": "min",
    "mase": "min",
    "mare": "min",
    "msle": "min",
    "decomposed_mse": "min",
    "euclid_distance": "min",
    'exp_var_score': 'max',
    'expanded_uncertainty': 'min',
    'fdc_fhv': 'min',
    'fdc_flv': 'min',
    'gmean_diff': 'min',
    'gmrae': 'min',
    'JS': 'min',
    'kendaull_tau': 'max',
    'kgeprime_bound': 'max',
    'kgenp_bound': 'max',
    'kl_sym': 'min',
    'lm_index': 'max',
    'maape': 'min',
    'mbe': 'min',
    'mbrae': 'min',
    'mapd': 'min',
    'max_error': 'min',
    'rse': 'min',
    'rrse': 'min',
    'rae': 'min',
    'ref_agreement_index': 'max',
    'rel_agreement_index': 'max',
    'relative_rmse': 'min',
    'rmspe': 'min',
    'rsr': 'min',
    'rmsse': 'min',
    'sa': 'min',
    'sc': 'min',
    'sga': 'min',
    'smape': 'min',
    'smdape': 'min',
    'sid': 'min',
    'skill_score_murphy': 'max',
    'std_ratio': 'min',
    'umbrae': 'min',
    've': 'min',
    'volume_error': 'min',
    'wape': 'min',
    'watt_m': 'min',
    'wmape': 'min',
    'norm_ape': 'min',
    'post_process_kge': 'min',
    'spearmann_corr': 'min',
    'log1p': 'min',
    'covariance': 'min',
    'brier_score': 'min',
    'bic': 'min',
    'sse': 'min',
    'amemiya_pred_criterion': 'min',
    'amemiya_adj_r2': 'min',
    'aitchison': 'min',
    'log_t': 'min',
    'log_p': 'min',
    '_assert_greater_than_one': 'min',
    'acc': 'min',
    'agreement_index': 'min',
    'aic': 'min',
    'cronbach_alpha': 'min',
    'centered_rms_dev': 'min',
    'cosine_similarity': 'min',
    '_error': 'min',
    '_relative_error': 'min',
    '_naive_prognose': 'min',
    '_minimal': 'min',
    '_hydro_metrics': 'min',
    'calculate_hydro_metrics': 'min',
    '_bounded_relative_error': 'min',
    '_ae': 'min',
    'mb_r': 'min',
    'mda': 'min',
    'mde': 'min',
    'mdape': 'min',
    'mdrae': 'min',
    'me': 'min',
    'mean_bias_error': 'min',
    'mean_var': 'min',
    'mean_poisson_deviance': 'min',
    'mean_gamma_deviance': 'min',
    'median_abs_error': 'min',
    'mle': 'min',
    'mod_agreement_index': 'min',
    'mpe': 'min',
    'mrae': 'min',
    'norm_euclid_distance': 'min',
    'nrmse_range': 'min',
    'nrmse_ipercentile': 'min',
    'nrmse_mean': 'min',
    'norm_ae': 'min',
    'log_prob': 'min',
    'rmdspe': 'min',
    'variability_ratio': 'min',
    "mre" : 'min'
}


def _assert_1darray(array_like, metric_type: str) -> np.ndarray:
    """Makes sure that the provided `array_like` is 1D numpy array"""

    # this will convert tensorflow and torch tensors to numpy
    if hasattr(array_like, 'numpy') and callable(getattr(array_like, 'numpy')):
        array_like = getattr(array_like, 'numpy')()

    # this will cover xarray DataArray
    if hasattr(array_like, 'to_numpy') and callable(getattr(array_like, 'to_numpy')):
        array_like = getattr(array_like, 'to_numpy')()

    if metric_type == "regression":
        return to_oneD_array(array_like)

    return maybe_to_oneD_array(array_like)


def maybe_treat_arrays(
        preprocess: bool = None,
        true=None,
        predicted=None,
        metric_type: str = None,
        **process_kws
):
    """
    This function is applied by default at the start/at the time of initiating
    the class. However, it can be used any time after that. This can be handy
    if we want to calculate error first by ignoring nan and then by no ignoring
    nan. Adopting from HydroErr_ . Removes the nan, negative, and inf values
    in two numpy arrays

    .. _HydroErr:
        https://github.com/BYU-Hydroinformatics/HydroErr/blob/master/HydroErr/HydroErr.py#L6210

    parameters
    ----------
    preprocess: bool, default None
        if True, preprocess the true and predicted arrays
    true: array_like
        array of true/actual/observed values
    predicted: array_like
        array of predicted/simulated/calculated values
    metric_type: str
        type of metric, either "regression" or "classification"
    process_kws:
        keyword arguments for preprocessing
            - remove_nan: bool, default True
            - remove_inf : bool, default True
            - replace_nan: float, default None
            - remove_zero: bool, default None
            - remove_neg: bool, default None
            - replace_inf: float, default None
    """

    if preprocess:
        predicted = _assert_1darray(predicted, metric_type)
        true = _assert_1darray(true, metric_type)
        assert len(predicted) == len(true), """
        lengths of provided arrays mismatch, predicted array: {}, true array: {}
        """.format(len(predicted), len(true))

        if metric_type == 'regression':  # todo
            true, predicted = treat_arrays(true, predicted, **process_kws)

    return true, predicted


def treat_arrays(
        true,
        predicted,
        remove_nan: bool = True,
        remove_inf: bool = True,
        replace_nan=None,
        remove_zero=None,
        remove_neg=None,
        replace_inf=None
):
    sim_copy = np.copy(predicted)
    obs_copy = np.copy(true)

    sim_copy, obs_copy = maybe_replace(sim_copy, obs_copy, replace_nan, replace_inf, remove_zero, remove_neg)

    if remove_nan:
        sim_copy, obs_copy = maybe_remove_nan(sim_copy, obs_copy)

    if remove_inf:
        sim_copy, obs_copy = maybe_remove_inf(sim_copy, obs_copy)

    return obs_copy, sim_copy


def maybe_remove_nan(sim_copy, obs_copy):

    data = np.array([sim_copy.flatten(), obs_copy.flatten()])
    data = np.transpose(data)
    if data.dtype.kind not in {'U', 'S'}:
        data = data[~np.isnan(data).any(1)]  # TODO check NaNs in an array containing strings
    sim_copy, obs_copy = data[:, 0], data[:, 1]

    return sim_copy, obs_copy

def maybe_remove_inf(sim_copy, obs_copy):

    data = np.array([sim_copy.flatten(), obs_copy.flatten()])
    data = np.transpose(data)
    if data.dtype.kind not in {'U', 'S'}:
        data = data[~np.isinf(data).any(1)]  # TODO infinity NaNs in an array containing strings
    sim_copy, obs_copy = data[:, 0], data[:, 1]

    return sim_copy, obs_copy

def maybe_replace(sim_copy, obs_copy, replace_nan, replace_inf, remove_zero, remove_neg):
    # Treat missing data in observed_array and simulated_array, rows in simulated_array or
    # observed_array that contain nan values
    all_treatment_array = np.ones(obs_copy.size, dtype=bool)

    if replace_nan and (np.any(np.isnan(obs_copy)) or np.any(np.isnan(sim_copy))):
        # Finding the NaNs
        sim_nan = np.isnan(sim_copy)
        obs_nan = np.isnan(obs_copy)
        # Replacing the NaNs with the input
        sim_copy[sim_nan] = replace_nan
        obs_copy[obs_nan] = replace_nan

        warnings.warn("Elements(s) {} contained NaN values in the simulated array and "
                      "elements(s) {} contained NaN values in the observed array and have been "
                      "replaced (Elements are zero indexed).".format(np.where(sim_nan)[0],
                                                                     np.where(obs_nan)[0]),
                      UserWarning)

    if replace_inf and (np.any(np.isinf(obs_copy)) or np.any(np.isinf(sim_copy))):
        # Finding the NaNs
        sim_inf = np.isinf(sim_copy)
        obs_inf = np.isinf(obs_copy)
        # Replacing the NaNs with the input
        sim_copy[sim_inf] = replace_inf
        obs_copy[obs_inf] = replace_inf

        warnings.warn("Elements(s) {} contained Inf values in the simulated array and "
                      "elements(s) {} contained Inf values in the observed array and have been "
                      "replaced (Elements are zero indexed).".format(np.where(sim_inf)[0],
                                                                     np.where(obs_inf)[0]),
                      UserWarning)

    # Treat zero data in observed_array and simulated_array, rows in simulated_array or
    # observed_array that contain zero values
    if remove_zero:
        if (obs_copy == 0).any() or (sim_copy == 0).any():
            zero_indices_fcst = ~(sim_copy == 0)
            zero_indices_obs = ~(obs_copy == 0)
            all_zero_indices = np.logical_and(zero_indices_fcst, zero_indices_obs)
            all_treatment_array = np.logical_and(all_treatment_array, all_zero_indices)

            warnings.warn(
                "Row(s) {} contained zero values and the row(s) have been removed (Rows are "
                "zero indexed).".format(np.where(~all_zero_indices)[0]),
                UserWarning
            )

    # Treat negative data in observed_array and simulated_array, rows in simulated_array or
    # observed_array that contain negative values

    # Ignore runtime warnings from comparing
    if remove_neg:
        with np.errstate(invalid='ignore'):
            obs_copy_bool = obs_copy < 0
            sim_copy_bool = sim_copy < 0

        if obs_copy_bool.any() or sim_copy_bool.any():
            neg_indices_fcst = ~sim_copy_bool
            neg_indices_obs = ~obs_copy_bool
            all_neg_indices = np.logical_and(neg_indices_fcst, neg_indices_obs)
            all_treatment_array = np.logical_and(all_treatment_array, all_neg_indices)

            warnings.warn("Row(s) {} contained negative values and the row(s) have been "
                          "removed (Rows are zero indexed).".format(np.where(~all_neg_indices)[0]),
                          UserWarning)

    obs_copy = obs_copy[all_treatment_array]
    sim_copy = sim_copy[all_treatment_array]

    return sim_copy, obs_copy