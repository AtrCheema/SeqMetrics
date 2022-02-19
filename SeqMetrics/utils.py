import os
import warnings
import itertools
from typing import Union
from types import FunctionType
from collections import OrderedDict

import scipy
import numpy as np
from scipy.special import xlogy
from scipy.stats import skew, kurtosis, variation, gmean, hmean

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
        save_path: str = None,
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
            either of `radial` or `bar`.
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
        save_path=None,
        **kwargs):
    zero_to_one = {}
    for k, v in errors.items():
        if v is not None:
            if lower < v < upper:
                zero_to_one[k] = v
    st = 0
    n = len(zero_to_one)
    for i in np.array(np.linspace(0, n, int(n/max_metrics_per_fig)+1),
                      dtype=np.int32):
        if i == 0:
            pass
        else:
            en = i
            d = take(st, en, zero_to_one)
            if plot_type == 'radial':
                plot_radial(d, lower, upper, save=save, show=show, save_path=save_path, **kwargs)
            else:
                plot_circular_bar(d, save=save, show=show, save_path=save_path, **kwargs)
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


def plot_circular_bar(
        metrics: dict,
        show=False,
        save: bool = True,
        save_path: str = '',
        **kwargs):
    """
    modified after https://www.python-graph-gallery.com/circular-barplot-basic
    :param metrics:
    :param show:
    :param save:
    :param save_path:
    :param kwargs:
        figsize:
        linewidth:
        edgecolor:
        color:
    :return:
    """
    import matplotlib.pyplot as plt

    # initialize the figure
    plt.close('all')
    plt.figure(figsize=kwargs.get('figsize', (8, 12)))
    ax = plt.subplot(111, polar=True)
    plt.axis('off')

    # Set the coordinates limits
    # upperLimit = 100
    lower_limit = 30
    value = np.array(list(metrics.values()))

    lower = round(np.min(list(metrics.values())), 4)
    upper = round(np.max(list(metrics.values())), 4)

    # Compute max and min in the dataset
    _max = max(value)  # df['Value'].max()

    # Let's compute heights: they are a conversion of each item value in those new coordinates
    # In our example, 0 in the dataset will be converted to the lowerLimit (10)
    # The maximum will be converted to the upperLimit (100)
    slope = (_max - lower_limit) / _max
    heights = slope * value + lower_limit

    # Compute the width of each bar. In total we have 2*Pi = 360°
    width = 2 * np.pi / len(metrics)

    # Compute the angle each bar is centered on:
    indexes = list(range(1, len(metrics) + 1))
    angles = [element * width for element in indexes]

    # Draw bars
    bars = ax.bar(
        x=angles,
        height=heights,
        width=width,
        bottom=lower_limit,
        linewidth=kwargs.get('linewidth', 2),
        edgecolor=kwargs.get('edgecolor', "white"),
        color=kwargs.get('color', "#61a4b2"),
    )

    # little space between the bar and the label
    label_padding = 4

    metric_names = {
        'r2': "$R^2$",
        'r2_mod': "$R^2$ mod",
        'adjusted_r2': 'adjusted $R^2$',
        # 'nse': "NSE"
    }

    # Add labels
    for bar, angle, label1, label2 in zip(bars, angles, metrics.keys(), metrics.values()):

        label1 = metric_names.get(label1, label1)
        label = f'{label1} {round(label2, 4)}'

        # Labels are rotated. Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle)

        # Flip some labels upside down
        if angle >= np.pi / 2 and angle < 3 * np.pi / 2:
            alignment = "right"
            rotation = rotation + 180
        else:
            alignment = "left"

        # Finally add the labels
        ax.text(
            x=angle,
            y=lower_limit + bar.get_height() + label_padding,
            s=label,
            ha=alignment,
            va='center',
            rotation=rotation,
            rotation_mode="anchor")

    if save:
        fname = f"{len(metrics)}_bar_errors_from_{lower}_to_{upper}.png"
        if save_path is not None:
            fname = os.path.join(save_path, fname)
        plt.savefig(fname, dpi=400, bbox_inches='tight')
    if show:
        plt.show()

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
    return float(np.exp(log_a.mean(axis=axis)))


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
    
    for k,v in stats.items():
        if 'int' in v.__class__.__name__:
            stats[k] = int(v)
        else:
            stats[k] = round(float(v), precision)

    return stats
