import numpy as np
from scipy.stats import skew, kurtosis, variation, gmean, hmean
import scipy
import datetime


def stats(feature) -> dict:
    """Gets all the possible stats about an array like object `feature`.
    `feature: array like
    """
    if not isinstance(feature, np.ndarray):
        if hasattr(feature, '__len__'):
            feature = np.array(feature)
        else:
            raise TypeError(f"input must be array like but it is of type {type(feature)}")

    _stats = dict()
    _stats['Skew'] = skew(feature)
    _stats['Kurtosis'] = kurtosis(feature)
    _stats['Mean'] = np.nanmean(feature)
    _stats['Geometric Mean'] = gmean(feature)
    _stats['Harmonic Mean'] = hmean(feature)
    _stats['Standard error of mean'] = scipy.stats.sem(feature)
    _stats['Median'] = np.nanmedian(feature)
    _stats['Variance'] = np.nanvar(feature)
    _stats['Coefficient of Variation'] = variation(feature)
    _stats['Std'] = np.nanstd(feature)
    _stats['Non zeros'] = np.count_nonzero(feature)
    _stats['10 quant'] = np.nanquantile(feature, 0.1)
    _stats['50 quant'] = np.nanquantile(feature, 0.5)
    _stats['90 quant'] = np.nanquantile(feature, 0.9)
    _stats['25 %ile'] = np.nanpercentile(feature, 25)
    _stats['50 %ile'] = np.nanpercentile(feature, 50)
    _stats['75 %ile'] = np.nanpercentile(feature, 75)
    _stats['Min'] = np.nanmin(feature)
    _stats['Max'] = np.nanmax(feature)
    _stats["Negative counts"] = float(np.sum(feature < 0.0))
    _stats["NaN counts"] = np.isnan(feature).sum()
    _stats['Counts'] = len(feature)

    return _stats

def dateandtime_now():
    """Returns the datetime in following format
    YYYYMMDD_HHMMSS
    """
    jetzt = datetime.datetime.now()
    dt = str(jetzt.year)
    for time in ['month', 'day', 'hour', 'minute', 'second']:
        _time = str(getattr(jetzt, time))
        if len(_time) < 2:
            _time = '0' + _time
        if time == 'hour':
            _time = '_' + _time
        dt += _time
    return dt
