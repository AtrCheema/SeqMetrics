
import os
import unittest
import site   # so that seqmetrics directory is in path

ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(ai4_dir)

import numpy as np

from SeqMetrics import ClassificationMetrics
from SeqMetrics import RegressionMetrics
from SeqMetrics import plot_metrics
from SeqMetrics.utils import list_subclass_methods
from SeqMetrics.utils import features
from SeqMetrics.utils import maybe_treat_arrays


t = np.random.random((20, 1))
p = np.random.random((20, 1))

er = RegressionMetrics(t, p)

all_errors = er.calculate_all()


class TestPlot(unittest.TestCase):

    def test_radial_pots(self):
        plot_metrics(all_errors,
                     plot_type='bar',
                     max_metrics_per_fig=50,
                     save_path=os.path.join(os.getcwd(), "results"),
                     show=False)

        try:
            import plotly.graph_objects as go
        except (ImportError, ModuleNotFoundError):
            return
    
        plot_metrics(all_errors,
                     plot_type='radial',
                     save_path=os.path.join(os.getcwd(), "results"),
                     show=False)
        return

    def test_circular_bar(self):
        np.random.seed(313)
        true = np.random.random((20, 1))
        pred = np.random.random((20, 1))

        plot_metrics(RegressionMetrics(true, pred).calculate_all(),
                     show=False, save=False, color="Blues")
        return

    def test_list_subclass_methods(self):
        class DP:
            def _pa(self): pass

        class D(DP):
            def a(self): pass
            def _a(self): pass
            def b(self): pass

        self.assertEqual(len(list_subclass_methods(D, True)), 2)
        self.assertEqual(len(list_subclass_methods(D, True, False)), 3)
        self.assertEqual(len(list_subclass_methods(D, True, False, ['b'])), 2)
        return

    def test_features_list(self):
        x = [1,2,3,3,25,2,2,4,5,12,3,5645]
        stats = features(x)
        for k, s in stats.items():
            assert isinstance(s, float) or isinstance(s, int), f"{k} is of type {type(s)}"
        return

    def test_features_array(self):
        x = np.random.random(20)
        stats = features(x)
        for k, s in stats.items():
            assert isinstance(s, float) or isinstance(s, int), f"{k} is of type {type(s)}"
        return

    def test_features_ndarray(self):
        x = np.random.random((20, 1))
        stats = features(x)

        for k, s in stats.items():
            assert isinstance(s, float) or isinstance(s, int), f"{k} is of type {type(s)}"
        return


class TestInputFromOtherLibraries(unittest.TestCase):

    def test_torch_tensor(self):
        try:
            import torch
        except (ModuleNotFoundError, ImportError):
            print('Cant run test_torch_tensor')
            torch = None

        if torch is not None:
            t_ = torch.tensor(np.random.random(10))
            p_ = torch.tensor(np.random.random(10))
            _t, _p = maybe_treat_arrays(True, t_, p_)
            assert isinstance(_t, np.ndarray)
            assert isinstance(_p, np.ndarray)

            kge_ = RegressionMetrics(t_, p_).kge()

            assert isinstance(kge_, float)

            acc_ = ClassificationMetrics(t_, p_).accuracy()

            assert isinstance(acc_, float)
        return


    def test_tf_tensor(self):
        try:
            import tensorflow as tf
        except (ModuleNotFoundError, ImportError):
            print('Cant run test_tf_tensor')
            tf = None

        if tf is not None:
            t_ = tf.constant(np.random.random(10))
            p_ = tf.constant(np.random.random(10))
            _t, _p = maybe_treat_arrays(True, t_, p_)
            assert isinstance(_t, np.ndarray)
            assert isinstance(_p, np.ndarray)

            kge_ = RegressionMetrics(t_, p_).kge()

            assert isinstance(kge_, float)

            acc_ = ClassificationMetrics(t_, p_).accuracy()

            assert isinstance(acc_, float)

        return

    def test_xr_dataarray(self):
        try:
            import xarray as xr
        except (ModuleNotFoundError, ImportError):
            print('Cant run test_xr_dataarray')
            xr = None

        if xr is not None:
            t_ = xr.DataArray(np.random.random(10))
            p_ = xr.DataArray(np.random.random(10))
            _t, _p = maybe_treat_arrays(True, t_, p_)
            assert isinstance(_t, np.ndarray)
            assert isinstance(_p, np.ndarray)

            kge_ = RegressionMetrics(t_, p_).kge()

            assert isinstance(kge_, float)

            acc_ = ClassificationMetrics(t_, p_).accuracy()

            assert isinstance(acc_, float)

        return

if __name__ == "__main__":
    unittest.main()
