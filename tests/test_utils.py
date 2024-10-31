
import os
import unittest
import site   # so that seqmetrics directory is in path

ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(ai4_dir)

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from SeqMetrics import plot_metrics
from SeqMetrics.utils import features
from SeqMetrics import RegressionMetrics
from SeqMetrics.utils import one_hot_encode
from SeqMetrics import ClassificationMetrics
from SeqMetrics.utils import confusion_matrix
from SeqMetrics.utils import maybe_treat_arrays
from SeqMetrics.utils import list_subclass_methods


t = np.random.random((20, 1))
p = np.random.random((20, 1))

er = RegressionMetrics(t, p)

all_errors = er.calculate_all()


class TestRegressionMetrics(unittest.TestCase):

    def test_treat_arrays(self):

        a = [1,2,2]
        b = [np.nan, np.nan, np.nan]

        # raise error
        #RegressionMetrics(a,b).mse()
        self.assertRaises(ValueError, RegressionMetrics, a, b)

        # raise error
        #RegressionMetrics(b, a).mse()
        self.assertRaises(ValueError, RegressionMetrics, b, a)
        return 


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

    def test_pandas_series(self):

        ts = pd.Series(np.random.random((20,)))
        ps = pd.Series(np.random.random((20,)))

        ers = RegressionMetrics(ts, ps)

        series_errors = ers.calculate_all()
        assert len(series_errors) > 100
        for er_name, er_val in series_errors.items():
            if er_val is not None:
                er_val = getattr(ers, er_name)()
                self.assertEqual(er_val.__class__.__name__, 'float', f'{er_name} is {er_val}')
        
        return

    def test_pandas_dataframe(self):
        
        tdf = pd.DataFrame(np.random.random((20, 1)))
        pdf = pd.DataFrame(np.random.random((20, 1)))

        erdf = RegressionMetrics(tdf, pdf)

        df_errors = erdf.calculate_all()
        assert len(df_errors) > 100
        for er_name, er_val in df_errors.items():
            if er_val is not None:
                er_val = getattr(erdf, er_name)()
                self.assertEqual(er_val.__class__.__name__, 'float', f'{er_name} is {er_val}')
        return

    def test_torch_tensor(self):
        try:
            import torch
        except (ModuleNotFoundError, ImportError):
            print('Cant run test_torch_tensor')
            torch = None

        if torch is not None:
            t_ = torch.tensor(np.random.random(10))
            p_ = torch.tensor(np.random.random(10))
            try:
                _t, _p = maybe_treat_arrays(True, t_, p_)
            except RuntimeError as e:
                print(f"Runtime Error Encountered while converting tensor to numpy {torch.__version__} {np.__version__} {os.name}")
                return

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


class TestOneHotEncoding(unittest.TestCase):
    """makes sure that one_hot_encode function in utils is works well
    for differnet kinds of inputs. We expect it to be used for only
    followign five kinds of inputs.
    """
    def test_boolean(self):
        boolean = np.array([True, False, False, False])
        np.testing.assert_array_equal(one_hot_encode(boolean), pd.get_dummies(boolean).values)
        return

    def test_binary_numerical(self):
        binary_numerical = np.array([1, 0, 0, 0])
        np.testing.assert_array_equal(one_hot_encode(binary_numerical), pd.get_dummies(binary_numerical))
        return

    def test_multicls_numerical(self):
        multicls_numerical = np.array([1, 0, 0, 2, 0, 3])
        np.testing.assert_array_equal(one_hot_encode(multicls_numerical), pd.get_dummies(multicls_numerical))
        return

    def test_multicls_numerical_large(self):
        multicls_numerical = np.random.randint(1, 10, 100)
        np.testing.assert_array_equal(one_hot_encode(multicls_numerical), pd.get_dummies(multicls_numerical))
        return

    def test_multicls_numerical_neg(self):
        multicls_numerical = np.random.randint(-5, 5, 100)
        np.testing.assert_array_equal(one_hot_encode(multicls_numerical), pd.get_dummies(multicls_numerical))
        return

    def test_binary_categorical(self):
        binary_categorical = np.array(['a', 'b', 'b', 'b'])
        np.testing.assert_array_equal(one_hot_encode(binary_categorical), pd.get_dummies(binary_categorical))
        return

    def test_multicls_categorical(self):
        multicls_categorical = np.array(['a', 'b', 'b', 'b', 'c'])
        np.testing.assert_array_equal(one_hot_encode(multicls_categorical), pd.get_dummies(multicls_categorical))
        return

    def test_floats(self):
        floats = np.random.random(10)
        np.testing.assert_array_equal(one_hot_encode(floats), pd.get_dummies(floats))
        return


class TestConfusionMatrix(unittest.TestCase):
    """makes sure that confusion_matrix function in utils is works well
    for differnet kinds of inputs. We expect it to be used for only
    followign five kinds of inputs.
    """
    def test_boolean(self):
        true = np.array([True, False, False, False])
        pred = np.array([True, True, True, True])
        np.testing.assert_array_equal(confusion_matrix(true, pred), sk_confusion_matrix(true, pred))
        return

    def test_binary_numerical(self):
        true = np.array([1, 0, 0, 0])
        pred = np.array([1, 1, 1, 1])

        np.testing.assert_array_equal(confusion_matrix(true, pred), sk_confusion_matrix(true, pred))
        return

    def test_binary_categorical(self):
        true = np.array(['a', 'b', 'b', 'b'])
        pred = np.array(['a', 'a', 'a', 'a'])
        np.testing.assert_array_equal(confusion_matrix(true, pred), sk_confusion_matrix(true, pred))
        return

    def test_multiclass_numerical(self):
        true = np.random.randint(1, 4, 100)
        pred = np.random.randint(1, 4, 100)
        np.testing.assert_array_equal(confusion_matrix(true, pred), sk_confusion_matrix(true, pred))
        return

    def test_multicls_categorical(self):
        true = np.array(['car', 'truck', 'truck', 'car', 'bike', 'truck'])
        pred = np.array(['car', 'car', 'bike', 'car', 'bike', 'truck'])
        np.testing.assert_array_equal(confusion_matrix(true, pred), sk_confusion_matrix(true, pred))
        return


if __name__ == "__main__":
    unittest.main()
