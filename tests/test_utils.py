import os
import unittest
import site   # so that ai4water directory is in path

ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(ai4_dir)

import numpy as np

from SeqMetrics import RegressionMetrics
from SeqMetrics import plot_metrics
from SeqMetrics.utils import list_subclass_methods
from SeqMetrics.utils import features


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

if __name__ == "__main__":
    unittest.main()
