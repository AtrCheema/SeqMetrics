
import os
import site
import unittest

seqmet_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(seqmet_dir)

from SeqMetrics import RegressionMetrics

class Test(unittest.TestCase):

    def test_errstate(self):
        t = [1,2,3]
        p = [1, 2, 0.0]
        RegressionMetrics(t,p, np_errstate={"invalid": "ignore"}).rmsle()
        return

    def test_raise_value_error(self):
        t = [1,2,3]
        p = [1, 2, 0.0]
        with self.assertRaises(ValueError):
            RegressionMetrics(t,p).calculate(metric="invalid_metric")


if __name__ == "__main__":
    unittest.main()