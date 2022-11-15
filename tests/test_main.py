
import unittest

from SeqMetrics import RegressionMetrics

class Test(unittest.TestCase):

    def test_errstate(self):
        t = [1,2,3]
        p = [1, 2, 0.0]
        RegressionMetrics(t,p, np_errstate={"invalid": "ignore"}).rmsle()
        return


if __name__ == "__main__":
    unittest.main()