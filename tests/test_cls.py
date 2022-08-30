
import os
import site
import unittest
cwd = os.path.dirname(__file__)
site.addsitedir(os.path.dirname(cwd))

import numpy as np 

from SeqMetrics import ClassificationMetrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.preprocessing import LabelBinarizer



class TestBinaryBooleanLabels(unittest.TestCase):
    t = np.array([True, False, False, False])
    p = np.array([True, True, True, True])

    metrics = ClassificationMetrics(t, p)

    def test_f1_score(self):
        self.assertEqual(self.metrics.f1_score(average="weighted"),
                         f1_score(self.t, self.p, average="weighted"))
        return

    def test_accuracy(self):
        val_score = self.metrics.accuracy()
        self.assertAlmostEqual(val_score, 0.25)
        return

    def test_confusion_metrics(self):
        cm = self.metrics.confusion_matrix()
        np.testing.assert_array_equal(cm, confusion_matrix(self.t, self.p))
        return


class TestBinaryNumericalLabels(unittest.TestCase):
    """binary classification when the arrays are nuerical values"""

    true = np.array([1, 0, 0, 0])
    pred = np.array([1, 1, 1, 1])
    metrics = ClassificationMetrics(true, pred)

    # ClassificationMetrics class should handle 2d arrays when their size == their length
    metrics_nd = ClassificationMetrics(true.reshape(-1,1), pred.reshape(-1,1))

    def test_accuracy(self):
        val_score = self.metrics.accuracy()
        self.assertAlmostEqual(val_score, 0.25)

        self.assertAlmostEqual(self.metrics_nd.accuracy(), 0.25)
        return

    def test_confusion_matrix(self):

        for normalize in [None, "all", "true", "pred"]:
            act_cm = confusion_matrix(self.true, self.pred, normalize=normalize)
            cm = self.metrics.confusion_matrix(normalize=normalize)
            np.testing.assert_array_equal(cm, act_cm)

            cm = self.metrics_nd.confusion_matrix(normalize=normalize)
            np.testing.assert_array_equal(cm, act_cm)

        return

    def test_precision(self):
        for average in [#'macro',
            'weighted', None]:
            act_precision = precision_score(self.true, self.pred, average=average)
            calc_precision = self.metrics.precision(average=average)
            np.testing.assert_almost_equal(act_precision, calc_precision)

            calc_precision = self.metrics_nd.precision(average=average)
            np.testing.assert_almost_equal(act_precision, calc_precision)

        return

    def test_recall(self):
        for average in ['macro', 'weighted', None]:

            act_recall = recall_score(self.true, self.pred, average=average)
            calc_recall = self.metrics.recall(average=average)
            np.testing.assert_almost_equal(act_recall, calc_recall)

            calc_recall = self.metrics_nd.recall(average=average)
            np.testing.assert_almost_equal(act_recall, calc_recall)

        return

    def test_f1_score(self):
        for average in ['macro', 'weighted', None]:
            act_f1_score = f1_score(self.true, self.pred, average=average)
            calc_f1_score = self.metrics.f1_score(average=average)
            np.testing.assert_almost_equal(act_f1_score, calc_f1_score)

            calc_f1_score = self.metrics_nd.f1_score(average=average)
            np.testing.assert_almost_equal(act_f1_score, calc_f1_score)
        return

    def test_balanced_accuracy(self):
        act_balance_acc = balanced_accuracy_score(self.true, self.pred)
        calc_balance_acc = self.metrics.balanced_accuracy()
        np.testing.assert_almost_equal(act_balance_acc, calc_balance_acc)

        calc_balance_acc = self.metrics_nd.balanced_accuracy()
        np.testing.assert_almost_equal(act_balance_acc, calc_balance_acc)
        return


class TestBinaryCategoricalLabels(unittest.TestCase):
    """binary classification when the arrays are nuerical values"""


class TestBinaryLogits(unittest.TestCase):
    """binary classification when the arrays are logits"""
    predictions = np.array([[0.12, 0.88],
                            [0.44, 0.66],
                            [0.59, 0.41]
                            ])
    targets = np.array([[0, 1],
                        [1, 0],
                        [1, 0]
                        ])

    t = np.argmax(targets, axis=1)
    p = np.argmax(predictions, axis=1)

    metrics = ClassificationMetrics(targets, predictions)

    def test_ce(self):
        ce = self.metrics.cross_entropy()
        return

    #     def test_class_all(self):
    #         class_metrics = ClassificationMetrics(self.targets, self.predictions, multiclass=True)
    #         all_metrics = class_metrics.calculate_all()
    #         assert len(all_metrics) > 1
    #         return

    def test_all(self):
        self.metrics.calculate_all()
        return

    def test_accuracy(self):
        calc_acc = self.metrics.accuracy()
        t = np.argmax(self.targets, axis=1)
        p = np.argmax(self.predictions, axis=1)
        act_acc = accuracy_score(t, p)
        self.assertAlmostEqual(calc_acc, act_acc)
        return

    def test_f1_score(self):
        t = np.argmax(self.targets, axis=1)
        p = np.argmax(self.predictions, axis=1)
        act_f1_score = f1_score(t, p, average='macro')
        calc_f1_score = self.metrics.f1_score(average="macro")
        self.assertAlmostEqual(act_f1_score, calc_f1_score)
        return

    def test_balance_accuracy(self):
        act_balance_accuracy = balanced_accuracy_score(self.t, self.p)
        calc_balance_accuracy = self.metrics.balanced_accuracy()
        self.assertAlmostEqual(act_balance_accuracy, calc_balance_accuracy)
        return


class TestMulticlassNumericLabels(unittest.TestCase):
    true = np.random.randint(1, 4, 100)
    pred = np.random.randint(1, 4, 100)
    metrics = ClassificationMetrics(true, pred, multiclass=True)

    def test_all(self):
        self.metrics.calculate_all()
        return

    def test_accuracy(self):
        acc = self.metrics.accuracy()
        acc2 = accuracy_score(self.true, self.pred)
        self.assertAlmostEqual(acc, acc2)
        return

    def test_confusion_matrix(self):

        for normalize in [None, "all", "true", "pred"]:
            act_cm = confusion_matrix(self.true, self.pred, normalize=normalize)
            cm = self.metrics.confusion_matrix(normalize=normalize)
            np.testing.assert_array_equal(cm, act_cm)

        return

    def test_precision(self):
        for average in ['macro', 'weighted', None]:

            act_precision = precision_score(self.true, self.pred, average=average)
            calc_precision = self.metrics.precision(average=average)
            np.testing.assert_almost_equal(act_precision, calc_precision)

        return

    def test_recall(self):
        for average in ['macro', 'weighted', None]:

            act_recall = recall_score(self.true, self.pred, average=average)
            calc_recall = self.metrics.recall(average=average)
            np.testing.assert_almost_equal(act_recall, calc_recall)

        return

    def test_f1_score(self):
        for average in ['macro', 'weighted', None]:
            act_f1_score = f1_score(self.true, self.pred, average=average)
            calc_f1_score = self.metrics.f1_score(average=average)
            np.testing.assert_almost_equal(act_f1_score, calc_f1_score)
        return

    def test_balanced_accuracy(self):
        act_balance_acc = balanced_accuracy_score(self.true, self.pred)
        calc_balance_acc = self.metrics.balanced_accuracy()
        np.testing.assert_almost_equal(act_balance_acc, calc_balance_acc)

        return

# class TestMulticlassCategoricalLabels(unittest.TestCase):
#     true = np.random.randint(1, 4, 100)
#     pred = np.random.randint(1, 4, 100)
#     metrics = ClassificationMetrics(true, pred, multiclass=True)

#     def test_all(self):
#         self.metrics.calculate_all()
#         return

#     def test_accuracy(self):
#         acc = self.metrics.accuracy()
#         acc2 = accuracy_score(self.true, self.pred)
#         self.assertAlmostEqual(acc, acc2)
#         return


class TestMulticlassLogits(unittest.TestCase):
    """Arrays are given as probabilities(logits"""
    predictions = np.array([[0.25, 0.25, 0.25, 0.25],
                            [0.01, 0.01, 0.01, 0.96]])
    targets = np.array([[0, 0, 0, 1],
                        [0, 0, 0, 1]])

    t = np.argmax(targets, axis=1)
    p = np.argmax(predictions, axis=1)
    metrics = ClassificationMetrics(targets, predictions, multiclass=True)

    def test_ce(self):

        # https://stackoverflow.com/a/47398312/5982232
        self.assertAlmostEqual(self.metrics.cross_entropy(), 0.71355817782)
        return

#     def test_class_all(self):
#         class_metrics = ClassificationMetrics(self.targets, self.predictions, multiclass=True)
#         all_metrics = class_metrics.calculate_all()
#         assert len(all_metrics) > 1
#         return

    def test_all(self):
        self.metrics.calculate_all()
        return

    def test_accuracy(self):
        calc_acc = self.metrics.accuracy()
        act_acc = accuracy_score(self.t, self.p)
        self.assertAlmostEqual(calc_acc, act_acc)
        return

    def test_f1_score(self):
        act_f1_score = f1_score(self.t, self.p, average='macro')
        calc_f1_score = self.metrics.f1_score(average="macro")
        self.assertAlmostEqual(act_f1_score, calc_f1_score)
        return

    def test_balance_accuracy(self):

        act_balance_accuracy = balanced_accuracy_score(self.t, self.p)
        calc_balance_accuracy = self.metrics.balanced_accuracy()
        self.assertAlmostEqual(act_balance_accuracy, calc_balance_accuracy)
        return


if __name__ == "__main__":
    unittest.main()