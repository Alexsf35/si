from unittest import TestCase
import numpy as np

from si.neural_networks.losses import (
    MeanSquaredError,
    BinaryCrossEntropy,
    CategoricalCrossEntropy
)


class TestMeanSquaredError(TestCase):

    def setUp(self):
        self.mse = MeanSquaredError()

        self.y_true = np.array([1.0, 2.0, 3.0])
        self.y_pred = np.array([1.5, 1.5, 2.5])

    def test_loss(self):
        loss = self.mse.loss(self.y_true, self.y_pred)
        expected = np.mean((self.y_true - self.y_pred) ** 2)

        self.assertAlmostEqual(loss, expected, places=6)

    def test_derivative(self):
        derivative = self.mse.derivative(self.y_true, self.y_pred)
        expected = 2 * (self.y_pred - self.y_true) / self.y_true.shape[0]

        self.assertTrue(np.allclose(derivative, expected))


class TestBinaryCrossEntropy(TestCase):

    def setUp(self):
        self.bce = BinaryCrossEntropy()

        self.y_true = np.array([1, 0, 1, 1])
        self.y_pred = np.array([0.9, 0.1, 0.8, 0.7])

    def test_loss(self):
        loss = self.bce.loss(self.y_true, self.y_pred)

        epsilon = 1e-15
        y_pred = np.clip(self.y_pred, epsilon, 1 - epsilon)

        expected = -np.mean(
            self.y_true * np.log(y_pred) +
            (1 - self.y_true) * np.log(1 - y_pred)
        )

        self.assertAlmostEqual(loss, expected, places=6)

    def test_derivative(self):
        derivative = self.bce.derivative(self.y_true, self.y_pred)

        epsilon = 1e-15
        y_pred = np.clip(self.y_pred, epsilon, 1 - epsilon)

        expected = (
            -(self.y_true / y_pred) +
            (1 - self.y_true) / (1 - y_pred)
        ) / self.y_true.shape[0]

        self.assertTrue(np.allclose(derivative, expected))


class TestCategoricalCrossEntropy(TestCase):

    def setUp(self):
        self.cce = CategoricalCrossEntropy()

        self.y_true = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        self.y_pred = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5]
        ])

    def test_loss(self):
        loss = self.cce.loss(self.y_true, self.y_pred)

        epsilon = 1e-15
        y_pred = np.clip(self.y_pred, epsilon, 1 - epsilon)

        expected = -np.mean(
            np.sum(self.y_true * np.log(y_pred), axis=1)
        )

        self.assertAlmostEqual(loss, expected, places=6)

    def test_derivative(self):
        derivative = self.cce.derivative(self.y_true, self.y_pred)

        epsilon = 1e-15
        y_pred = np.clip(self.y_pred, epsilon, 1 - epsilon)

        expected = -self.y_true / y_pred / self.y_true.shape[0]

        self.assertTrue(np.allclose(derivative, expected))
