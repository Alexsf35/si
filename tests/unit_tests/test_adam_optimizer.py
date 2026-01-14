from unittest import TestCase
import numpy as np

from si.neural_networks.optimizers import Adam


class TestAdamOptimizer(TestCase):

    def setUp(self):
        self.optimizer = Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )

        self.weights = np.array([[1.0, -2.0], [3.0, -4.0]])
        self.gradients = np.array([[0.1, -0.2], [0.3, -0.4]])

    def test_initialization(self):
        """
        Test if Adam initializes internal variables correctly.
        """
        self.assertEqual(self.optimizer.t, 0)
        self.assertIsNone(self.optimizer.m)
        self.assertIsNone(self.optimizer.v)

    def test_first_update(self):
        """
        Test if Adam performs a valid first update.
        """
        updated_weights = self.optimizer.update(self.weights, self.gradients)

        # Check shapes
        self.assertEqual(updated_weights.shape, self.weights.shape)

        # Internal states should be initialized
        self.assertIsNotNone(self.optimizer.m)
        self.assertIsNotNone(self.optimizer.v)

        # Time step should increase
        self.assertEqual(self.optimizer.t, 1)

        # Weights should change
        self.assertFalse(np.allclose(updated_weights, self.weights))

    def test_multiple_updates(self):
        """
        Test if Adam correctly updates parameters over multiple steps.
        """
        w1 = self.optimizer.update(self.weights, self.gradients)
        w2 = self.optimizer.update(w1, self.gradients)

        # Time step should increase
        self.assertEqual(self.optimizer.t, 2)

        # Weights should keep changing
        self.assertFalse(np.allclose(w1, w2))

    def test_no_nan_or_inf(self):
        """
        Ensure numerical stability (no NaN or infinity values).
        """
        w = self.weights
        for _ in range(10):
            w = self.optimizer.update(w, self.gradients)

        self.assertFalse(np.isnan(w).any())
        self.assertFalse(np.isinf(w).any())

    def test_zero_gradient(self):
        """
        If gradient is zero, weights should not change.
        """
        zero_grad = np.zeros_like(self.weights)
        updated_weights = self.optimizer.update(self.weights, zero_grad)

        self.assertTrue(np.allclose(updated_weights, self.weights))
