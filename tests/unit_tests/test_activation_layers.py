from unittest import TestCase
import numpy as np

from si.neural_networks.activation import TanhActivation, SoftmaxActivation


class TestTanhActivation(TestCase):

    def setUp(self):
        np.random.seed(42)
        self.layer = TanhActivation()
        self.input_data = np.array([
            [-1.0, 0.0, 1.0],
            [2.0, -2.0, 0.5]
        ])
        self.layer.set_input_shape(self.input_data.shape[1:])

    def test_forward_shape(self):
        """Output shape must match input shape"""
        output = self.layer.forward_propagation(self.input_data, training=True)
        self.assertEqual(output.shape, self.input_data.shape)

    def test_output_range(self):
        """Tanh output must be in the interval [-1, 1]"""
        output = self.layer.forward_propagation(self.input_data, training=True)
        self.assertTrue(np.all(output >= -1))
        self.assertTrue(np.all(output <= 1))

    def test_backward_shape(self):
        """Backward propagation output shape must match input"""
        self.layer.forward_propagation(self.input_data, training=True)
        output_error = np.ones_like(self.input_data)
        input_error = self.layer.backward_propagation(output_error)
        self.assertEqual(input_error.shape, self.input_data.shape)

    def test_parameters(self):
        """Activation layers have no trainable parameters"""
        self.assertEqual(self.layer.parameters(), 0)


class TestSoftmaxActivation(TestCase):

    def setUp(self):
        np.random.seed(42)
        self.layer = SoftmaxActivation()
        self.input_data = np.array([
            [1.0, 2.0, 3.0],
            [1000.0, 1000.0, 1000.0]  # numerical stability test
        ])
        self.layer.set_input_shape(self.input_data.shape[1:])

    def test_forward_shape(self):
        """Output shape must match input shape"""
        output = self.layer.forward_propagation(self.input_data, training=True)
        self.assertEqual(output.shape, self.input_data.shape)

    def test_probabilities_sum_to_one(self):
        """Softmax outputs must sum to 1 for each sample"""
        output = self.layer.forward_propagation(self.input_data, training=True)
        row_sums = np.sum(output, axis=1)
        self.assertTrue(np.allclose(row_sums, np.ones(output.shape[0])))

    def test_output_range(self):
        """Softmax outputs must be between 0 and 1"""
        output = self.layer.forward_propagation(self.input_data, training=True)
        self.assertTrue(np.all(output >= 0))
        self.assertTrue(np.all(output <= 1))

    def test_numerical_stability(self):
        """Softmax must not produce NaN or Inf values"""
        output = self.layer.forward_propagation(self.input_data, training=True)
        self.assertFalse(np.any(np.isnan(output)))
        self.assertFalse(np.any(np.isinf(output)))

    def test_backward_shape(self):
        """Backward propagation output shape must match input"""
        self.layer.forward_propagation(self.input_data, training=True)
        output_error = np.ones_like(self.input_data)
        input_error = self.layer.backward_propagation(output_error)
        self.assertEqual(input_error.shape, self.input_data.shape)

    def test_parameters(self):
        """Softmax has no trainable parameters"""
        self.assertEqual(self.layer.parameters(), 0)
