from unittest import TestCase
import numpy as np

from si.neural_networks.layers import Dropout


class TestDropout(TestCase):

    def setUp(self):
        np.random.seed(42)
        self.input_data = np.ones((10, 5))
        self.dropout_prob = 0.5
        self.dropout = Dropout(probability=self.dropout_prob)
        self.dropout.set_input_shape(self.input_data.shape[1:])

    def test_forward_training_shape(self):
        """Output shape must be equal to input shape during training"""
        output = self.dropout.forward_propagation(
            self.input_data,
            training=True
        )
        self.assertEqual(output.shape, self.input_data.shape)

    def test_forward_inference_identity(self):
        """During inference, dropout must not change the input"""
        output = self.dropout.forward_propagation(
            self.input_data,
            training=False
        )
        self.assertTrue(np.array_equal(output, self.input_data))

    def test_mask_values(self):
        """Mask should contain only 0s and 1s"""
        _ = self.dropout.forward_propagation(
            self.input_data,
            training=True
        )
        unique_values = np.unique(self.dropout.mask)
        self.assertTrue(np.all(np.isin(unique_values, [0, 1])))

    def test_dropout_effect(self):
        """Some values should be dropped (set to zero) during training"""
        output = self.dropout.forward_propagation(
            self.input_data,
            training=True
        )
        self.assertTrue(np.any(output == 0))

    def test_backward_propagation(self):
        """Backward propagation should apply the same mask"""
        self.dropout.forward_propagation(
            self.input_data,
            training=True
        )
        output_error = np.ones_like(self.input_data)
        input_error = self.dropout.backward_propagation(output_error)

        self.assertTrue(
            np.array_equal(input_error, output_error * self.dropout.mask)
        )

    def test_parameters(self):
        """Dropout has no trainable parameters"""
        self.assertEqual(self.dropout.parameters(), 0)

    def test_output_shape(self):
        """Output shape must match input shape"""
        self.assertEqual(
            self.dropout.output_shape(),
            self.input_data.shape[1:]
        )
