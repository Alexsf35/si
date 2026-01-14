from typing import Tuple, Iterator

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.neural_networks.layers import Layer
from si.neural_networks.losses import LossFunction, MeanSquaredError
from si.neural_networks.optimizers import Optimizer, SGD
from si.metrics.mse import mse


class NeuralNetwork(Model):
    """
    It represents a neural network model that is made by a sequence of layers.
    """

    def __init__(self, epochs: int = 100, batch_size: int = 128, optimizer: Optimizer = SGD,
                 learning_rate: float = 0.01, verbose: bool = False, loss: LossFunction = MeanSquaredError,
                 metric: callable = mse, **kwargs):
        """
        Initialize the neural network.
        """
        # arguments
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer(learning_rate=learning_rate, **kwargs)
        self.verbose = verbose
        self.loss = loss()
        self.metric = metric

        # attributes
        self.layers = []
        self.history = {}

    def add(self, layer: Layer) -> 'NeuralNetwork':
        """
        Add a layer to the neural network.
        """
        # set the input shape of the layer based on the output layer of the previous layer
        if self.layers:
            # Pega a camada anterior para referência
            previous_layer = self.layers[-1]
            layer.set_input_shape(input_shape=previous_layer.output_shape())

        # initialize the layer with the optimizer (if the layer needs one)
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer)

        # append the layer to self.layers
        self.layers.append(layer)
        return self

    def _create_mini_batches(self, X: np.ndarray, y: np.ndarray = None,
                             shuffle: bool = True) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate mini-batches for the given data (internal helper).
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        assert self.batch_size <= n_samples, "Batch size cannot be greater than the number of samples"
        
        if shuffle:
            np.random.shuffle(indices)
            
        for start_idx in range(0, n_samples - self.batch_size + 1, self.batch_size):
            end_idx = start_idx + self.batch_size
            batch_indices = indices[start_idx:end_idx]
            
            if y is not None:
                yield X[batch_indices], y[batch_indices]
            else:
                yield X[batch_indices], None

    def _forward(self, input_data: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation (internal helper).
        """
        activation = input_data
        for layer in self.layers:
            activation = layer.forward_propagation(activation, training)
        return activation

    def _backward(self, error_signal: float) -> float:
        """
        Perform backward propagation (internal helper).
        """
        current_error = error_signal
        for layer in reversed(self.layers):
            current_error = layer.backward_propagation(current_error)
        return current_error

    def _fit(self, dataset: Dataset) -> 'NeuralNetwork':
        X = dataset.X
        y = dataset.y
        
        # Garante que y é 2D (ex: [N, 1])
        if np.ndim(y) == 1:
            y = y.reshape(-1, 1)

        self.history = {}

        for epoch in range(1, self.epochs + 1):
            # Listas para armazenar previsões e targets reais para calcular a loss da época inteira
            epoch_preds = []
            epoch_targets = []

            # Compute mini batches
            for X_batch, y_batch in self._create_mini_batches(X, y):
                
                # Perform forward propagation
                # training=True garante que camadas como Dropout funcionem corretamente
                predictions = self._forward(X_batch, training=True)

                # Compute the loss derivative error
                # Calcula o erro inicial baseada na função de custo (ex: derivative da MSE)
                loss_derivative = self.loss.derivative(y_batch, predictions)

                # Backpropagate the error
                self._backward(loss_derivative)

                # Armazena resultados parciais para métricas da época
                epoch_preds.append(predictions)
                epoch_targets.append(y_batch)

            #Compute the loss based on true labels and predictions (Full Epoch)
            # Concatena todos os batches para ter o resultado da época inteira
            full_epoch_preds = np.concatenate(epoch_preds)
            full_epoch_targets = np.concatenate(epoch_targets)

            epoch_loss = self.loss.loss(full_epoch_targets, full_epoch_preds)

            #Compute the metric
            if self.metric is not None:
                epoch_metric = self.metric(full_epoch_targets, full_epoch_preds)
                metric_text = f"{self.metric.__name__}: {epoch_metric:.4f}"
            else:
                epoch_metric = 'NA'
                metric_text = "NA"

            # Save the loss and metric in the history dictionary
            self.history[epoch] = {'loss': epoch_loss, 'metric': epoch_metric}

            # Print if verbose
            if self.verbose:
                print(f"Epoch {epoch}/{self.epochs} - loss: {epoch_loss:.4f} - {metric_text}")

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the labels for the given dataset.
        """
        # Calls forward propagation with training=False
        return self._forward(dataset.X, training=False)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Compute the score of the neural network on the given dataset.
        """
        if self.metric is None:
             raise ValueError("No metric specified for the neural network.")
             
        return self.metric(dataset.y, predictions)