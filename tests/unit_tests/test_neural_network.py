from unittest import TestCase
import numpy as np
import os

from si.data.dataset import Dataset
from si.io.csv_file import read_csv
from si.neural_networks.layers import DenseLayer
from si.neural_networks.losses import MeanSquaredError
from si.neural_networks.optimizers import SGD
from si.metrics.mse import mse
from si.neural_networks.neural_network import NeuralNetwork
from sklearn.preprocessing import LabelEncoder

class TestNeuralNetwork(TestCase):

    def setUp(self):
        # Usar dataset pequeno para testes rápidos
        self.csv_file = os.path.join(os.path.dirname(__file__), "../../datasets/iris/iris.csv")
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)
        
        # Reduzir tamanho para testes
        self.dataset.X = self.dataset.X[:30]
        self.dataset.y = self.dataset.y[:30]

        # Converte labels string para números
        le = LabelEncoder()
        self.dataset.y = le.fit_transform(self.dataset.y)

        # Configura rede simples
        self.nn = NeuralNetwork(
            epochs=2,
            batch_size=10,
            optimizer=SGD,
            learning_rate=0.01,
            verbose=False,
            loss=MeanSquaredError,
            metric=mse
        )
        # Adiciona camada densa
        self.nn.add(DenseLayer(n_units=5, input_shape=(self.dataset.X.shape[1],)))
        self.nn.add(DenseLayer(n_units=1))
        
    def test_add_layer(self):
        nn_test = NeuralNetwork()
        dense = DenseLayer(n_units=3, input_shape=(4,))
        nn_test.add(dense)
        self.assertEqual(len(nn_test.layers), 1)
        self.assertEqual(nn_test.layers[0].output_shape(), (3,))

    def test_fit(self):
        self.nn._fit(self.dataset)
        # verifica se history foi atualizado
        self.assertEqual(len(self.nn.history), self.nn.epochs)
        for epoch_data in self.nn.history.values():
            self.assertIn('loss', epoch_data)
            self.assertIn('metric', epoch_data)
            self.assertIsInstance(epoch_data['loss'], float)
            self.assertIsInstance(epoch_data['metric'], float)

    def test_predict(self):
        self.nn._fit(self.dataset)
        preds = self.nn._predict(self.dataset)
        self.assertEqual(preds.shape[0], self.dataset.X.shape[0])
        # Saída deve ter 1 coluna (porque a última DenseLayer tem n_units=1)
        self.assertEqual(preds.shape[1], 1)

    def test_score(self):
        self.nn._fit(self.dataset)
        preds = self.nn._predict(self.dataset)
        score_val = self.nn._score(self.dataset, preds)
        self.assertIsInstance(score_val, float)
        # Métrica (MSE) não pode ser negativa
        self.assertGreaterEqual(score_val, 0.0)
