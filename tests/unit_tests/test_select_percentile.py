from unittest import TestCase
import numpy as np
import os
from typing import Tuple

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification 
from si.feature_selection.select_percentile import SelectPercentile

# --- Funções Auxiliares para Teste de Desempate ---

def _mock_f_classification_for_ties(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Função mock para gerar scores controlados que forçam um empate na fronteira
    de seleção (k=3), garantindo que a lógica de ordenação por índice funcione.
    """
    # Scores: [10.0 (F0), 5.0 (F1), 7.5 (F2), 7.5 (F3)]. Queremos os 3 melhores (75%).
    F_scores = np.array([10.0, 5.0, 7.5, 7.5])
    p_values = np.zeros(4) 
    return F_scores, p_values

# --- CLASSE DE TESTE ---

class TestSelectPercentile(TestCase):

    def setUp(self):
        # Carrega o Iris dataset para uso em todos os testes
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)
        # O dataset tem 4 features: 0, 1, 2, 3

    def test_fit_stores_scores(self):
        """Testa se o fit executa o f_classification e armazena F e p."""
        
        selector = SelectPercentile(score_func=f_classification)
        selector.fit(self.dataset)
        
        # Verifica se os scores foram calculados e têm o tamanho correto (4 scores)
        self.assertIsNotNone(selector.F)
        self.assertIsNotNone(selector.p)
        self.assertEqual(len(selector.F), self.dataset.shape[1])

    def test_transform_percentile_50_iris(self):
        """Testa a seleção de 50% de features (2 features de 4) no Iris."""
        
        selector = SelectPercentile(score_func=f_classification, percentile=50)
        selector.fit(self.dataset) 
        
        transformed_dataset = selector.transform(self.dataset)
        
        # 1. Verificar o número de features selecionadas (k=2)
        self.assertEqual(transformed_dataset.X.shape[1], 2)
        
        # 2. Verificar os nomes das features selecionadas (as mais importantes no Iris)
        expected_feature_names = ['petal_length', 'petal_width']
        self.assertEqual(transformed_dataset.features, expected_feature_names)

    def test_transform_with_tie_breaking(self):
        """
        Testa a lógica de desempate, garantindo que o número exato de features seja selecionado.
        (75% de 4 = 3 features. O 2º e 3º melhores têm o mesmo score).
        """
        
        # O mock garante que [F0 (10.0), F2 (7.5), F3 (7.5)] são selecionados.
        selector = SelectPercentile(score_func=_mock_f_classification_for_ties, percentile=75)
        selector.fit(self.dataset) 
        
        transformed_dataset = selector.transform(self.dataset)
        
        # 1. Esperamos k=3 features
        self.assertEqual(transformed_dataset.X.shape[1], 3)
        
        # 2. Nomes esperados: 'sepal_length' (F0), 'petal_length' (F2), 'petal_width' (F3)
        expected_feature_names = ['sepal_length', 'petal_length', 'petal_width']
        self.assertEqual(transformed_dataset.features, expected_feature_names)
        
    def test_edge_case_percentile_0_and_100(self):
        """Testa os casos limite 0% (nenhuma) e 100% (todas)."""
        
        # 1. Teste 0%
        selector_0 = SelectPercentile(score_func=f_classification, percentile=0)
        selector_0.fit(self.dataset)
        transformed_0 = selector_0.transform(self.dataset)
        self.assertEqual(transformed_0.X.shape[1], 0)
        
        # 2. Teste 100%
        selector_100 = SelectPercentile(score_func=f_classification, percentile=100)
        selector_100.fit(self.dataset)
        transformed_100 = selector_100.transform(self.dataset)
        self.assertEqual(transformed_100.X.shape[1], self.dataset.X.shape[1])
        self.assertTrue(np.allclose(transformed_100.X, self.dataset.X))