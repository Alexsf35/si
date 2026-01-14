from unittest import TestCase
import numpy as np
import os

from datasets import DATASETS_PATH

from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split

from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.ensemble.stacking_classifier import StackingClassifier


class TestStackingClassifier(TestCase):

    def setUp(self):
        """
        Load the breast cancer binary dataset
        """
        self.csv_file = os.path.join(
            DATASETS_PATH, 'breast_bin', 'breast-bin.csv'
        )

        self.dataset = read_csv(
            filename=self.csv_file,
            features=True,
            label=True
        )

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

        # Base models
        self.knn = KNNClassifier(k=3)
        self.log_reg = LogisticRegression()
        self.dt = DecisionTreeClassifier()

        # Final model
        self.knn_final = KNNClassifier(k=3)

        self.stacking = StackingClassifier(
            models=[self.knn, self.log_reg, self.dt],
            final_model=self.knn_final
        )

    def test_fit(self):
        """
        Test if stacking classifier fits without errors
        """
        self.stacking.fit(self.train_dataset)
        self.assertTrue(self.stacking.is_fitted())

    def test_predict(self):
        """
        Test if predict returns correct number of predictions
        """
        self.stacking.fit(self.train_dataset)

        predictions = self.stacking.predict(self.test_dataset)

        self.assertEqual(predictions.shape[0], self.test_dataset.y.shape[0])

    def test_predict_values(self):
        """
        Test if predictions are valid class labels
        """
        self.stacking.fit(self.train_dataset)

        predictions = self.stacking.predict(self.test_dataset)

        unique_labels = np.unique(self.dataset.y)
        self.assertTrue(np.all(np.isin(predictions, unique_labels)))

    def test_score(self):
        """
        Test if score returns a value between 0 and 1
        """
        self.stacking.fit(self.train_dataset)

        score = self.stacking.score(self.test_dataset)

        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
