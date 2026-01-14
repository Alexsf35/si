from unittest import TestCase
import numpy as np
import os

from datasets import DATASETS_PATH

from si.io.csv_file import read_csv
from si.models.knn_classifier import KNNClassifier
from si.metrics.accuracy import accuracy
from si.model_selection.randomized_search import randomized_search_cv


class TestRandomizedSearchCV(TestCase):

    def setUp(self):
        # dataset
        self.csv_file = os.path.join(
            DATASETS_PATH,
            'breast_bin',
            'breast-bin.csv'
        )

        self.dataset = read_csv(
            filename=self.csv_file,
            features=True,
            label=True
        )

        # model
        self.model = KNNClassifier()

        # hyperparameter grid
        self.param_grid = {
            "k": [1, 3, 5, 7]
        }

    def test_output_structure(self):
        """Test if output dictionary has correct keys"""

        results = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_grid=self.param_grid,
            scoring=accuracy,
            cv=3,
            n_iter=2
        )

        self.assertIsInstance(results, dict)
        self.assertIn("hyperparameters", results)
        self.assertIn("scores", results)
        self.assertIn("best_hyperparameters", results)
        self.assertIn("best_score", results)

    def test_number_of_iterations(self):
        """Test if number of evaluated combinations equals n_iter"""

        n_iter = 3

        results = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_grid=self.param_grid,
            scoring=accuracy,
            cv=3,
            n_iter=n_iter
        )

        self.assertEqual(len(results["scores"]), n_iter)
        self.assertEqual(len(results["hyperparameters"]), n_iter)

    def test_scores_range(self):
        """Test if scores are valid accuracy values"""

        results = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_grid=self.param_grid,
            scoring=accuracy,
            cv=3,
            n_iter=3
        )

        for score in results["scores"]:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_best_score_consistency(self):
        """Test if best_score is max of scores"""

        results = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_grid=self.param_grid,
            scoring=accuracy,
            cv=3,
            n_iter=3
        )

        self.assertEqual(
            results["best_score"],
            max(results["scores"])
        )

    def test_best_hyperparameters_valid(self):
        """Test if best hyperparameters come from tested grid"""

        results = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_grid=self.param_grid,
            scoring=accuracy,
            cv=3,
            n_iter=3
        )

        best_params = results["best_hyperparameters"]

        self.assertIn("k", best_params)
        self.assertIn(best_params["k"], self.param_grid["k"])

    def test_invalid_hyperparameter_raises_error(self):
        """Test if invalid hyperparameter raises ValueError"""

        invalid_grid = {
            "invalid_param": [1, 2, 3]
        }

        with self.assertRaises(ValueError):
            randomized_search_cv(
                model=self.model,
                dataset=self.dataset,
                hyperparameter_grid=invalid_grid,
                scoring=accuracy,
                cv=3,
                n_iter=2
            )
