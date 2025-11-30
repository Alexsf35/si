import unittest

import numpy as np

from si.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())


    def test_dropna(self):
        X = np.array([
            [1, 2, np.nan],
            [4, 5, 6],
            [np.nan, 1, 2]
            ])
        y = np.array([1, 2, 3])
        dataset = Dataset(X, y)

        dataset.dropna()

        # Depois do dropna só deve restar a linha [4,5,6]
        self.assertEqual(dataset.X.shape, (1, 3))
        self.assertEqual(dataset.y.shape, (1,))
        self.assertTrue(np.array_equal(dataset.X[0], np.array([4, 5, 6])))

    def test_fillna_value(self):
        X = np.array([
            [1, np.nan, 3],
            [4, 5, np.nan]
        ])
        dataset = Dataset(X, y=None)

        dataset.fillna(value=0)

        # Todos os NaN devem ser substituídos por 0
        self.assertFalse(np.isnan(dataset.X).any())
        self.assertTrue(np.array_equal(
            dataset.X,
            np.array([[1, 0, 3], [4, 5, 0]])
        ))

    def test_fillna_mean(self):
        X = np.array([
            [1, np.nan, 3],
            [3, 5, np.nan]
        ])
        dataset = Dataset(X, y=None)

        dataset.fillna(mean=True)

        # mean da coluna 1 = (5)/1 = 5
        # mean da coluna 2 = (3)/1 = 3
        self.assertEqual(dataset.X[0, 1], 5)
        self.assertEqual(dataset.X[1, 2], 3)

    def test_fillna_median(self):
        X = np.array([
            [1, np.nan, 3],
            [3, 7, np.nan]
        ])
        dataset = Dataset(X, y=None)

        dataset.fillna(median=True)

        # mediana da coluna 1 = 7
        # mediana da coluna 2 = 3
        self.assertEqual(dataset.X[0, 1], 7)
        self.assertEqual(dataset.X[1, 2], 3)

    def test_remove_by_index(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([10, 20, 30])
        dataset = Dataset(X, y)

        dataset.remove_by_index(1)  # remove a segunda linha

        self.assertEqual(dataset.X.shape, (2, 2))
        self.assertEqual(dataset.y.shape, (2,))
        self.assertTrue(np.array_equal(dataset.X, np.array([[1, 2], [5, 6]])))
        self.assertTrue(np.array_equal(dataset.y, np.array([10, 30])))
