from typing import Callable

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance
import numpy as np

class KNNregressor(Model):
    """
    K-Nearest Neighbors Regressor.

    This model predicts the target value of a sample based on the average
    of the target values of its 'k' nearest neighbors in the training dataset.

    Parameters
    ----------
    k: int, default=1
        The number of nearest neighbors to use for prediction.
    distance: Callable, default=euclidean_distance
        The distance function to use for calculating the distance between samples.
        It should take two numpy arrays (samples) and return a float distance.

    Attributes
    ----------
    dataset: Dataset
        The training dataset stored during the fit phase.
    """

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        """
        Initialize the KNN regressor

        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use
        """
        # parameters
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance

        self.dataset = None


    def _fit(self, dataset: Dataset):
        """
        Store the training dataset.

        Parameters
        ----------
        dataset : Dataset
            Training dataset.

        Returns
        -------
        self : KNNregressor
            The fitted model.
        """
        self.dataset=dataset

        return self
    
    def _predict(self, dataset:Dataset) -> np.array:
        """
        Predict the target values for a test dataset.

        Parameters
        ----------
        dataset : Dataset
            Test dataset.

        Returns
        -------
        numpy.ndarray
            Predicted values (y_pred).
        """
        predictions=[]

        for sample in dataset.X:
            distances = self.distance(sample, self.dataset.X)
            nearest_neighbors_i = np.argsort(distances)[:self.k]
            mean_nearest_neighbors = np.mean(self.dataset.y[nearest_neighbors_i])
            predictions.append((mean_nearest_neighbors))

        return np.array(predictions)
    
    def _score(self, dataset: Dataset, predictions) -> float:
        """
        Compute the RMSE error on a test dataset.

        Parameters
        ----------
        dataset : Dataset
            Test dataset.

        Returns
        -------
        float
            Root Mean Squared Error between predictions and true values.
        """
        y_true= dataset.y
        return rmse(y_true, predictions)
