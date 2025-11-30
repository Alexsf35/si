import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse

class RidgeRegressionLeastSquares(Model):

    """
    Ridge Regression using the closed-form Least Squares solution.

    This model fits a linear regression with L2 regularization
    (Ridge penalty) and optionally scales the data before training.
    """

    def __init__(self, l2_penalty : float=1, scale : bool=True, **kwargs) -> None:
        """
        Initialize the Ridge Regression model.

        Parameters
        ----------
        l2_penalty : float
            L2 regularization strength (lambda).
        scale : bool
            Whether to scale features before training.
        """
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.scale = scale

        self.theta=None
        self.theta_zero=None
        self.mean=None
        self.std=None


    def _fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Fit the model to the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: RidgeRegressionLeastSquares
            The fitted model
        """
        if self.scale:
            # compute mean and std
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            # scale the dataset
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X


        X_intercept= np.c_[np.ones(X.shape[0]),X]

        penalty= self.l2_penalty * np.eye(np.shape(X_intercept)[1])
        penalty[0,0]=0

        A = X_intercept.T.dot(X_intercept) + penalty
        B = X_intercept.T.dot(dataset.y)
        theta_vector = np.linalg.inv(A).dot(B)

        self.theta_zero = theta_vector[0]
        self.theta = theta_vector[1:]


    def _predict(self, dataset: Dataset)-> np.array:
        """
        Predict the target values for a new dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to predict on.

        Returns
        -------
        y_pred : np.ndarray
            Predicted target values.
        """
        if self.scale:
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        X_intercept= np.c_[np.ones(X.shape[0]),X]

        thetas=np.r_[self.theta_zero, self.theta]

        pred_y= X_intercept.dot(thetas)

        return pred_y
    

    def _score(self, dataset: Dataset, predictions) -> float:
        """
        Compute the Mean Square Error of the model on the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the MSE on
            
        Returns
        -------
        mse: float
            The Mean Square Error of the model
        """
        return mse(dataset.y, predictions)


