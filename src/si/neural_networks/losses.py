from abc import ABCMeta, abstractmethod
import numpy as np


class LossFunction(metaclass=ABCMeta):
    """
    Base class for loss functions.
    """

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the loss value.

        Parameters
        ----------
        y_true: np.ndarray
            True labels
        y_pred: np.ndarray
            Predicted labels

        Returns
        -------
        float
            Loss value
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the loss with respect to the predictions.

        Parameters
        ----------
        y_true: np.ndarray
            True labels
        y_pred: np.ndarray
            Predicted labels

        Returns
        -------
        np.ndarray
            Derivative of the loss
        """
        raise NotImplementedError



class MeanSquaredError(LossFunction):
    """
    Mean Squared Error loss function.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the mean squared error loss.

        Parameters
        ----------
        y_true: np.ndarray
            True labels
        y_pred: np.ndarray
            Predicted labels

        Returns
        -------
        float
            Mean squared error
        """
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the mean squared error loss.

        Parameters
        ----------
        y_true: np.ndarray
            True labels
        y_pred: np.ndarray
            Predicted labels

        Returns
        -------
        np.ndarray
            Derivative of the loss with respect to y_pred
        """
        return (2 / y_true.shape[0]) * (y_pred - y_true)



class BinaryCrossEntropy(LossFunction):
    """
    Binary Cross-Entropy loss function.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the binary cross-entropy loss.

        Parameters
        ----------
        y_true: np.ndarray
            True binary labels (0 or 1)
        y_pred: np.ndarray
            Predicted probabilities

        Returns
        -------
        float
            Binary cross-entropy loss
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        return -np.mean(y_true * np.log(y_pred) +(1 - y_true) * np.log(1 - y_pred))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the binary cross-entropy loss.

        Parameters
        ----------
        y_true: np.ndarray
            True binary labels
        y_pred: np.ndarray
            Predicted probabilities

        Returns
        -------
        np.ndarray
            Derivative of the loss with respect to y_pred
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        return (y_pred - y_true) / (y_pred * (1 - y_pred)) / y_true.shape[0]



class CategoricalCrossEntropy(LossFunction):
    """
    Categorical Cross-Entropy loss function.
    Used for multi-class classification with one-hot encoded labels.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the categorical cross-entropy loss.

        Parameters
        ----------
        y_true: np.ndarray
            True labels (one-hot encoded)
        y_pred: np.ndarray
            Predicted class probabilities

        Returns
        -------
        float
            Categorical cross-entropy loss
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the categorical cross-entropy loss.

        Parameters
        ----------
        y_true: np.ndarray
            True labels (one-hot encoded)
        y_pred: np.ndarray
            Predicted class probabilities

        Returns
        -------
        np.ndarray
            Derivative of the loss with respect to y_pred
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        return -y_true / y_pred / y_true.shape[0]
