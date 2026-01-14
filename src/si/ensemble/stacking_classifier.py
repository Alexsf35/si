from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
import numpy as np


class StackingClassifier(Model):
    """
    Stacking Classifier

    The StackingClassifier is an ensemble learning method that combines multiple
    base classifiers by training a final classifier on their predictions.
    The base models are first trained on the original dataset, and their
    predictions are then used as input features for the final model, which
    produces the final prediction.

    This approach allows the model to leverage the strengths of different
    classifiers and often results in improved predictive performance.

    Parameters
    ----------
    models: list of Model
        A list of base models to be trained in the first level of the stacking ensemble
    final_model: Model
        The model responsible for making the final predictions based on the
        outputs of the base models

    Attributes
    ----------
    models: list of Model
        The base models of the stacking ensemble
    final_model: Model
        The final model trained on the predictions of the base models
    """

    def __init__(self, models, final_model):
        """
        Initialize the StackingClassifier

        Parameters
        ----------
        models: list of Model
            A list of base models to be used in the stacking ensemble
        final_model: Model
            The model used to combine the predictions of the base models
            and produce the final prediction
        """
        self.models = models
        self.final_model = final_model

    def _get_meta_X(self, X) -> np.ndarray:
        """
        It generates the meta-feature matrix from the predictions of the base models

        Each column of the resulting matrix corresponds to the predictions of one
        base model over the given input samples.

        Parameters
        ----------
        X: np.ndarray (n_samples, n_features)
            The feature matrix from which the meta-features will be generated

        Returns
        -------
        meta_X: np.ndarray (n_samples, n_models)
            The meta-feature matrix composed of the base models' predictions
        """
        base_preds = []
        for model in self.models:
            base_preds.append(model.predict(Dataset(X)))
        return np.column_stack(base_preds)

    def _fit(self, dataset: Dataset) -> 'StackingClassifier':
        
        """
        It fits the stacking classifier to the given dataset

        The method first trains all base models using the original dataset.
        Then, it builds a new dataset composed of the predictions of the base
        models and trains the final model on this meta-dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the stacking classifier to

        Returns
        -------
        self: StackingClassifier
            The fitted stacking classifier
        """
        for model in self.models:
            model.fit(dataset)

        meta_X = self._get_meta_X(dataset.X)
        meta_dataset = Dataset(meta_X, dataset.y)

        self.final_model.fit(meta_dataset)

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the classes of the given dataset using the stacking ensemble

        The predictions are obtained by first generating predictions from the
        base models, which are then combined and passed to the final model to
        produce the final predictions.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes of

        Returns
        -------
        predictions: np.ndarray
            The predicted class labels
        """
        meta_X = self._get_meta_X(dataset.X)
        meta_dataset = Dataset(meta_X)
        return self.final_model.predict(meta_dataset)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        It returns the accuracy of the stacking classifier on the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the stacking classifier on
        
        predictions: np.ndarray
            An array containing the predicted labels

        Returns
        -------
        accuracy: float
            The accuracy of the stacking classifier
        """
        return accuracy(dataset.y, predictions)
