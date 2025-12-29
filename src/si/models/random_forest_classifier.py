import numpy as np
from typing import Literal
from statistics import mode
from si.metrics.accuracy import accuracy
from si.base.model  import Model
from si.data.dataset import Dataset
from si.models.decision_tree_classifier import DecisionTreeClassifier

class RandomForestClassifier(Model):
    """
    Random Forest Classifier

    A random-forest classifier that builds an ensemble of decision trees,
    each trained on a bootstrap sample of the dataset and using a random
    subset of features. Predictions are made by majority voting across trees.

    Parameters
    ----------
    n_estimators : int
        Number of decision trees (estimators) to train.
    max_features : int
        Number of random features to use in each tree. If None, defaults to sqrt(n_features).
    min_sample_split : int, default=2
        Minimum number of samples required to perform a split in a decision tree.
    max_depth : int, default=10
        Maximum depth allowed for each decision tree.
    mode : {'gini', 'entropy'}, default='gini'
        Impurity measure used to perform splits in the decision trees.
    seed : int, default=42
        Random seed for reproducibility.
    **kwargs :
        Additional arguments passed to the parent Model class.

    Estimated Parameters
    --------------------
    trees : list
        A list of tuples (features_i, tree), where:
            - features_i: indices of the features used by the tree
            - tree: trained DecisionTreeClassifier instance
    """

    def __init__(self, n_estimators : int, max_features : int, min_sample_split: int = 2, max_depth: int = 10, mode: Literal['gini', 'entropy'] = 'gini', seed: int=42, **kwargs):


        super().__init__(**kwargs)

        self.n_estimators = n_estimators 
        self.max_features = max_features 
        self.min_sample_split = min_sample_split 
        self.max_depth = max_depth 
        self.mode = mode 
        self.seed = seed 

        self.trees = []


    def _fit(self, dataset: Dataset) -> 'RandomForestClassifier' :

        """
        Trains the random forest classifier.

        The method creates bootstrap samples of the dataset, randomly selects
        a subset of features for each tree, trains a decision tree on each
        bootstrap sample, and stores the resulting ensemble.

        Parameters
        ----------
        dataset : Dataset
            The dataset to train the ensemble of trees.

        Returns
        -------
        self : RandomForestClassifier
            The fitted model.
        """
        np.random.seed(self.seed)

        if self.max_features == None:
            self.max_features = int(np.sqrt(len(dataset.features)))

        for _ in range(self.n_estimators):

            samples_i= np.random.choice(dataset.shape()[0], size=dataset.shape()[0], replace=True)
            features_i= np.random.choice(dataset.shape()[1], size=self.max_features, replace=False)

            X_bootstrap = dataset.X[samples_i][:, features_i]
            y_bootstrap = dataset.y[samples_i]
            features_bootstrap = [dataset.features[i] for i in features_i]

            bootstrap_dataset = Dataset(
                X_bootstrap,
                y_bootstrap,
                features=features_bootstrap,
                label=dataset.label
            )

            tree=DecisionTreeClassifier(
                min_sample_split=self.min_sample_split,
                max_depth=self.max_depth,
                mode=self.mode
                )
            
            tree._fit(bootstrap_dataset)

            self.trees.append((features_i,tree))

        return self
    
    def _predict(self, dataset: Dataset) -> np.array: 
        """
        Predicts the labels for a given dataset using majority voting.

        For each tree in the forest, predictions are obtained using only the 
        features that were selected during training. The final prediction for 
        each sample is the most common class (mode) across all tree predictions.

        Parameters
        ----------
        dataset : Dataset
            Dataset to obtain predictions for.

        Returns
        -------
        np.ndarray
            Array of predicted class labels (one label per sample).
        """

        all_predictions=[]

        for features_i, tree in self.trees:
            dataset_sub = Dataset(dataset.X[:,features_i], y=None)

            predictions= tree.predict(dataset_sub)

            all_predictions.append(predictions)
        
        all_predictions=np.array(all_predictions)

        # ou so [mode(preds) for preds in all_predictions.T]
        return [int(mode(preds)) if type(mode(preds))==np.int64 or type(mode(preds))==np.float64 else str(mode(preds)) for preds in all_predictions.T]
    
    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculates the accuracy of the model on a dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to calculate the accuracy on.
        predictions: np.ndarray
            Predictions

        Returns
        -------
        float
            The accuracy of the model on the dataset.
        """
        return accuracy(dataset.y, predictions)
    
if __name__ == '__main__':
    from si.io.csv_file import read_csv
    from si.model_selection.split import train_test_split

    # Load dataset
    data = read_csv('C:/Users/Alexandre/Documents/GitHub/si/datasets/iris/iris.csv', sep=',', features=True, label=True)


    # Train-test split
    train, test = train_test_split(data, test_size=0.33, random_state=42)

    # Create Random Forest model
    model = RandomForestClassifier(
        n_estimators=10,
        max_features=2,
        min_sample_split=2,
        max_depth=5,
        mode='gini',
        seed=42
    )

    # Fit model
    model.fit(train)

    # Make predictions and compute accuracy
    score = model.score(test)

    print("Random Forest accuracy:", score)
