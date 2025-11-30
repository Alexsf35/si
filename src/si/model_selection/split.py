from typing import Tuple
import random

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test


def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Splits the dataset into training and testing sets while preserving the
    proportion of classes in the target variable (y). This is called
    stratified sampling.

    Parameters
    ----------
    dataset : Dataset
        The dataset to split.
    test_size : float, default=0.2
        The proportion of the dataset to include in the test split.
        Must be between 0.0 and 1.0.
    random_state : int, default=42
        Controls the randomness of the splitting process. Useful for reproducibility.

    Returns
    -------
    Tuple[Dataset, Dataset]
        A tuple containing the training and testing datasets, respectively.
    """
    labels,counts=np.unique(dataset.y, return_counts=True)

    train_i=[]
    test_i=[]

    np.random.seed(random_state)

    for i in range(len(labels)):
        n_test_samples = int(round(test_size * counts[i]))
        class_i = np.where(dataset.y == labels[i])[0]
        shuffled_i=np.random.permutation(class_i)
        test_i.extend(shuffled_i[:n_test_samples])
        train_i.extend(shuffled_i[n_test_samples:])

    train_dataset=Dataset(X=dataset.X[train_i], y= dataset.y[train_i], features=dataset.features, label=dataset.label)
    test_dataset=Dataset(X=dataset.X[test_i], y= dataset.y[test_i], features=dataset.features, label=dataset.label)

    return train_dataset,test_dataset
    

    
