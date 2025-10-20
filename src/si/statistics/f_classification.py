from si.data.dataset import Dataset
import numpy as np
from typing import Tuple
import scipy

def f_classification(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    classes = dataset.get_classes()
    samples_per_class = []

    for class_ in classes:
        mask = dataset.y == class_
        class_X = dataset.X[mask,:]
        samples_per_class.append(class_X)

    F,p = scipy.stats.f_oneway(*samples_per_class)
    return F,p