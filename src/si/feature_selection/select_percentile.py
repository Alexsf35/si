from typing import Callable
from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectPercentile(Transformer):
    """
    K-Percentile Feature Selection transformer.

    This transformer selects a subset of features based on the highest scores
    computed by a scoring function (e.g., F-classification), retaining a specified
    percentage of the total features. 
    
    The selection mechanism is designed to handle ties at the threshold to ensure
    that the exact number of features defined by the percentile is retained.

    Parameters
    ----------
    score_func : callable, default=f_classification
        Function taking a dataset (X, y) and returning arrays of feature scores (F-scores)
        and p-values.
        
    percentile : int, default=10
        Percentile (between 0 and 100) indicating the proportion of features to retain.
        E.g., percentile=40 means the top 40% of features will be selected.

    Attributes
    ----------
    F : numpy.ndarray, shape (n_features,)
        The F-scores (or feature scores) computed for each feature in the training dataset.

    p : numpy.ndarray, shape (n_features,)
        The p-values associated with the F-scores.
    """
    def __init__(self, score_func: Callable = f_classification, percentile: int = 10,**kwargs):
        """
        Initialize SelectPercentile.

        Parameters
        ----------
        score_func : callable, default=f_classification
            Function taking a dataset and returning arrays of F-scores and p-values.
        
        percentile : int, default=10
            Percentile (between 0 and 100) indicating the proportion of features to retain.
        """
        super().__init__(**kwargs)
        self.score_func=score_func
        self.percentile=percentile
        self.F = None
        self.p = None

    def _fit(self, dataset:Dataset) -> 'SelectPercentile':
        """
        Compute the F-scores and p-values of all features using the scoring function.

        Parameters
        ----------
        dataset : Dataset
            A labeled dataset.

        Returns
        -------
        self : object
            Returns self.
        """
        self.F,self.p = self.score_func(dataset)
        return self


    def _transform(self,dataset:Dataset) -> Dataset:
        """
        Transform the dataset by selecting the top percentile of features according to F-values.
    
        Parameters
        ----------
        dataset : Dataset
            A labeled dataset.

        Returns
        -------
        dataset : Dataset
            A new dataset containing only the selected top-percentile features.
        """

        scores = [(self.F[i], i) for i in range(len(self.F))]

        scores.sort(key=lambda x: (x[0],x[1]))

        k = len(scores) - int((self.percentile*len(scores))/100)

        features_i= sorted([index for score, index in scores[k:]])

        new_X=dataset.X[:,features_i]
        new_features=dataset.features[features_i]

        return Dataset(X=new_X, y=dataset.y, features=new_features, label=dataset.label)
