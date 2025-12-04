import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset

class PCA(Transformer):
    """
    Principal Component Analysis (PCA) implemented using
    eigenvalue decomposition of the covariance matrix.

    Parameters
    ----------
    n_components : int
        Number of principal components to retain.

    Attributes
    ----------
    mean : array-like, shape (n_features,)
        Mean vector of the training samples.

    components : array-like, shape (n_components, n_features)
        Principal components (each row is an eigenvector).

    explained_variance : array-like, shape (n_components,)
        Variance explained by each selected principal component.
    """
    def __init__(self,n_components: int, **kwargs)-> None:

        super().__init__(**kwargs)

        self.n_components = n_components

        self.mean = None
        self.components = None
        self.explained_variance = None

    def _fit(self, dataset: Dataset)-> 'PCA':
        """
        Fit PCA by computing the mean, covariance matrix,
        eigenvalues and eigenvectors of the centered data.

        Parameters
        ----------
        dataset : Dataset
            Input dataset used for estimating the principal components.

        Returns
        -------
        self : PCA
            Fitted PCA transformer.
        """
        X = dataset.X
        self.mean = np.mean(X,axis=0)
        centered_data = X - self.mean

        covariance_matrix=np.cov(centered_data, rowvar=False)
        eigen_values,eigen_vectors=np.linalg.eig(covariance_matrix)

        sorted_i=np.argsort(eigen_values)[::-1]

        eigen_values=eigen_values[sorted_i]
        eigen_vectors=eigen_vectors[:,sorted_i]

        # cada coluna é um componente (PC)
        self.components= eigen_vectors[:,:self.n_components].T

        total_variance = np.sum(eigen_values)
        self.explained_variance = (eigen_values / total_variance)[:self.n_components]

        return self
    
    def _transform(self, dataset) -> np.array:
        """
        Transform the dataset into its lower-dimensional
        representation using the learned principal components.

        Parameters
        ----------
        dataset : Dataset
            Dataset to project onto the selected principal components.

        Returns
        -------
        X_reduced : ndarray, shape (n_samples, n_components)
            The dataset represented in the PCA subspace.
        """
        X = dataset.X
        centered_data= X - self.mean 

        return np.dot(centered_data,self.components.T)








