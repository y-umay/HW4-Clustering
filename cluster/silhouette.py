import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("Input X must be a 2D numpy array")
        if not isinstance(y, np.ndarray) or y.ndim != 1:
            raise ValueError("Input y must be a 1D numpy array")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of observations")
        
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            raise ValueError("Silhouette score requires at least 2 clusters")
        
        silhouette_scores = np.zeros(len(X))
        
        for i, x in enumerate(X):
            cluster = y[i]
            in_cluster = X[y == cluster]
            a = np.mean(cdist([x], in_cluster, metric='euclidean')) if len(in_cluster) > 1 else 0
            
            b = np.inf
            for other_cluster in unique_labels:
                if other_cluster == cluster:
                    continue
                other_cluster_points = X[y == other_cluster]
                if len(other_cluster_points) > 0:
                    b = min(b, np.mean(cdist([x], other_cluster_points, metric='euclidean')))
            
            silhouette_scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0
        
        return silhouette_scores

