# Write your k-means unit tests here
import numpy as np
import pytest
from cluster import KMeans

'''
To run python -m pytest -v test/* 
'''

class TestKMeans:
    def setup_method(self):
        self.X = np.array([
            [1, 2], [1, 4], [1, 0],
            [10, 2], [10, 4], [10, 0]
        ])
        self.kmeans = KMeans(k=2, max_iter=100, tol=1e-4)
    
    def test_initialization(self):
        assert self.kmeans.k == 2, "Expected k to be 2"
        assert self.kmeans.max_iter == 100, "Expected max_iter to be 100"
        assert self.kmeans.tol == 1e-4, "Expected tol to be 1e-4"
    
    def test_fit(self):
        self.kmeans.fit(self.X)
        assert self.kmeans.centroids is not None, "Centroids should not be None after fitting"
        assert self.kmeans.labels is not None, "Labels should not be None after fitting"
    
    def test_predict(self):
        self.kmeans.fit(self.X)
        labels = self.kmeans.predict(self.X)
        assert len(labels) == len(self.X), "Number of predictions should match number of samples"
    
    def test_get_centroids(self):
        self.kmeans.fit(self.X)
        centroids = self.kmeans.get_centroids()
        assert centroids.shape == (2, 2), "Centroids shape should be (2, 2)"
    
    def test_get_error(self):
        self.kmeans.fit(self.X)
        error = self.kmeans.get_error(self.X)
        assert isinstance(error, float), "Error should be a float"
        assert error >= 0, "Error should be non-negative"
    
    def test_invalid_k(self):
        with pytest.raises(ValueError, match="The number of centroids must be a positive integer"):
            KMeans(k=0)
    
    def test_invalid_fit_input(self):
        with pytest.raises(ValueError, match="Input must be a 2D numpy array"):
            self.kmeans.fit(np.array([1, 2, 3]))

    def test_invalid_predict_input(self):
        self.kmeans.fit(self.X)  # Ensure model is trained
        with pytest.raises(ValueError, match="Input data must have the same number of features as training data"):
            self.kmeans.predict(np.array([[1], [2], [3]]))  # Incorrect number of features (should be 2, not 1)

if __name__ == '__main__':
    pytest.main()