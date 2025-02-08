# write your silhouette score unit tests here
import pytest
import numpy as np
from cluster import Silhouette

'''
To run python -m pytest -v test/* 
'''

class TestSilhouette:
    def setup_method(self):
        self.X = np.array([
            [1, 2], [2, 3], [3, 4],
            [8, 8], [9, 9], [10, 10]
        ])
        self.y = np.array([0, 0, 0, 1, 1, 1])
        self.silhouette = Silhouette()

    def test_valid_silhouette(self):
        scores = self.silhouette.score(self.X, self.y)
        assert isinstance(scores, np.ndarray), "Scores should be a numpy array"
        assert scores.shape == (6,), "Scores should have the same number of elements as samples"
        assert np.all((scores >= -1) & (scores <= 1)), "Scores should be in range [-1, 1]"

    def test_invalid_X_type(self):
        with pytest.raises(ValueError, match="Input X must be a 2D numpy array"):
            self.silhouette.score([[1, 2], [2, 3], [3, 4]], self.y)

    def test_invalid_y_type(self):
        with pytest.raises(ValueError, match="Input y must be a 1D numpy array"):
            self.silhouette.score(self.X, [0, 0, 0, 1, 1, 1])

    def test_X_y_length_mismatch(self):
        y_mismatched = np.array([0, 0, 1, 1, 1])  # One less label
        with pytest.raises(ValueError, match="X and y must have the same number of observations"):
            self.silhouette.score(self.X, y_mismatched)

    def test_single_cluster_error(self):
        y_single_cluster = np.array([0, 0, 0, 0, 0, 0])
        with pytest.raises(ValueError, match="Silhouette score requires at least 2 clusters"):
            self.silhouette.score(self.X, y_single_cluster)

if __name__ == '__main__':
    pytest.main()