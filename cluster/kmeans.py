import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        # Check if input values provided correctly
        if not isinstance(k, int) or k<=0:
            raise ValueError("The number of centroids must be a positive integer.")
        if tol <= 0:
            raise ValueError("Minumum error tolerance must be a positive float.")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("Maximum number of iterations before stopping must be a positive integer.")
        #Initializes attributes for the class
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None
        self.error = None

        
    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        # Check if input values provided correctly
        if not isinstance(mat, np.ndarray) or mat.ndim != 2:
            raise ValueError("Input must be a 2D numpy array.")
        if len(mat) < self.k:
            raise ValueError("Number of observations must be >= k.")
        
        # To make random selection of initial centroids the same in each run
        # This seed number is tied to the cluster generated in main.py. For different data you might need to adjust the seed number for optimal clustering. 
        np.random.seed(10)
        # Initialize k disntinct centroids from input matrix
        indices = np.random.choice(len(mat), self.k, replace=False)
        self.centroids = mat[indices]
        # Optimization of clustering
        for _ in range(self.max_iter):
            # Assign data to the nearest centorid based on Euclidean distance:
            distances = cdist(mat, self.centroids, metric='euclidean')
            self.labels = np.argmin(distances, axis=1)
            # Update centroids
            new_centroids = []
            for i in range(self.k):
                cluster_points = mat[self.labels == i]
                if len(cluster_points) > 0:
                    new_centroids.append(cluster_points.mean(axis=0))
                else: 
                    new_centroids.append(mat[np.random.choice(len(mat))]) # Reinitialize empty cluster
            # Check for convergence 
            new_centroids = np.array(new_centroids)
            if np.all(np.linalg.norm(new_centroids - self.centroids, axis=1) < self.tol):
                break
            self.centroids = new_centroids
            # Set the location of new centroids for the next iteration
            self.centroids = new_centroids
        # Asign final clustering error
        self.error = self.get_error(mat)
        

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        if self.centroids is None:
            raise ValueError("Model not trained. Call fit() first.")
        if mat.shape[1] != self.centroids.shape[1]:
            raise ValueError("Input data must have the same number of features as training data")
        return np.argmin(cdist(mat, self.centroids, metric='euclidean'), axis=1)


    def get_error(self, mat:np.ndarray) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        if self.centroids is None:
            raise ValueError("Model not trained. Call fit() first.")
        mat = mat if mat is not None else self.training_data
        labels = self.predict(mat)
        return sum(np.linalg.norm(mat[i] - self.centroids[labels[i]])**2 for i in range(len(mat)))

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        if self.centroids is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.centroids
        

