import numpy as np
from scipy.spatial.distance import pdist, squareform

class DistanceMatrix:
    """
    A class to calculate and store distance matrices for clustering algorithms.
    This class is used by the DIANA clustering algorithm.
    """
    def __init__(self, data):
        """
        Initialize the distance matrix from data points.
        
        Parameters:
        -----------
        data : numpy.ndarray or pandas.DataFrame
            The input data points
        """
        # Convert to numpy array if it's not already
        if hasattr(data, 'values'):
            data = data.values
            
        # Calculate the pairwise distances between all points
        distances = pdist(data, metric='euclidean')
        
        # Convert the condensed distance matrix to a square form
        self.matrix = squareform(distances)
        
    def __getitem__(self, indices):
        """
        Access the distance matrix with numpy-like indexing.
        
        Parameters:
        -----------
        indices : int, slice, list, or tuple
            Indices to access the distance matrix
            
        Returns:
        --------
        numpy.ndarray
            Subset of the distance matrix
        """
        return self.matrix[indices]
