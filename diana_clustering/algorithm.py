import numpy as np
from diana_clustering.distance_matrix import DistanceMatrix

class Diana:
    def __init__(self, data=None, n_clusters=3):
        '''
        constructor of the class, it takes the main data frame as input
        '''
        self.data = data
        self.n_clusters = n_clusters
        if data is not None:
            self.n_samples, self.n_features = data.shape

    def fit(self, data=None):
        '''
        this method uses the main Divisive Analysis algorithm to do the clustering

        arguments
        ----------
        data - DataFrame or numpy array (optional)
              The input data if not provided during initialization

        returns
        -------
        cluster_labels - numpy array
                         an array where cluster number of a sample corresponding to
                         the same index is stored
        '''
        if data is not None:
            self.data = data
            self.n_samples, self.n_features = data.shape
        
        if self.data is None:
            raise ValueError("No data provided for clustering")
            
        similarity_matrix = DistanceMatrix(self.data)  # similarity matrix of the data
        clusters = [list(range(self.n_samples))]      # list of clusters, initially the whole dataset is a single cluster
        
        while len(clusters) < self.n_clusters:
            c_diameters = [np.max(similarity_matrix[cluster][:, cluster]) if len(cluster) > 1 else 0 
                          for cluster in clusters]  # cluster diameters
            
            if all(diam == 0 for diam in c_diameters):
                break  # No more divisible clusters
                
            max_cluster_dia = np.argmax(c_diameters)  # maximum cluster diameter
            
            # If cluster has only one point, skip
            if len(clusters[max_cluster_dia]) <= 1:
                break
                
            # Find the most dissimilar point to form the splinter group
            mean_distances = np.mean(similarity_matrix[clusters[max_cluster_dia]][:, clusters[max_cluster_dia]], axis=1)
            max_difference_index = np.argmax(mean_distances)
            
            # Create initial splinter group
            splinters = [clusters[max_cluster_dia][max_difference_index]]
            
            # Make a copy of the cluster to modify
            last_clusters = clusters[max_cluster_dia].copy()
            del last_clusters[max_difference_index]
            
            # Iteratively reassign points
            while True:
                split = False
                for j in range(len(last_clusters))[::-1]:
                    # Calculate distances to splinter group and remaining cluster
                    splinter_distances = np.mean([similarity_matrix[last_clusters[j], splint] for splint in splinters])
                    
                    # Skip if only one point remains
                    if len(last_clusters) <= 1:
                        remaining_distances = float('inf')
                    else:
                        remaining_indices = [last_clusters[k] for k in range(len(last_clusters)) if k != j]
                        remaining_distances = np.mean([similarity_matrix[last_clusters[j], remain] for remain in remaining_indices])
                    
                    # If closer to splinters, move point
                    if splinter_distances <= remaining_distances:
                        splinters.append(last_clusters[j])
                        del last_clusters[j]
                        split = True
                        break
                
                if not split:
                    break
                    
            # Remove original cluster and add the two new ones
            del clusters[max_cluster_dia]
            if splinters:  # Only add if not empty
                clusters.append(splinters)
            if last_clusters:  # Only add if not empty
                clusters.append(last_clusters)
            
            # If we've reached the desired number of clusters, break
            if len(clusters) >= self.n_clusters:
                break

        # Assign cluster labels
        cluster_labels = np.zeros(self.n_samples, dtype=int)
        for i, cluster in enumerate(clusters):
            for idx in cluster:
                cluster_labels[idx] = i

        return cluster_labels
