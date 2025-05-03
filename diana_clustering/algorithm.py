import numpy as np
from distance_matrix import DistanceMatrix

class Diana:
    def __init__(self, data):
        '''
        constructor of the class, it takes the main data frame as input
        '''
        self.data = data
        self.n_samples, self.n_features = data.shape

    def fit(self, n_clusters):
        '''
        this method uses the main Divisive Analysis algorithm to do the clustering

        arguments
        ----------
        n_clusters - integer
                     number of clusters we want

        returns
        -------
        cluster_labels - numpy array
                         an array where cluster number of a sample corresponding to
                         the same index is stored
        '''
        similarity_matrix = DistanceMatrix(self.data)  # similarity matrix of the data
        clusters = [list(range(self.n_samples))]      # list of clusters, initially the whole dataset is a single cluster
        while True:
            c_diameters = [np.max(similarity_matrix[cluster][:, cluster]) for cluster in clusters]  # cluster diameters
            max_cluster_dia = np.argmax(c_diameters)  # maximum cluster diameter
            max_difference_index = np.argmax(np.mean(similarity_matrix[clusters[max_cluster_dia]][:, clusters[max_cluster_dia]], axis=1))
            splinters = [clusters[max_cluster_dia][max_difference_index]]  # splinter group
            last_clusters = clusters[max_cluster_dia]
            del last_clusters[max_difference_index]
            while True:
                split = False
                for j in range(len(last_clusters))[::-1]:
                    splinter_distances = similarity_matrix[last_clusters[j], splinters]
                    last_distances = similarity_matrix[last_clusters[j], np.delete(last_clusters, j, axis=0)]
                    if np.mean(splinter_distances) <= np.mean(last_distances):
                        splinters.append(last_clusters[j])
                        del last_clusters[j]
                        split = True
                        break
                if split == False:
                    break
            del clusters[max_cluster_dia]
            clusters.append(splinters)
            clusters.append(last_clusters)
            if len(clusters) == n_clusters:
                break

        cluster_labels = np.zeros(self.n_samples)
        for i in range(len(clusters)):
            cluster_labels[clusters[i]] = i

        return cluster_labels
