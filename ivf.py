import numpy as np
from sklearn.cluster import MiniBatchKMeans


class IVF:
    def __init__(self, K, D, batch_size, DB_Size):
        """
        K: number of coarse clusters (inverted lists)
        D: dimension of original vectors
        """
        self.K = K
        self.D = D
        self.FLAT = True
        self.batch_size = batch_size
        self.coarse_centroids = None
        self.inverted_lists = [[] for _ in range(K)]
        self.is_trained = False
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.K,
            random_state=42,
            batch_size=self.batch_size,
            n_init='auto'
        )

    def fit(self, vectors):
        """
        Train coarse quantizer using k-means on the original vectors
        (we don't load all data at once)
        """

        self.kmeans.fit(vectors)
        self.coarse_centroids = self.kmeans.cluster_centers_
        self.is_trained = True
        assert self.kmeans.labels_.ndim == 1

    def predict(self, vectors):
        """
        Returns the cluster ID for a batch of vectors
        """
        return self.kmeans.predict(vectors)
