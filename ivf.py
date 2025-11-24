import numpy as np
from sklearn.cluster import MiniBatchKMeans


class IVF:
    def __init__(self, K, D, batch_size):
        """
        K: number of coarse clusters (inverted lists)
        D: dimension of original vectors
        """
        self.K = K
        self.D = D
        self.batch_size = batch_size
        self.coarse_centroids = None
        self.inverted_lists = [[] for _ in range(K)]
        self.is_trained = False

    def fit(self, vectors):
        """
        Train coarse quantizer using k-means on the original vectors
        """
        kmeans = MiniBatchKMeans(
            n_clusters=self.K,
            random_state=42,
            batch_size=self.batch_size,
            compute_labels=True,
        )
        kmeans.fit(vectors)
        self.coarse_centroids = kmeans.cluster_centers_
        print(self.coarse_centroids)
        self.is_trained = True
        assert kmeans.labels_.ndim == 1
        assignments = kmeans.labels_
        for idx, cluster_id in enumerate(assignments):
            self.inverted_lists[cluster_id].append(vectors[idx])

    # def assign(self, vectors):
    #     """
    #     Assign new vectors to nearest coarse centroid (no training)
    #     Returns array of cluster ids
    #     """
    #     if not self.is_trained:
    #         raise RuntimeError("[IVF] must be trained before assign")
    #     dists = np.sum(
    #         (vectors[:, None, :] - self.coarse_centroids[None, :, :]) ** 2, axis=2
    #     )
    #     nearest = np.argmin(dists, axis=1)
    #     return nearest


vectors = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    [11, 10, 9, 8, 7, 6, 5, 4, 3, 2],
    [12, 11, 10, 9, 8, 7, 6, 5, 4, 3],
    [20, 18, 16, 14, 12, 10, 8, 6, 4, 2],
    [21, 19, 17, 15, 13, 11, 9, 7, 5, 3],
    [22, 20, 18, 16, 14, 12, 10, 8, 6, 4],
]

ivf = IVF(4, 10, 1024)
ivf.fit(vectors)
# ivf.assign(vectors)
print(ivf.inverted_lists)
print(ivf.coarse_centroids)
