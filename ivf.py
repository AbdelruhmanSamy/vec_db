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
        self.batch_size = batch_size
        self.coarse_centroids = None
        self.inverted_lists = [[] for _ in range(K)]
        self.is_trained = False
        self.assignments = [[] for _ in range(DB_Size)]

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
        self.is_trained = True
        assert kmeans.labels_.ndim == 1
        self.assignments = kmeans.labels_
        for idx, cluster_id in enumerate(self.assignments):
            self.inverted_lists[cluster_id].append(
                (
                    idx,
                    np.array(vectors[idx] - self.coarse_centroids[cluster_id]).tolist(),
                )
            )

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
