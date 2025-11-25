import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import List

class ProductQuantizer:
    def __init__(self, M, K, D, kmeans_batch_size):
        """
        M: number of subvectors
        K: codebook size per subvector
        D: original dimension
        """
        self.M = M
        self.K = K
        self.D = D
        self.batch_size = kmeans_batch_size
        if self.D % self.M != 0:
            raise ValueError("[PQ] D is not divisible by M")
        self.d_sub = D // M  # subvector dimension
        self.codebooks = None  # List of M codebooks, each (K, d_sub)
        self.is_trained = False
        
    def split_vectors(self, vectors: np.ndarray) -> List[np.ndarray]:
        """
        split vectors (N, D) to M sub-vectors
        returns list of M arrays, each array has shape (N, d_sub)
        """
        # print("Vectors Shape: ", vectors.shape[:])
        N = vectors.shape[0]
        reshape = vectors.reshape(N, self.M, self.d_sub)  # reshape (N,D)->(N,M,d_sub)
        # print("Reshaped Vectors: ", reshape)
        return np.split(reshape, self.M, axis=1)

    def fit(self, vectors):
        """Learn codebooks from training vectors"""
        subvectors_list = self.split_vectors(vectors)
        self.codebooks = []
        for i, subvectors in enumerate(subvectors_list):
            subvectors = subvectors.squeeze(axis=1)  # (N, d_sub, 1) -> (N, d_sub)
            kmeans = MiniBatchKMeans(
                n_clusters=self.K, random_state=42, batch_size=self.batch_size
            )
            kmeans.fit(subvectors)
            self.codebooks.append(kmeans.cluster_centers_)
        # print("Codebooks: ", self.codebooks)
        self.is_trained = True
        print(f"[PQ] trained with M={self.M} subvectors and K={self.K} codewords each")

    def encode(self, vectors):
        """Encode vectors to PQ codes"""
        if not self.is_trained or self.codebooks is None:
            raise RuntimeError("[PQ] must be trained before encode")

        subvectors_list = self.split_vectors(vectors)
        N = vectors.shape[0]
        codes = np.empty((N, self.M), dtype=np.int32)

        for i, subvectors in enumerate(subvectors_list):
            subvectors = subvectors.squeeze(axis=1)
            centroids = self.codebooks[i]

            dists = np.sum(
                (subvectors[:, None, :] - centroids[None, :, :]) ** 2, axis=2
            )
            codes[:, i] = np.argmin(dists, axis=1)
        return codes

    def decode(self, codes):
        """
        Reconstruct approximate vectors from codes.
        
        Parameters:
        -----------
        codes : np.ndarray
            Shape (N, M) - N vectors, M subquantizers
            OR (M,) - single vector
            
        Returns:
        --------
        np.ndarray : Shape (N, D) or (D,) - reconstructed vectors
        """
        if codes.ndim == 1:
            codes = codes.reshape(1, -1)
            
        N = codes.shape[0]
        
        approx_vectors = np.zeros((N, self.M, self.d_sub), dtype=np.float32)
        
        for m in range(self.M):
            approx_vectors[:, m , :] = self.codebooks[m][codes[:, m]]
            
        return approx_vectors.reshape(N, -1) if N > 1 else approx_vectors.reshape(-1)

        
    def compute_asymmetric_distance(self, query, codes):
        """
        Compute distances between query and PQ-encoded vectors.
        
        Parameters:
        -----------
        query : np.ndarray, shape (D,)
            Full-precision query vector
        codes : np.ndarray, shape (N, M)
            PQ codes for N database vectors
            
        Returns:
        --------
        np.ndarray : Shape (N,) - distances from query to each vector
        """
        
        if codes.ndim == 1:
            codes = codes.reshape(1, -1)
            
        N = codes.shape[0]
        query_subvectors = query.reshape(self.M, self.d_sub)
            
        distance_table = np.zeros((self.M, self.K), dtype=np.float32)
        
        for m in range(self.M):
            distance_table[m, :] = np.sum(
                (self.codebooks[m] - query_subvectors[m]) ** 2,
                axis=1
            )
            
        distances = np.zeros(N, dtype=np.float32)
        
        # Vectorized distance computation
        distances = np.take_along_axis(
            distance_table[np.newaxis, :, :],  # (1, M, K)
            codes[:, :, np.newaxis],           # (N, M, 1)
            axis=2
        ).squeeze(axis=2)                     # (N, M)

        return np.sqrt(np.sum(distances, axis=1))
