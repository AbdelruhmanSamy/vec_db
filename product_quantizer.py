class ProductQuantizer:
    def __init__(self, M, K, D):
        """
        M: number of subvectors
        K: codebook size per subvector
        D: original dimension
        """
        self.M = M
        self.K = K
        self.D = D
        self.d_sub = D // M  # subvector dimension
        self.codebooks = None  # List of M codebooks, each (K, d_sub)
        self.is_trained = False
    
    def fit(self, vectors):
        """Learn codebooks from training vectors"""
        pass
    
    def encode(self, vectors):
        """Encode vectors to PQ codes"""
        pass
    
    def decode(self, codes):
        """Reconstruct approximate vectors from codes"""
        pass
    
    def compute_asymmetric_distance(self, query, codes):
        """Compute distances between query and PQ-encoded vectors"""
        pass