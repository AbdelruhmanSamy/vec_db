import numpy as np
import unittest

# Assuming your class is in a file named pq.py, import it. 
# If it's in the same file, just ensure the class definition is above this code.
from pq import ProductQuantizer 

class TestProductQuantizer(unittest.TestCase):

    def setUp(self):
        """
        Set up a controlled environment with known values.
        We will use:
        D = 4 (Total dimension)
        M = 2 (Subvectors) -> d_sub = 2
        K = 2 (Codebook size)
        """
        self.D = 4
        self.M = 2
        self.K = 2
        self.pq = ProductQuantizer(M=self.M, K=self.K, D=self.D)

        # --- Manually Inject Codebooks ---
        # Subspace 1 (Dimensions 0, 1)
        # Centroid 0: [0, 0], Centroid 1: [10, 10]
        cb1 = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
        
        # Subspace 2 (Dimensions 2, 3)
        # Centroid 0: [100, 100], Centroid 1: [200, 200]
        cb2 = np.array([[100.0, 100.0], [200.0, 200.0]], dtype=np.float32)
        
        self.pq.codebooks = [cb1, cb2]
        self.pq.is_trained = True

    def test_init_validation(self):
        """Test that invalid D/M ratios raise errors."""
        with self.assertRaises(ValueError):
            ProductQuantizer(M=3, K=256, D=10) # 10 is not divisible by 3

    def test_decode_single_vector(self):
        """
        Test decoding a single code vector [1, 0].
        Should pick Centroid 1 from CB1 and Centroid 0 from CB2.
        """
        codes = np.array([1, 0]) # Shape (M,)
        reconstructed = self.pq.decode(codes)
        
        # Expected: [10, 10] + [100, 100] -> [10, 10, 100, 100]
        expected = np.array([10.0, 10.0, 100.0, 100.0], dtype=np.float32)
        
        self.assertTrue(np.allclose(reconstructed, expected), 
                        f"Expected {expected}, got {reconstructed}")

    def test_decode_batch(self):
        """
        Test decoding a batch of vectors.
        """
        # Vector A: [0, 0] -> [0, 0, 100, 100]
        # Vector B: [1, 1] -> [10, 10, 200, 200]
        codes = np.array([[0, 0], [1, 1]])
        
        reconstructed = self.pq.decode(codes)
        
        expected_A = np.array([0., 0., 100., 100.])
        expected_B = np.array([10., 10., 200., 200.])
        
        self.assertTrue(np.allclose(reconstructed[0], expected_A))
        self.assertTrue(np.allclose(reconstructed[1], expected_B))

    def test_compute_asymmetric_distance_logic(self):
        """
        Test the exact math of asymmetric distance.
        Distance(Query, Code) should equal Distance(Query, Decoded_Vector).
        """
        # Query vector
        query = np.array([2.0, 2.0, 105.0, 105.0], dtype=np.float32)
        
        # Code: [0, 0] -> corresponds to vector [0, 0, 100, 100]
        code = np.array([0, 0])
        codes_batch = np.array([code])
        
        # Calculate Manually:
        # Subvector 1: Query [2,2] vs Centroid [0,0] -> d^2 = 2^2 + 2^2 = 8
        # Subvector 2: Query [105,105] vs Centroid [100,100] -> d^2 = 5^2 + 5^2 = 50
        # Total Sq Dist = 58
        # Sqrt(58) ~= 7.61577
        
        dist_calc = self.pq.compute_asymmetric_distance(query, codes_batch)
        expected_dist = np.sqrt(58.0)
        
        self.assertTrue(np.isclose(dist_calc[0], expected_dist),
                        f"Expected {expected_dist}, got {dist_calc[0]}")

    def test_consistency_decode_vs_distance(self):
        """
        The 'Asymmetric Distance' is defined as the Euclidean distance 
        between the Query and the Decoded (Approximated) vector.
        This test ensures your two functions are mathematically consistent 
        with each other using random data.
        """
        # Setup random larger environment
        N, M, K, D = 100, 4, 16, 16
        pq_rand = ProductQuantizer(M, K, D)
        
        # Mock random codebooks
        pq_rand.codebooks = [np.random.rand(K, D//M) for _ in range(M)]
        pq_rand.is_trained = True
        
        # Random Query and Random Codes
        query = np.random.rand(D)
        codes = np.random.randint(0, K, size=(N, M))
        
        # 1. Get distances via your optimized function
        dists_optimized = pq_rand.compute_asymmetric_distance(query, codes)
        
        # 2. Get distances via Brute Force (Decode -> calc Euclidean)
        decoded_vectors = pq_rand.decode(codes) # (N, D)
        
        # Compute standard Euclidean distance manually
        # sqrt(sum((x-y)^2))
        dists_brute_force = np.linalg.norm(decoded_vectors - query, axis=1)
        
        # 3. Assert they are the same
        self.assertTrue(np.allclose(dists_optimized, dists_brute_force, rtol=1e-5),
                        "Optimized distance computation does not match brute force on decoded vectors")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)