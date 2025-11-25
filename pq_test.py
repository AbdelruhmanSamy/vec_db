import numpy as np
from sklearn.cluster import MiniBatchKMeans
import time
from pq import ProductQuantizer

D = 70
M = 10
K = 5
N = 10

print("Generating random data...")
vectors = np.random.rand(N, D).astype(np.float32)

pq = ProductQuantizer(M=M, K=K, D=D, kmeans_batch_size=2048)

print("Training PQ...")
t0 = time.time()
pq.fit(vectors)
print(f"Training finished in {time.time() - t0:.2f} seconds")

print("Encoding...")
t1 = time.time()
codes = pq.encode(vectors)
print(codes)
print(f"Encoding finished in {time.time() - t1:.2f} seconds")

print("Decoding...")
t2 = time.time()
decoded = pq.decode(codes)
print(f"Decoding finished in {time.time() - t2:.2f} seconds")

# -----------------------------
# Compute reconstruction error
# -----------------------------

# Mean squared error per vector
mse = np.mean((vectors - decoded) ** 2)

# Root mean squared error
rmse = np.sqrt(mse)

# Error percentage relative to original vector magnitude
orig_norm = np.mean(np.linalg.norm(vectors, axis=1))
err_percent = (rmse / orig_norm) * 100

print("\n====================")
print(" PQ ERROR REPORT")
print("====================")
print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"Average vector norm: {orig_norm:.6f}")
print(f"Reconstruction Error (%): {err_percent:.4f}%")
print("====================")
