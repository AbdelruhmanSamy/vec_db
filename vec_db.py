from typing import Dict, List, Annotated, Tuple
import numpy as np
import os
import gc
import gzip
import pickle
from ivf import IVF
from pq import ProductQuantizer

M = 35
DB_SEED_NUMBER = 10
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64
K = 200
BATCH_SIZE = 1024

defult_dict = {"K": 1000, "n_probe": 20}
params = {
    1_000_000: {"K": 1000, "n_probe": 20},
    10_000_000: {"K": 2500, "n_probe": 25},
    20_000_000: {"K": 3000, "n_probe": 30}
}


class VecDB:
    def __init__(
        self,
        centroids_path="centroids.dat",
        database_file_path="saved_db.dat",
        index_file_path="index",
        new_db=True,
        db_size=None,
        mode="ivf_flat",
    ) -> None:
        
        self.db_path = database_file_path
        self.mode = mode
        self.codes = None
        self.index_path = index_file_path
        self.centroids_path = centroids_path
        self.pq = None
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            
            if not os.path.exists(self.db_path):
                self.generate_database(db_size)
                
            else:
                self._build_index()

    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(
            self.db_path, dtype=np.float32, mode="w+", shape=vectors.shape
        )
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(
            self.db_path, dtype=np.float32, mode="r+", shape=full_shape
        )
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(
                self.db_path,
                dtype=np.float32,
                mode="r",
                shape=(1, DIMENSION),
                offset=offset,
            )
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(
            self.db_path, dtype=np.float32, mode="r", shape=(num_records, DIMENSION)
        )
        return vectors

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        curr_params = params.get(self._get_num_records(), defult_dict)
        n_probe = curr_params.get("n_probe")
        return self.retrieve_ivf(query, top_k, n_probe)

    def retrieve_ivf(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5, n_probe=1):
        
        coarse_centroids = self._load_centroids()
        
        dists = np.linalg.norm(coarse_centroids - query, axis=1)
        top_centroid_indices = np.argsort(dists)[:n_probe] 

        candidates = []
        for centroid_idx in top_centroid_indices:
            partition = self._load_partition(int(centroid_idx))
            candidates.append(partition)

        candidates = np.concatenate(candidates)
        
        candidates.sort()
        
        all_data = self.get_all_rows()
        vectors = np.array(all_data[candidates]) 
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
        
        q_norm = np.linalg.norm(query)
        if q_norm > 0: query = query / q_norm
        
        scores = np.dot(vectors, query.T).flatten()
        
        if len(scores) > top_k:
            best_local = np.argpartition(scores, -top_k)[-top_k:]
            best_local = best_local[np.argsort(scores[best_local])[::-1]]
        else:
            best_local = np.argsort(scores)[::-1]
            
        return candidates[best_local].tolist()


    def retrieve_pq(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        scores = []
        num_records = self._get_num_records()
        coarse_centroids = self._load_centroids()
        # here we assume that the row number is the ID of each vector
        scores = np.linalg.norm(coarse_centroids - query, axis=1)
        centroid_idx = np.argmin(scores)
        residual = query - coarse_centroids[centroid_idx]
        encoded_res = self.pq.encode(np.array([residual])).tolist()[0]
        partition = self._load_partition(int(centroid_idx))
        scores = []
        for code, idx in partition:
            score = self._cal_score_pq(encoded_res, code)
            scores.append((score, idx))
        scores = sorted(scores, reverse=True)[:top_k]
        # return [self.get_one_row(s[1]) for s in scores]
        return [s[1] for s in scores]

    def _load_partition(self, idx: int) -> list[Tuple[np.ndarray, int]]:
        partition = None
        with gzip.open(f"{self.index_path}/index_{idx}.dat", "rb") as f:
            partition = pickle.load(f)
        return partition

    def _load_centroids(self) -> np.ndarray:
        coarse_centroids = None
        with gzip.open(f"{self.index_path}/{self.centroids_path}", "rb") as f:
            coarse_centroids = pickle.load(f)

        return coarse_centroids

    def _cal_score_pq(
        self, encoded_query: np.ndarray, encoded_db_code: np.ndarray
    ) -> float:
        """
        Compute approximate similarity between a PQ-encoded query residual and
        a PQ-encoded database code.

        Both encoded_query and encoded_db_code are arrays of centroid indices.
        """
        score = 0.0
        for i, (q_idx, db_idx) in enumerate(zip(encoded_query, encoded_db_code)):
            # Get centroid vectors from PQ codebook
            q_centroid = self.pq.codebooks[i][q_idx]
            db_centroid = self.pq.codebooks[i][db_idx]

            # Cosine similarity per sub-vector
            dot = np.dot(q_centroid, db_centroid)
            norm_q = np.linalg.norm(q_centroid)
            norm_db = np.linalg.norm(db_centroid)
            score += dot / (norm_q * norm_db)
        return score

    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index_ivf(self, K):
        total_records = self._get_num_records()
        sample_size = 50_000
        vectors = self.get_all_rows()
        
        rng = np.random.default_rng(DB_SEED_NUMBER)
        sample_indices = rng.choice(total_records, sample_size, replace=False)
        sample_indices.sort() 
        sample_vectors = np.array(vectors[sample_indices]) 
        ivf = IVF(K, DIMENSION, BATCH_SIZE, self._get_num_records())
        ivf.fit(sample_vectors)

        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)

        with gzip.open(f"{self.index_path}/{self.centroids_path}", "wb") as f:
            pickle.dump(ivf.coarse_centroids, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("Assigning clusters...")
        all_labels = np.zeros(total_records, dtype=np.int32)
        batch_size = 100_000
        
        for start_idx in range(0, total_records, batch_size):
            end_idx = min(start_idx + batch_size, total_records)
            
            # Read batch from memmap
            vecs = np.array(vectors[start_idx:end_idx])
            labels = ivf.predict(vecs)
            
            all_labels[start_idx:end_idx] = labels
            
            if start_idx % 5_000_000 == 0:
                print(f"  Processed {start_idx}...")
                gc.collect()

        # 3. Invert the List (RAM OPTIMIZED)
        # We sort the labels to group IDs together
        print("Grouping IDs...")
        sorted_ids = np.argsort(all_labels) # original Row IDs sorted by cluster
        sorted_labels = all_labels[sorted_ids]
        
        # Find split points where label changes
        diffs = np.where(np.diff(sorted_labels) != 0)[0] + 1
        splits = np.split(sorted_ids, diffs)
        present_clusters = np.unique(sorted_labels)
        
        print("Saving index files...")
        for i, cluster_id in enumerate(present_clusters):
            # Get IDs for this cluster
            ids = splits[i].astype(np.int32)
            
            # Save to index_X.dat
            file_path = os.path.join(self.index_path, f"index_{cluster_id}.dat")
            with gzip.open(file_path, "wb") as f:
                pickle.dump(ids, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        print("Index Build Complete.")

    def _build_index(self):
        curr_params = params.get(self._get_num_records(), defult_dict)
        num_clusters = curr_params.get("K")
        num_subvectors= curr_params.get("M")
        print(f"curr_params: {curr_params}\n")

        if self.mode == "ivf_flat":
            self._build_index_ivf(num_clusters)
        elif self.mode == "ivf_pq":
            self._build_index_pq(num_clusters, num_subvectors)
        else:
            print("Choose an indexing method")

    def _build_index_pq(self, K, M):
        vectors = self.get_all_rows()
        ivf = IVF(K, DIMENSION, BATCH_SIZE, self._get_num_records())
        ivf.fit(vectors)
        inverted_index = ivf.inverted_lists

        for vecs in inverted_index:
            for idx, residual in vecs:
                vectors[idx] = residual
        pq = ProductQuantizer(M, K, DIMENSION, BATCH_SIZE)
        self.pq = pq
        pq.fit(vectors)
        codes = pq.encode(vectors)
        self.codes = codes
        for i, vecs in enumerate(inverted_index):
            for j, tup in enumerate(vecs):
                inverted_index[i][j] = (codes[tup[0]], tup[0])
        
        if not os.path.exists("ivf_flat"):
            os.makedirs("ivf_flat") 

        with gzip.open(f"ivf_flat/{self.centroids_path}", "wb") as f:
            pickle.dump(ivf.coarse_centroids, f, protocol=pickle.HIGHEST_PROTOCOL)

        for i, vecs in enumerate(inverted_index):
            with gzip.open(f"{self.index_path}_{i}.dat", "wb") as f:
                pickle.dump(inverted_index[i], f, protocol=pickle.HIGHEST_PROTOCOL)


# vec_db = VecDB(db_size=100_000, mode="ivf_flat")
# rng = np.random.default_rng(DB_SEED_NUMBER)
# query = rng.random((1, DIMENSION), dtype=np.float32)[0]
# query = rng.random((1, DIMENSION), dtype=np.float32)[0]
# query = rng.random((1, DIMENSION), dtype=np.float32)[0]
# query = rng.random((1, DIMENSION), dtype=np.float32)[0]
# print(query)
# print(vec_db.retrieve(query))
