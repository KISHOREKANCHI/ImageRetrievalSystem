import faiss
import os
import numpy as np

class FaissIndex:
    def __init__(self, dim: int, path: str):
        """
        dim  : embedding dimension (384 for MiniLM)
        path : where FAISS index is stored
        """
        self.dim = dim
        self.path = path

        os.makedirs(os.path.dirname(path), exist_ok=True)

        if os.path.exists(path):
            self.index = faiss.read_index(path)
        else:
            # Inner Product + L2-normalized vectors = cosine similarity
            self.index = faiss.IndexFlatIP(dim)

    def add(self, vector: np.ndarray):
        """
        vector: shape (1, dim)
        """
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        faiss.normalize_L2(vector)
        self.index.add(vector)

    def search(self, vector: np.ndarray, top_k: int):
        """
        vector: shape (1, dim)
        returns: scores, indices
        """
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        faiss.normalize_L2(vector)
        return self.index.search(vector, top_k)

    def save(self):
        faiss.write_index(self.index, self.path)
