# FAISS index handling

import faiss
import numpy as np
import os

class FaissIndex:
    def __init__(self, dim, index_path):
        self.dim = dim
        self.index_path = index_path

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors):
        self.index.add(vectors)

    def search(self, query_vector, top_k=5):
        scores, ids = self.index.search(query_vector, top_k)
        return scores, ids

    def save(self):
        faiss.write_index(self.index, self.index_path)
