# FAISS index handling

import faiss
import numpy as np
import os

class FaissIndex:
    def __init__(self, dim, index_path, nlist=100):
        self.dim = dim
        self.index_path = index_path

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFPQ(
                quantizer,
                dim,
                nlist,
                8,   # m
                8    # bits
            )
            self.index.nprobe = 10

    def train(self, vectors):
        if not self.index.is_trained:
            self.index.train(vectors)

    def add(self, vectors):
        if not self.index.is_trained:
            raise RuntimeError("FAISS index not trained")
        self.index.add(vectors)

    def search(self, query_vector, top_k):
        return self.index.search(query_vector, top_k)

    def save(self):
        faiss.write_index(self.index, self.index_path)
