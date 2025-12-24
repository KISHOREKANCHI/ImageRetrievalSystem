"""Thin wrapper around FAISS index creation, training, and search.

This module initializes either a pre-existing index or constructs an
IndexIVFPQ index (with an IndexFlatIP quantizer) for efficient nearest
neighbor search. The wrapper exposes simple `train`, `add`, `search`,
and `save` helpers.
"""
# FAISS index handling
import faiss
import numpy as np
import os


class FaissIndex:
    """Manage a FAISS index stored on-disk.

    Args:
        dim (int): dimensionality of vectors.
        index_path (str): filesystem path to read/write index.
        nlist (int): number of inverted lists for IVFPQ.
    """
    def __init__(self, dim, index_path, nlist=100):
        self.dim = dim
        self.index_path = index_path

        if os.path.exists(index_path):
            # Load existing index from disk
            self.index = faiss.read_index(index_path)
        else:
            # Create a new IVFPQ index using IndexFlatIP as the coarse quantizer
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFPQ(
                quantizer,
                dim,
                nlist,
                8,   # m (subvector count)
                8    # bits per code
            )
            self.index.nprobe = 10

    def train(self, vectors):
        """Train the index on a set of vectors if required.

        `vectors` should be a NumPy array of shape (n, dim) and dtype float32.
        """
        if not self.index.is_trained:
            self.index.train(vectors)

    def add(self, vectors):
        """Add vectors to the index. The index must be trained first."""
        if not self.index.is_trained:
            raise RuntimeError("FAISS index not trained")
        self.index.add(vectors)

    def search(self, query_vector, top_k):
        """Search the index and return (scores, ids).

        `query_vector` should have shape (1, dim).
        """
        return self.index.search(query_vector, top_k)

    def save(self):
        """Persist the FAISS index to disk at `self.index_path`."""
        faiss.write_index(self.index, self.index_path)