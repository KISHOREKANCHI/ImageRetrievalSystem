import os
from sentence_transformers import SentenceTransformer
from storage.metadata import MetadataStore
from storage.faiss_index import FaissIndex
from utils.opener import open_image

def run(query, index_path, meta_db, top_k=5):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    metadata = MetadataStore(meta_db)
    index = FaissIndex(dim=384, path=index_path)

    query_vec = embedder.encode([query], convert_to_numpy=True)

    scores, ids = index.search(query_vec, top_k)

    print("\n🔍 Results:\n")

    for score, idx in zip(scores[0], ids[0]):
        path = metadata.get_path(int(idx))
        if path:
            print(f"✅ {path}  (score={score:.3f})")
            open_image(path)
