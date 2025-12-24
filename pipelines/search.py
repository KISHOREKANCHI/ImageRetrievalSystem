# Phase 3 (query)

from core.model import CLIPModel
from core.embedder import embed_text
from storage.faiss_index import FaissIndex
from storage.metadata import MetadataStore


def run(query, index_path, meta_db, top_k=5):
    model = CLIPModel()
    index = FaissIndex(dim=512, index_path=index_path)
    metadata = MetadataStore(meta_db)

    query_vector = embed_text(model, query)
    scores, ids = index.search(query_vector, top_k=top_k)

    results = metadata.get_images(ids[0].tolist())

    print("\n🔍 Results:")
    for score, (img_id, path) in zip(scores[0], results):
        print(f"{path}  (score={score:.4f})")

