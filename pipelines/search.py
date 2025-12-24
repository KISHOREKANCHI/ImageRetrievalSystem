"""Search pipeline: embed a text query, run FAISS, and show results.

This module loads the CLIP model and FAISS index, turns the user's text
query into a vector, and resolves nearest neighbors to image paths using
the metadata DB.
"""

from core.model import CLIPModel
from core.embedder import embed_text
from storage.faiss_index import FaissIndex
from storage.metadata import MetadataStore
from utils.opener import open_image


CONFIDENCE_THRESHOLD = 0.25
TOP_K = 5


def run(query, index_path, meta_db):
    """Execute a search for `query` against the FAISS index.

    The function prints results and opens high-confidence images using
    the platform default viewer.
    """
    # Load model and stores
    model = CLIPModel()
    index = FaissIndex(dim=512, index_path=index_path)
    metadata = MetadataStore(meta_db)

    # Embed text query
    query_vector = embed_text(model, query)

    # Search FAISS
    scores, ids = index.search(query_vector, TOP_K)
    scores = scores.flatten()
    ids = ids.flatten()

    # Resolve IDs → image paths
    results = []
    for score, idx in zip(scores, ids):
        path = metadata.get_path(int(idx))
        if path is not None:
            results.append((path, float(score)))

    if not results:
        print("⚠️ No results returned from FAISS")
        return

    print("\n🔍 Results (opening high confidence images):\n")

    for path, score in results:
        if score >= CONFIDENCE_THRESHOLD:
            print(f"✅ {path}  (score={score:.4f})")
            open_image(path)
        else:
            print(f"⚠️ LOW CONFIDENCE {path}  (score={score:.4f})")
