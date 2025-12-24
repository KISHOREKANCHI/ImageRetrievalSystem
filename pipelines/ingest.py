# Phase 1 & 2 (local + colab)

import os
from tqdm import tqdm
from core.model import CLIPModel
from core.embedder import embed_image
from storage.faiss_index import FaissIndex
from storage.metadata import MetadataStore


def run(image_dir, index_path, meta_db):
    model = CLIPModel()
    index = FaissIndex(dim=512, index_path=index_path)
    metadata = MetadataStore(meta_db)

    images = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for img_path in tqdm(images, desc="Ingesting images"):
        vector = embed_image(model, img_path)
        index.add(vector)
        metadata.add_image(img_path)

    index.save()
    print("✅ Ingestion complete")

