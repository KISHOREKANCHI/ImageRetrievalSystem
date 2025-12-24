# Phase 1 & 2 (local + colab)

import os
import numpy as np
from tqdm import tqdm

from core.model import CLIPModel
from core.embedder import embed_image
from storage.faiss_index import FaissIndex
from storage.metadata import MetadataStore
from utils.hash_utils import hash_file


def run(image_dir, index_path, meta_db, ingest_mode="append"):
    # ---------- REBUILD SAFETY ----------
    if ingest_mode == "rebuild":
        if os.path.exists(index_path):
            os.remove(index_path)
            print("🗑️ Deleted old FAISS index")

        if os.path.exists(meta_db):
            os.remove(meta_db)
            print("🗑️ Deleted old metadata DB")

    model = CLIPModel()
    index = FaissIndex(dim=512, index_path=index_path)
    metadata = MetadataStore(meta_db)

    images = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"📸 Found {len(images)} images on disk")

    vectors_buffer = []
    meta_buffer = []
    seen_hashes = set()  # 🔑 FIX

    for img_path in tqdm(images, desc="Processing images"):
        img_hash = hash_file(img_path)

        # Skip duplicates (both DB + current batch)
        if img_hash in seen_hashes or metadata.image_exists(img_hash):
            continue

        vector = embed_image(model, img_path)

        if ingest_mode == "rebuild":
            seen_hashes.add(img_hash)
            vectors_buffer.append(vector)
            meta_buffer.append((img_path, img_hash))
        else:
            index.add(vector)
            metadata.add_image(img_path, img_hash)

    if ingest_mode == "rebuild":
        if not vectors_buffer:
            print("⚠️ No new images to ingest")
            return

        vectors_np = np.vstack(vectors_buffer).astype("float32")

        print("🧠 Training FAISS index...")
        index.train(vectors_np)

        print("➕ Adding vectors to FAISS index...")
        index.add(vectors_np)

        print("🗂️ Writing metadata...")
        for img_path, img_hash in meta_buffer:
            metadata.add_image(img_path, img_hash)

    index.save()
    print("✅ Ingestion complete")
