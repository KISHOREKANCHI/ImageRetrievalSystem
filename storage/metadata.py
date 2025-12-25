import sqlite3
import os

class MetadataStore:
    def __init__(self, db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._create()

    def _create(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT UNIQUE,
            caption TEXT
        )
        """)
        self.conn.commit()

    def add(self, image_path, caption):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO images (image_path, caption) VALUES (?, ?)",
            (image_path, caption)
        )
        self.conn.commit()

    def get_path(self, faiss_id):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT image_path FROM images WHERE id = ?",
            (faiss_id + 1,)
        )
        row = cur.fetchone()
        return row[0] if row else None
