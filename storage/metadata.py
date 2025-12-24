# SQLite / JSON metadata

import sqlite3

class MetadataStore:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT,
                image_hash TEXT UNIQUE
            )
        """)

    def image_exists(self, image_hash):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT 1 FROM images WHERE image_hash = ?",
            (image_hash,)
        )
        return cur.fetchone() is not None

    def add_image(self, image_path, image_hash):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO images (image_path, image_hash) VALUES (?, ?)",
            (image_path, image_hash)
        )
        self.conn.commit()
        return cur.lastrowid

    def get_images(self, ids):
        cur = self.conn.cursor()
        q = ",".join("?" * len(ids))
        cur.execute(
            f"SELECT id, image_path FROM images WHERE id IN ({q})",
            ids
        )
        return cur.fetchall()

    def get_path(self, faiss_id):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT image_path FROM images WHERE id = ?",
            (int(faiss_id) + 1,)
        )
        row = cur.fetchone()
        return row[0] if row else None
