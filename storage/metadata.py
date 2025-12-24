# SQLite / JSON metadata

"""Simple SQLite-backed metadata store mapping images to IDs and hashes.

The metadata DB stores `image_path` and a unique `image_hash` used to
detect duplicates during ingestion. Note: `get_path` expects a FAISS ID
and offsets by +1 to match the DB autoincrement row id convention used
by the ingestion pipeline.
"""

import sqlite3


class MetadataStore:
    """Wrapper around an SQLite connection that manages image metadata."""
    def __init__(self, db_path):
        # Connect to SQLite and ensure the table exists
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
        """Return True if an image with `image_hash` already exists."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT 1 FROM images WHERE image_hash = ?",
            (image_hash,)
        )
        return cur.fetchone() is not None

    def add_image(self, image_path, image_hash):
        """Insert a new image record and commit; returns the new row id."""
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO images (image_path, image_hash) VALUES (?, ?)",
            (image_path, image_hash)
        )
        self.conn.commit()
        return cur.lastrowid

    def get_images(self, ids):
        """Return a list of (id, image_path) for the provided DB ids."""
        cur = self.conn.cursor()
        q = ",".join("?" * len(ids))
        cur.execute(
            f"SELECT id, image_path FROM images WHERE id IN ({q})",
            ids
        )
        return cur.fetchall()

    def get_path(self, faiss_id):
        """Resolve a FAISS index id to a stored image path.

        The ingestion pipeline stores image records with an autoincrementing
        `id` that is expected to correspond to FAISS positions. This method
        adds +1 to the provided FAISS id to look up the DB row.
        """
        cur = self.conn.cursor()
        cur.execute(
            "SELECT image_path FROM images WHERE id = ?",
            (int(faiss_id) + 1,)
        )
        row = cur.fetchone()
        return row[0] if row else None
