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
                image_path TEXT
            )
        """)

    def add_image(self, image_path):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO images (image_path) VALUES (?)",
            (image_path,)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_images(self, ids):
        cursor = self.conn.cursor()
        q = ",".join("?" * len(ids))
        cursor.execute(
            f"SELECT id, image_path FROM images WHERE id IN ({q})",
            ids
        )
        return cursor.fetchall()
