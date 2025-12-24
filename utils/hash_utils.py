# utils/hash_utils.py
"""Utility functions for hashing files.

The project uses file hashes to detect duplicate images during ingestion.
"""

import hashlib

def hash_file(path, chunk_size=8192):
    """Compute and return the SHA-256 hex digest for a file.

    Args:
        path (str): Path to the file to hash.
        chunk_size (int): Number of bytes to read per iteration.

    Returns:
        str: Hexadecimal SHA-256 digest of the file contents.
    """
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)
    return sha.hexdigest()
