# utils/hash_utils.py
import hashlib

def hash_file(path, chunk_size=8192):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)
    return sha.hexdigest()
