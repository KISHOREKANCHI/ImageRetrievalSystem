# utils/opener.py
"""Small utility to open an image file with the OS default viewer.

`pipelines/search.py` calls `open_image` to preview high-confidence results.
"""

import os
import sys
import subprocess
import time 

def open_image(path):
    """Open `path` with the platform's default image viewer.

    If the file does not exist, prints a warning and returns.
    """
    if not os.path.exists(path):
        print(f"⚠️ File not found: {path}")
        return
   
    os.startfile(path)  # Windows
    time.sleep(0.3)  # Give the viewer time to open