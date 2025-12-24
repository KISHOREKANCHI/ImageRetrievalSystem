# utils/opener.py
import os
import sys
import subprocess


def open_image(path):
    if not os.path.exists(path):
        print(f"⚠️ File not found: {path}")
        return

    if sys.platform.startswith("win"):
        os.startfile(path)  # Windows
    elif sys.platform.startswith("darwin"):
        subprocess.run(["open", path])  # macOS
    else:
        subprocess.run(["xdg-open", path])  # Linux
