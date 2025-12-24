# main.py
# Entry point and CLI for the Image Retrieval System.
#
# Provides two subcommands:
# - `ingest`: scan a directory, embed images and add them to the FAISS index
# - `search`: embed a text query and return nearest images from FAISS

import argparse
from pipelines.ingest import run as ingest_run
from pipelines.search import run as search_run
import os

# Default locations for data components. These can be overridden via CLI flags.
IMAGE_DIR = os.path.join("data", "images")
INDEX_PATH = os.path.join("data", "index", "faiss.index")
META_DB = os.path.join("data", "metadata", "meta.db")


def main():
    """Parse CLI arguments and dispatch to the appropriate pipeline."""
    parser = argparse.ArgumentParser(
        description="Semantic Image Search System"
    )

    # Use subcommands for ingest and search modes
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Ingest mode: accepts image_dir and ingest_mode
    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument(
        "--image_dir",
        default=IMAGE_DIR,
        help="Directory of images"
    )
    ingest_parser.add_argument(
        "--ingest_mode",
        choices=["rebuild", "append"],
        default="append",
        help="Ingest mode"
    )

    # Search mode: requires a text query and optional top_k
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument(
        "query", type=str, help="Text query"
    )
    search_parser.add_argument(
        "--top_k", type=int, default=5, help="Number of results"
    )

    args = parser.parse_args()

    if args.mode == "ingest":
        # Delegate to the ingest pipeline
        ingest_run(
            image_dir=args.image_dir,
            index_path=INDEX_PATH,
            meta_db=META_DB,
            ingest_mode=args.ingest_mode
        )

    elif args.mode == "search":
        # Delegate to the search pipeline
        search_run(
            query=args.query,
            index_path=INDEX_PATH,
            meta_db=META_DB
        )


if __name__ == "__main__":
    main()
