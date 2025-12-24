# main.py
import argparse
from pipelines.ingest import run as ingest_run
from pipelines.search import run as search_run


IMAGE_DIR = "data/images"
INDEX_PATH = "data/index/faiss.index"
META_DB = "data/metadata/meta.db"


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Image Search System"
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Ingest mode
    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument(
        "--image_dir", default=IMAGE_DIR, help="Directory of images"
    )

    # Search mode
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument(
        "query", type=str, help="Text query"
    )
    search_parser.add_argument(
        "--top_k", type=int, default=5, help="Number of results"
    )

    args = parser.parse_args()

    if args.mode == "ingest":
        ingest_run(
            image_dir=args.image_dir,
            index_path=INDEX_PATH,
            meta_db=META_DB
        )

    elif args.mode == "search":
        search_run(
            query=args.query,
            index_path=INDEX_PATH,
            meta_db=META_DB,
            top_k=args.top_k
        )


if __name__ == "__main__":
    main()
