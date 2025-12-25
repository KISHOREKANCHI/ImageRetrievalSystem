import argparse
from pipelines.ingest import run as ingest_run
from pipelines.search import run as search_run

IMAGE_DIR = "data/images"
INDEX_PATH = "data/index/faiss.index"
META_DB = "data/metadata/meta.db"

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("ingest")

    s = sub.add_parser("search")
    s.add_argument("query")

    args = parser.parse_args()

    if args.cmd == "ingest":
        ingest_run(IMAGE_DIR, INDEX_PATH, META_DB)

    elif args.cmd == "search":
        search_run(args.query, INDEX_PATH, META_DB)

if __name__ == "__main__":
    main()
