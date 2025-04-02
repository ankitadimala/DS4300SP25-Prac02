import subprocess
import os
import json
import argparse
import indexing

# path setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMBEDDING_SCRIPT = os.path.join(PROJECT_ROOT, "src", "embedding.py")
EMBEDDING_RESULTS_DIR = os.path.join(PROJECT_ROOT, "embedding_results")
CONFIG_FILE = os.path.join(PROJECT_ROOT, "last_indexed_config.json")

# define CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="all-MiniLM-L6-v2")
parser.add_argument("--chunk_size", type=int, default=200)
parser.add_argument("--overlap", type=int, default=50)
parser.add_argument("--vector_db", choices=["faiss", "chroma", "redis"], default="redis")
args = parser.parse_args()

embedding_output_file = os.path.join(
    EMBEDDING_RESULTS_DIR,
    f"{args.model.replace('/', '_')}__chunk{args.chunk_size}_overlap{args.overlap}.json"
)

def main():
    # preprocess data and generate embeddings
    print("Preprocessing PDFs and generating embeddings...")
    subprocess.run([
        "python", EMBEDDING_SCRIPT,
        "--model", args.model,
        "--chunk_size", str(args.chunk_size),
        "--overlap", str(args.overlap)
    ], check=True)

    # index into appropriate db
    print("Indexing into vector databases...")
    indexing.create_hnsw_index()
    indexing.load_and_store_embeddings(embedding_output_file, vector_db=args.vector_db)

    # write metadata for future query use
    config = {
        "model": args.model,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "vector_dbs": [args.vector_db]
    }

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

    print("Vector databases loaded and ready for querying!")

if __name__ == "__main__":
    main()