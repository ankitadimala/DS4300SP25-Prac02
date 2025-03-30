import subprocess
import os
import indexing

# --- Path setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMBEDDING_SCRIPT = os.path.join(PROJECT_ROOT, "src", "embedding.py")
EMBEDDING_RESULTS_DIR = os.path.join(PROJECT_ROOT, "embedding_results")

# --- Experiment configuration ---
chunk_size = 200
overlap = 50
embed_model = "all-MiniLM-L6-v2"

# --- Construct output filename based on config ---
embedding_output_file = os.path.join(
    EMBEDDING_RESULTS_DIR,
    f"{embed_model.replace('/', '_')}__chunk{chunk_size}_overlap{overlap}.json"
)

def main():
    print("Preprocessing PDFs and generating embeddings...")
    subprocess.run(["python", EMBEDDING_SCRIPT])

    print("Indexing into vector databases...")
    indexing.create_hnsw_index()
    indexing.load_and_store_embeddings(embedding_output_file)

    print("Vector databases loaded and ready for querying!")

if __name__ == "__main__":
    main()
