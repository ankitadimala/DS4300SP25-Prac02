import time
import tracemalloc
import psutil
import csv
import os
from datetime import datetime
import embedding, indexing, query

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR = os.path.join(PROJECT_ROOT, "experiment_logs")
LOG_FILE = None  # will be set in log_result

def log_result(row: dict):
    global LOG_FILE

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    if not LOG_FILE:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        LOG_FILE = os.path.join(LOG_DIR, f"rag_test_results_{timestamp}.csv")
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)

    print(f"Experiment logged to: {os.path.abspath(LOG_FILE)}")


def run_experiment(
    embed_model: str,
    chunk_size: int,
    overlap: int,
    vector_db: str,
    llm_model: str,
    question: str
):
    print(f"\nRunning experiment with {embed_model}, chunk={chunk_size}, overlap={overlap}, db={vector_db}, llm={llm_model}")
    row = {
        "embed_model": embed_model,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "vector_db": vector_db,
        "llm_model": llm_model,
        "question": question
    }

    # TRACK MEMORY & TIME
    process = psutil.Process(os.getpid())
    tracemalloc.start()
    start_all = time.time()

    # === Step 1: Embedding ===
    t0 = time.time()
    embedding.run_embedding_pipeline(
        selected_models=[embed_model],
        selected_chunk_sizes=[chunk_size],
        selected_overlaps=[overlap]
    )
    embed_time = time.time() - t0
    row["embedding_time_sec"] = round(embed_time, 2)

    # === Step 2: Indexing ===
    filepath = f"embedding_results/{embed_model.replace('/', '_')}__chunk{chunk_size}_overlap{overlap}.json"
    t1 = time.time()
    indexing.load_and_store_embeddings(filepath, vector_db)
    index_time = time.time() - t1
    row["indexing_time_sec"] = round(index_time, 2)

    # === Step 3: Querying ===
    t2 = time.time()
    contexts = query.query_vector_db(
        question,
        embed_model=embed_model,
        vector_db=vector_db,
        top_k=5
    )
    response = f"(LLM output placeholder) Retrieved {len(contexts)} chunks."
    query_time = time.time() - t2
    row["query_time_sec"] = round(query_time, 2)

    # === Final Metrics ===
    current, peak = tracemalloc.get_traced_memory()
    row["peak_memory_mb"] = round(peak / 1024 / 1024, 2)
    tracemalloc.stop()
    row["total_runtime_sec"] = round(time.time() - start_all, 2)
    row["llm_response"] = response.strip().replace("\n", " ")[:500]  # truncate for CSV

    log_result(row)
    print("Logged experiment.\n")

if __name__ == "__main__":
    from itertools import product

    embed_models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
    chunk_sizes = [200, 500]
    overlaps = [0, 50]
    vector_dbs = ["redis", "faiss", "chroma"]
    llm_models = ["mistral", "llama2"]

    question = "What are ACID properties in databases?"

    for (embed_model, chunk_size, overlap, vector_db, llm_model) in product(
        embed_models, chunk_sizes, overlaps, vector_dbs, llm_models
    ):
        run_experiment(
            embed_model=embed_model,
            chunk_size=chunk_size,
            overlap=overlap,
            vector_db=vector_db,
            llm_model=llm_model,
            question=question
        )

