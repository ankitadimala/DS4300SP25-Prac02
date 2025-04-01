import argparse
import json
import os
import time
import sys
import tracemalloc
from query import query_llm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_FILE = os.path.join(PROJECT_ROOT, "last_indexed_config.json")

if __name__ == "__main__":
    # Try to read defaults from config file
    default_model = "all-MiniLM-L6-v2"
    default_chunk_size = 200
    default_overlap = 50
    default_source = "redis"

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                default_model = config.get("model", default_model)
                default_chunk_size = config.get("chunk_size", default_chunk_size)
                default_overlap = config.get("overlap", default_overlap)
                vector_dbs = config.get("vector_dbs", [])
                if default_source not in vector_dbs and vector_dbs:
                    default_source = vector_dbs[0]
        except Exception as e:
            print(f"Failed to read config file: {e}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="What are ACID properties in databases?")
    parser.add_argument("--source", default=default_source)
    parser.add_argument("--llm_model", default="mistral")
    parser.add_argument("--system_prompt", default="You are a helpful assistant. Use the provided course material to answer the question.")
    parser.add_argument("--model")
    parser.add_argument("--chunk_size", type=int)
    parser.add_argument("--overlap", type=int)
    args = parser.parse_args()

    print("\n--- Manual RAG Query ---")
    print(f"Question: {args.question}")
    print(f"Vector DB: {args.source}")
    print(f"LLM model: {args.llm_model}")
    print(f"System prompt: {args.system_prompt}\n")

    print("Asking LLM...")
    tracemalloc.start()
    start_time = time.time()

    response = query_llm(
        question=args.question,
        source=args.source,
        model=args.llm_model,
        system_prompt=args.system_prompt,
        embed_model=args.model or default_model,
        chunk_size=args.chunk_size or default_chunk_size,
        overlap=args.overlap or default_overlap
    )

    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\nGot response!")
    print("<LLM_RESPONSE>")
    print(response)
    print("</LLM_RESPONSE>")
    print(f"<QUERY_MEMORY_MB>{peak / 1024 / 1024:.2f}</QUERY_MEMORY_MB>")
    print(f"\nQuery completed in {end_time - start_time:.2f} seconds.")
    sys.stdout.flush()
