import os
os.environ["TRANSFORMERS_NO_FLEX_ATTENTION"] = "1"

from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR
import time
import json
from chunking import chunk_by_tokens
from preprocessing import process_folder

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# Config
CHUNK_SIZES = [200, 500, 1000]
OVERLAPS = [0, 50, 100]
EMBED_MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "intfloat/e5-base-v2"
]
OUTPUT_DIR = "embedding_results"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_model(name):
    if "instructor" in name:
        return INSTRUCTOR(name)
    return SentenceTransformer(name)

def get_embedding(text, model, model_name):
    if "instructor" in model_name:
        instruction = "Represent the academic concept for retrieval:"
        return model.encode([[instruction, text]])[0]
    else:
        return model.encode(text)
    
def run_embedding_pipeline(
    selected_models=None,
    selected_chunk_sizes=None,
    selected_overlaps=None
):
    print("Running embedding pipeline...")

    # Load and clean raw PDFs
    raw_data = process_folder(DATA_DIR)

    # Use full config if none provided
    models = selected_models or EMBED_MODELS
    chunk_sizes = selected_chunk_sizes or CHUNK_SIZES
    overlaps = selected_overlaps or OVERLAPS

    for chunk_size in chunk_sizes:
        for overlap in overlaps:
            chunks = []
            for filename, text in raw_data.items():
                module_name = filename.split(".")[0]
                text_chunks = chunk_by_tokens(text, chunk_size, overlap)
                for i, chunk in enumerate(text_chunks):
                    chunks.append({
                        "text": chunk,
                        "module": module_name,
                        "source": filename,
                        "slide_number": i + 1
                    })

            for model_name in models:
                try:
                    model = get_model(model_name)
                    print(f"Embedding with {model_name} | chunk={chunk_size}, overlap={overlap}")
                    start = time.time()

                    embedded = []
                    for entry in chunks:
                        vector = get_embedding(entry["text"], model, model_name)
                        entry["embedding"] = vector.tolist() if hasattr(vector, "tolist") else vector
                        entry["id"] = f"{model_name}_{chunk_size}_{overlap}_{entry['module']}_{entry['slide_number']}"
                        embedded.append(entry)

                    filename = f"{model_name.replace('/', '_')}__chunk{chunk_size}_overlap{overlap}.json"
                    output_path = os.path.join(OUTPUT_DIR, filename)

                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(embedded, f, indent=2)

                    print(f"Saved {len(embedded)} entries to {output_path} in {time.time() - start:.2f}s")

                except Exception as e:
                    print(f"Failed for model {model_name}: {e}")


if __name__ == "__main__":
    run_embedding_pipeline()