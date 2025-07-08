import os
import json
import argparse
from sentence_transformers import SentenceTransformer
from preprocessing import process_folder

# embedding helpers
model_cache = {}

def get_model(model_name):
    if model_name not in model_cache:
        model_cache[model_name] = SentenceTransformer(model_name)
    return model_cache[model_name]

def get_embedding(text, model):
    return model.encode(text, show_progress_bar=False).tolist()

# embedding pipeline
def run_embedding_pipeline(selected_models, selected_chunk_sizes, selected_overlaps):
    
    # set input and output directories
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "embedding_results"))
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # preprocess data
    raw_data = process_folder(DATA_DIR)

    # generate embeddings
    for model_name in selected_models:
        model = get_model(model_name)
        for chunk_size in selected_chunk_sizes:
            for overlap in selected_overlaps:
                output_file = os.path.join(
                    OUTPUT_DIR,
                    f"{model_name.replace('/', '_')}__chunk{chunk_size}_overlap{overlap}.json"
                )

                print(f"Embedding with {model_name} | chunk={chunk_size}, overlap={overlap}")
                output = []
                for key, text in raw_data.items():
                    embedding = get_embedding(text, model)
                    output.append({"text": text, "embedding": embedding, "module": key, "slide_number": 1})

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(output, f, indent=2)

                print(f"Saved {len(output)} entries to {output_file}")

def main():
    # define CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--chunk_size", type=int, required=True)
    parser.add_argument("--overlap", type=int, required=True)
    args = parser.parse_args()

    # run pipeline
    run_embedding_pipeline(
        selected_models=[args.model],
        selected_chunk_sizes=[args.chunk_size],
        selected_overlaps=[args.overlap]
    )

if __name__ == "__main__":
    main()
