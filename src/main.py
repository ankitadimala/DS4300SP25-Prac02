import os
import time
import json
import preprocessing, chunking, indexing


def main():
    start_time = time.time()

    print("Preprocessing PDFs...")
    raw_data = preprocessing.process_folder("data")

    print("Chunking text...")
    all_chunks = []
    for filename, text in raw_data.items():
        module_name = os.path.splitext(filename)[0]
        chunks = chunking.chunk_by_tokens(text, chunk_size=200, overlap=50)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "slide_number": i + 1,
                "text": chunk,
                "module": module_name,
                "source": filename
            })

    with open("slides_metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=4)

    print(f"Saved {len(all_chunks)} chunks to slides_metadata.json")

    print("Creating Redis index and storing embeddings...")
    indexing.create_hnsw_index()
    indexing.process_and_store()
    
    print("Indexing complete. Ready for querying.")

    end_time = time.time()
    print(f"\n Pipeline completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()