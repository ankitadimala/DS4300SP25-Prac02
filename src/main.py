import preprocess
import indexing

def main():
    print("Extracting slides...")
    preprocessing_test.process_pdfs("data/")
    print("Indexing slides in Redis, ChromaDB, and FAISS...")
    indexing.create_hnsw_index()
    indexing.process_and_store()
    print("System ready for querying.")

if __name__ == "__main__":
    main()
