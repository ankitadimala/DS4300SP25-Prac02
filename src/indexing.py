import redis
import chromadb
import faiss
import ollama
import numpy as np
import json

redis_client = redis.Redis(host="localhost", port=6380, db=0)
chroma_client = chromadb.PersistentClient(path="./vector_storage")
chroma_collection = chroma_client.get_or_create_collection("course_notes")
INDEX_NAME = "embedding_index"
VECTOR_DIM = 768
DOC_PREFIX = "doc:"
faiss_index = faiss.IndexFlatL2(VECTOR_DIM)
faiss_vectors = []
faiss_metadata = []

def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass
    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC COSINE
        """
    )

def get_embedding(text):
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]

def store_embedding(doc_id, text, module):
    embedding = get_embedding(text)
    key = f"{DOC_PREFIX}{doc_id}"

    redis_client.hset(
        key,
        mapping={
            "text": text,
            "module": module,
            "embedding": np.array(embedding, dtype=np.float32).tobytes(),
        },
    )

    chroma_collection.add(
        ids=[doc_id],
        documents=[text],
        embeddings=[embedding],
        metadatas=[{"module": module}],
    )

    faiss_vectors.append(embedding)
    faiss_metadata.append({"doc_id": doc_id, "text": text, "module": module})

def process_and_store():
    with open("slides_metadata.json", "r", encoding="utf-8") as f:
        slides_data = json.load(f)
    for i, slide in enumerate(slides_data):
        store_embedding(str(i), slide["text"], slide["module"])
    faiss_index.add(np.array(faiss_vectors, dtype=np.float32))


