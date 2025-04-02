import os
import json
import redis
import numpy as np
import chromadb
import faiss

VECTOR_DIM = 768
DOC_PREFIX = "doc:"
INDEX_NAME = "embedding_index"

redis_client = redis.Redis(host="localhost", port=6380, decode_responses=True)
chroma_client = chromadb.PersistentClient(path="./vector_storage")

faiss_index = faiss.IndexFlatL2(VECTOR_DIM)
faiss_vectors = []
faiss_metadata = []

# create chroma collection based on appropriate dimension
def create_chroma_collection(dimension, name="course_notes"):
    try:
        chroma_client.delete_collection(name)
    except Exception:
        pass

    try:
        return chroma_client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
            dimension=dimension
        )
    except TypeError:
        # fallback for chrom versions that do not support dimension argument
        print(f"Chroma doesn't support `dimension=` in this version. Falling back without it.")
        return chroma_client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None
        )

# index redis
def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT module TEXT slide_number TEXT
        embedding VECTOR HNSW 6 
        DIM {VECTOR_DIM} 
        TYPE FLOAT32 
        DISTANCE_METRIC COSINE
        """
    )

# index faiss
def reset_faiss_index(new_dim):
    global faiss_index, faiss_metadata
    print(f"Resetting FAISS index to dimension: {new_dim}")
    faiss_index = faiss.IndexFlatL2(new_dim)
    faiss_metadata = []

# load appropriate db with embeddings
def load_and_store_embeddings(filepath, vector_db="faiss", collection_name=None):
    print(f"Loading embeddings from: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if vector_db == "faiss":
        embedding_dim = len(data[0]["embedding"])
        if embedding_dim != VECTOR_DIM:
            print(f"Warning: FAISS embedding dimension mismatch! Expected {VECTOR_DIM}, got {embedding_dim}")
        reset_faiss_index(embedding_dim)

    elif vector_db == "chroma":
        embedding_dim = len(data[0]["embedding"])
        collection = collection_name or "course_notes"

        print(f"Creating Chroma collection: {collection} with dim={embedding_dim}")
        chroma_collection = create_chroma_collection(embedding_dim, name=collection)

    elif vector_db == "redis":
        redis_client.flushdb()
        try:
            create_hnsw_index()
        except Exception as e:
            print(f"Failed to create Redis index: {e}")
    else:
        raise ValueError(f"Unsupported vector DB: {vector_db}")

    for i, entry in enumerate(data):
        embedding = np.array(entry["embedding"], dtype=np.float32)

        if vector_db == "faiss":
            faiss_index.add(np.array([embedding]))
            faiss_metadata.append(entry)

        elif vector_db == "chroma":
            chroma_collection.add(
                documents=[entry["text"]],
                embeddings=[embedding],
                metadatas=[{
                    "module": entry["module"],
                    "slide_number": entry["slide_number"]
                }],
                ids=[entry.get("id", str(i))]
            )


        elif vector_db == "redis":
            redis_client.hset(
                f"{DOC_PREFIX}{i}",
                mapping={
                    "text": entry["text"],
                    "module": entry["module"],
                    "slide_number": entry["slide_number"],
                    "embedding": embedding.tobytes()
                }
            )