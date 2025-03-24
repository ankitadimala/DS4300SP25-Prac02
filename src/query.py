import numpy as np
import ollama
import redis
import faiss
import json
import chromadb
from redis.commands.search.query import Query
from indexing import VECTOR_DIM, get_embedding, faiss_index, faiss_metadata

# --- Redis Setup ---
redis_client = redis.Redis(host="localhost", port=6380, db=0)
INDEX_NAME = "embedding_index"

# --- ChromaDB Setup ---
chroma_client = chromadb.PersistentClient(path="./vector_storage")
chroma_collection = chroma_client.get_or_create_collection("course_notes")


# --- FAISS Setup ---
def load_faiss_vectors_if_needed():
    if faiss_index.ntotal == 0:
        with open("slides_metadata.json", "r", encoding="utf-8") as f:
            slides = json.load(f)
        for slide in slides:
            embedding = get_embedding(slide["text"])
            faiss_index.add(np.array([embedding], dtype=np.float32))
            faiss_metadata.append(slide)


# --- Vector DB Retrieval Methods ---
def retrieve_redis(query_text):
    embedding = get_embedding(query_text)
    q = (
        Query("*=>[KNN 5 @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("text", "module", "score")
        .dialect(2)
    )
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )
    return [doc.text for doc in res.docs]


def retrieve_chroma(query_text):
    embedding = get_embedding(query_text)
    results = chroma_collection.query(query_embeddings=[embedding], n_results=5)
    return results["documents"][0]


def retrieve_faiss(query_text):
    load_faiss_vectors_if_needed()
    embedding = get_embedding(query_text)
    D, I = faiss_index.search(np.array([embedding], dtype=np.float32), 5)
    return [faiss_metadata[i]["text"] for i in I[0]]


# --- LLM Query ---
def query_llm(question, source="redis", model="mistral"):
    if source == "redis":
        contexts = retrieve_redis(question)
    elif source == "chroma":
        contexts = retrieve_chroma(question)
    elif source == "faiss":
        contexts = retrieve_faiss(question)
    else:
        raise ValueError("Invalid source. Choose from: redis, chroma, faiss")

    context = "\n\n".join(contexts)
    messages = [
        {"role": "system", "content": "Answer using the provided course material."},
        {"role": "user", "content": f"{question}\n\nContext:\n{context}"}
    ]
    response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"]