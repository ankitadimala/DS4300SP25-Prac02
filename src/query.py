import os
import json
import numpy as np
import ollama
from embedding import get_model, get_embedding
from indexing import redis_client, INDEX_NAME, DOC_PREFIX
from redis.commands.search.query import Query
import faiss
import chromadb

# Optional: for redis ft queries
import redis.commands.search.aggregation as aggregations

# For FAISS metadata tracking
faiss_metadata = []

def reset_faiss_index(dimension):
    global faiss_index, faiss_metadata
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_metadata = []

def query_vector_db(
    question,
    embed_model="all-MiniLM-L6-v2",
    vector_db="faiss",
    chunk_size=200,
    overlap=0,
    top_k=3
):
    # Embed the question
    model = get_model(embed_model)
    query_vec = get_embedding(question, model, embed_model)

    # === FAISS retrieval ===
    if vector_db == "faiss":
        # Load relevant embedding file
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        embedding_file = os.path.join(
            PROJECT_ROOT,
            "embedding_results",
            f"{embed_model.replace('/', '_')}__chunk{chunk_size}_overlap{overlap}.json"
        )

        with open(embedding_file, "r", encoding="utf-8") as f:
            entries = json.load(f)

        # Reset and rebuild FAISS index
        reset_faiss_index(len(entries[0]["embedding"]))

        for entry in entries:
            emb = np.array(entry["embedding"], dtype=np.float32)
            faiss_index.add(np.array([emb]))
            faiss_metadata.append(entry)

        D, I = faiss_index.search(np.array([query_vec], dtype=np.float32), top_k)
        results = [faiss_metadata[i] for i in I[0]]
        return results

    # === Chroma retrieval ===
    elif vector_db == "chroma":
        chroma_client = chromadb.PersistentClient(path="./vector_storage")

        # Load collection based on embed dimension
        embed_dim = len(query_vec)
        collection_name = f"course_notes_{embed_dim}"
        chroma_collection = chroma_client.get_or_create_collection(collection_name)

        query_results = chroma_collection.query(
            query_texts=[question],
            n_results=top_k
        )

        results = []
        for doc, meta in zip(query_results["documents"][0], query_results["metadatas"][0]):
            results.append({
                "text": doc,
                "module": meta.get("module", "N/A"),
                "slide_number": meta.get("slide_number", "N/A")
            })
        return results


    # === Redis retrieval ===
    elif vector_db == "redis":
        redis_query = (
            Query(f"*=>[KNN {top_k} @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("text", "module", "slide_number", "score")
            .dialect(2)
        )
        redis_results = redis_client.ft(INDEX_NAME).search(
            redis_query,
            query_params={"vec": np.array(query_vec, dtype=np.float32).tobytes()}
        )
        results = []
        for doc in redis_results.docs:
            results.append({
                "text": doc.text,
                "module": doc.module,
                "slide_number": doc.slide_number
            })
        return results

    else:
        raise ValueError(f"Unsupported vector DB: {vector_db}")


def query_llm(question, source="faiss", model="mistral", top_k=5, embed_model="all-MiniLM-L6-v2", chunk_size=200, overlap=0):
    contexts = query_vector_db(
        question=question,
        embed_model=embed_model,
        vector_db=source,
        chunk_size=chunk_size,
        overlap=overlap,
        top_k=top_k
    )

    # Assemble context for the prompt
    context = "\n\n".join([chunk["text"] for chunk in contexts])

    # Build messages for the chat model
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided course material to answer the question."},
        {"role": "user", "content": f"{question}\n\nCourse Material:\n{context}"}
    ]

    # Call the local LLM
    response = ollama.chat(model=model, messages=messages)

    return response["message"]["content"]
