# DS4300SP25-Prac02: Local Retrieval-Augmented Generation System 

## Overview 
This project implements a local Retrieval-Augmented Generation (RAG) system for querying course notes collected throughout the semester. The system ingests PDF documents, cleans and chunks the text, generated embeddings using multiple models, indexes these embeddings into vector databases (Redis with RediSearch, Chroma, and FAISS), and finally retrieves context for a user query to generate a response via a locally running LLM (using Ollama).

The system is designed to be highly modular, supporting configurable chunking strategies, multiple embedding models, vector databases, and different loccal LLMs. The goal is to analyze the performance and output quality across various configurations. 

---

## Directory Structure 
DS4300SP25-Prac02/
├── data/                         # Raw course notes (PDFs)
├── embedding_results/            # JSON files with generated embeddings
├── experiment_logs/              # CSV logs from grid experiments
├── llm_outputs/                  # Full LLM responses from grid experiments
├── src/
│   ├── chunking.py               # Token-based chunking of text
│   ├── embedding.py              # Embedding pipeline & model loader
│   ├── indexing.py               # Indexing embeddings into Redis, Chroma, FAISS
│   ├── load_dbs.py               # Driver script: runs the embedding and indexing pipelines
│   ├── preprocessing.py          # PDF extraction, cleaning, and chunking
│   ├── query.py                  # Retrieval and query interface (builds prompts and calls LLM)
│   ├── test_query.py             # Manual testing of the query interface
│   └── test_harness.py           # Grid experiment driver (systematic parameter testing)
├── requirements.txt              # Python dependencies
└── README.md                     # This file

---

## Requirements 
This project requires **Python 3.8+** and the following packages: 

- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [NLTK](https://www.nltk.org/)
- [Sentence Transformers](https://www.sbert.net/)
- [NumPy](https://numpy.org/)
- [Redis](https://redis.io/) (with [RediSearch](https://oss.redislabs.com/redisearch/); use Redis Stack)
- [ChromaDB](https://www.trychroma.com/)
- [FAISS](https://github.com/facebookresearch/faiss) (CPU version recommended: `faiss-cpu`)
- [Ollama](https://ollama.com/) (for local LLM inference)
- [psutil](https://pypi.org/project/psutil/)

A sample `requirements.txt` is provided. Install dependencies with: 
'''bash 
pip install -r requirements.txt 

---

## How to Run the System

**Docker for Redis**
If you do not already have Redis with RediSearch installed locally, you can run it using Docker. 

'''bash
docker pull redis/redis-stack:latest
docker run --name redis-stack -p 6379:6379 -d redis/redis-stack:latest

### Embedding and Indexing 
Manually load the databases, generate embeddings from the preprocessed text, and load them into a vector database. 

**Default Usage**
To run with default parameters (200-token chunks with 50-token overlap, using all-MminiLM-L6-v2 with Redis):

    python src/load_dbs.py

**Custom Parameters**
To make specifications about chunk size, chunk overlap, embedding model, and database used:

    python src/load_dbs.py --chunk_size [###] --overlap [##] --model ["all-MiniLM-L6-v2", "all-mpnet-base-v2", or "intfloat/e5-base-v2"] --vector_db ["faiss", "chroma", or "redis"]

Example:

    python src/load_dbs.py --chunk_size 500 --overlap 50 --model "all-MiniLM-L6-v2" --vector_db "redis"

### Querying the System 

to manually query an llm:

    python src/test_query.py

to make specifications about the question, model and system prompt:

    python src/test_query.py --question ["lorem ipsum"] --llm_model ["lorem ipsum"] -- system prompt ["lorem ipsum"]

ex:

    python src/test_query.py --question ["What is the CAP principle?"] --llm_model ["mistral"] -- system prompt ["You are a database tutor. Answer the queries with information only from the course slides."]

without any arguments, it will default to asking about the ACID properties to mistral with the prompt: "You are a helpful assistant. Use the provided course material to answer the question."

---

to run grid tests:

    python src/test_harness.py

be warned, it takes a looong time to complete

test results are written to experiment logs and llm_outputs