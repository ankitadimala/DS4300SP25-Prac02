# DS4300SP25-Prac02

to manually load the databases, in the terminal type:

    python src/load_dbs.py

to make specifications about chunk size, chunk overlap, embedding model, and database used:

    python src/load_dbs.py --chunk_size [###] --overlap [##] --model ["all-MiniLM-L6-v2", "all-mpnet-base-v2", or "intfloat/e5-base-v2"] --vector_db ["faiss", "chroma", or "redis"]

ex:

    python src/load_dbs.py --chunk_size 500 --overlap 50 --model "all-MiniLM-L6-v2" --vector_db "redis"

without any arguments, it will default to 200/50 chunks with all-MiniLM-L6-v2 in redis

---

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