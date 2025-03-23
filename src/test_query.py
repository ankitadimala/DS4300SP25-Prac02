from query import query_llm

question = "What are ACID properties in databases?"

# Choose one of: "redis", "chroma", "faiss"
source = "faiss"

# Choose one of your downloaded Ollama models
model = "llama2"  # or "llama2", etc.

print(f"\nQuestion: {question}")
print(f"Using vector DB: {source}")
print(f"Model: {model}\n")

print(f"Asking LLM...")
response = query_llm(question, source=source, model=model)
print("\nGot response!")

print("LLM Response:\n\n", response)
