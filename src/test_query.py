from query import query_llm
import time

question = "When was Redis originally released?"

# Choose one of: "redis", "chroma", "faiss"
source = "faiss"

# Choose one of your downloaded Ollama models
model = "tinyllama"

print(f"\nQuestion: {question}")
print(f"Using vector DB: {source}")
print(f"Model: {model}\n")

print(f"Asking LLM...")
start_time = time.time()
response = query_llm(question, source=source, model=model)
end_time = time.time()
print("\nGot response!")

print("LLM Response:\n\n", response)
print(f"\n Query completed in {end_time - start_time:.2f} seconds.")
