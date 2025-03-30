from query import query_llm
import time

question = "What does the A stand for in ACID properties?"

# Choose one of: "redis", "chroma", "faiss"
source = "chroma"

# Choose one of your downloaded Ollama models: mistral, llama2, tinyllama, 
# gemma:2b, phi, dolphin-mixtral
model = "mistral"

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
