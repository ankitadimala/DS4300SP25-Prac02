from inspect import signature
import chromadb
print("✅ Version:", chromadb.__version__)
print("🧪 Signature:", signature(chromadb.PersistentClient().create_collection))
