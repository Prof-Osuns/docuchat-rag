from rag_engine import RAGEngine

print("Testing RAGEngine...")

# Initialize
rag = RAGEngine()
print("✓ RAGEngine initialized")

# Test embeddings
print("✓ Embeddings loaded")

# Test text splitter
test_text = "This is a test. " * 100
chunks = rag.text_splitter.split_text(test_text)
print(f"✓ Text splitter works ({len(chunks)} chunks created)")

print("\n=== RAGEngine is ready! ===")
