print("Testing imports...")

try:
    from langchain_community.document_loaders import PyPDFLoader
    print("✓ PyPDFLoader works!")
except Exception as e:
    print(f"✗ PyPDFLoader failed: {e}")

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    print("✓ RecursiveCharacterTextSplitter works!")
except Exception as e:
    print(f"✗ RecursiveCharacterTextSplitter failed: {e}")

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("✓ HuggingFaceEmbeddings works!")
except Exception as e:
    print(f"✗ HuggingFaceEmbeddings failed: {e}")

try:
    from langchain_community.vectorstores import FAISS
    print("✓ FAISS works!")
except Exception as e:
    print(f"✗ FAISS failed: {e}")

try:
    from langchain.chains import RetrievalQA
    print("✓ RetrievalQA works!")
except Exception as e:
    print(f"✗ RetrievalQA failed: {e}")

try:
    from langchain_community.llms import HuggingFaceHub
    print("✓ HuggingFaceHub works!")
except Exception as e:
    print(f"✗ HuggingFaceHub failed: {e}")

print("\n=== TEST COMPLETE ===")