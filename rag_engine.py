from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient
import os

class RAGEngine:
    def __init__(self):
        """Initialize RAG components"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

      
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        self.vectorstore = None
        self.llm = None

    def load_pdf(self, pdf_path):
        """Load and process PDF"""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        return chunks
    
    def create_vectorstore(self, chunks):
        """Create FAISS vector database"""
        self.vectorstore = FAISS.from_documents(
            chunks,
            self.embeddings
        )
        return self.vectorstore
    
    def setup_qa_chain(self, hf_token):
        """Setup question-answering with LLM"""
        

        self.llm = InferenceClient(token=hf_token)
         
        
        return self.llm
    
    def ask_question(self, question):
        """Ask a question and get answer"""
        if not self.llm or not self.vectorstore:
            raise ValueError("System not initialized. Load documents first.")
        
        # Retrieve relevant documents using similarity search
        docs = self.vectorstore.similarity_search(question, k=3)
       

        # Combine context from documents
        context = "\n\n".join([doc.page_content for doc in docs])

        # Create prompt
        prompt = f"""Based on the following context, answer the question. If the answer is not in the context, say "I cannot find that information in the document."

    Context:
    {context}

    Question: {question}

    Answer:"""
        
        # Get answer from LLM using Inference Client
        try:
            response = self.llm.text_generation(
                prompt,
                model="google/flan-t5-large",
                max_new_tokens=512,
                temperature=0.5
            )
            answer = response
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
        

        return {
            "answer": answer,
            "sources": docs
        }
    
