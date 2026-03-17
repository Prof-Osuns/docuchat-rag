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
        self.client = None

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
        self.client = InferenceClient(token=hf_token)
        return self.client
    
    def ask_question(self, question):
        """Ask a question and get answer"""
        if not self.client or not self.vectorstore:
            raise ValueError("System not initialized. Load documents first.")
        
        # Retrieve relevant documents using similarity search
        docs = self.vectorstore.similarity_search(question, k=3)
       

        # Combine context from documents
        context = "\n\n".join([doc.page_content for doc in docs])

        # Shorter, clearer prompt
        prompt = f"""Answer this question using only the information below.
   

Information:
{context}

Question: {question}

Answer (be concise and specific):"""
       
        # Get answer using chat completion
        try:
            messages = [{"role": "user", "content": prompt}]
            completion = self.client.chat_completion(
                messages=messages,
                model="meta-llama/Llama-3.2-3B-Instruct", # Free model
                max_tokens=256,
                temperature=0.3
            )
            answer = completion.choices[0].message.content.strip()
        except Exception as e:
            # Simple fallback
            answer = f"I found relevant information but couldn't generate a complete answer. Here's what I found: {context[:300]}..."

        return {
            "answer": answer,
            "sources": docs
        }
    
