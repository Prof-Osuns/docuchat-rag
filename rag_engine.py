from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
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
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

        self.llm = HuggingFaceHub(
            repo_id="google/flan-t5-large",
            task="text2text-generation",
            model_kwargs={"temperature": 0.5, "max_length": 512},
            huggingfacehub_api_token=hf_token
        )

        
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
        
        # Get answer from LLM
        answer = self.llm.invoke(prompt)

        return {
            "answer": answer,
            "sources": docs
        }
    
