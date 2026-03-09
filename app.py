import streamlit as st
from rag_engine import RAGEngine
import os
import tempfile

# Page configuration
st.set_page_config(
    page_title="DocuChat - AI Document Assistant",
    page_icon="📄",
    layout="wide",
)

# Initialize session state
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False

# Title
st.title("📄 DocuChat - AI Document Assistant")
st.markdown("###Upload a PDF document and ask questions powered by AI")

# Sidebar
st.sidebar.header("Settings")

# Hugging Face token input
hf_token = st.sidebar.text_input(
    "Hugging Face API Token",
    type="password",
    help="Get free token at huggingface.co"
)

# File upload
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="Select a PDF file to upload and process"
)

# Process document button
if uploaded_file and hf_token:
    if st.button("Process Document", type="primary"):
        with st.spinner("Processing PDF... This may take a minute..."):
            try:
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_pdf_path = tmp_file.name
            
                # Initialize RAG
                st.session_state.rag_engine = RAGEngine()

                # Load and process
                chunks = st.session_state.rag_engine.load_pdf(tmp_path)
                st.session_state.rag_engine.create_vectorstore(chunks)
                st.session_state.rag_engine.setup_qa_chain(hf_token)

                # Clean up temp file
                os.unlink(tmp_path)

                st.session_state.document_processed = True
                st.sidebar.success(f"Processed {len(chunks)} chunks from {uploaded_file.name}!")


            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")

# Main Chat interface
if st.session_state.document_processed and st.session_state.rag_engine:
    st.subheader("Ask Questions About Your Document:")

    # Question input
    question = st.text_input(
        "Your question:",
        placeholder="e.g., What is this document about?",
        key="question_input"
    )

    # Ask button
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("Ask", type="primary")

    if ask_button and question:
        with st.spinner("Thinking..."):
            try:
                # Get answer
                result = st.session_state.rag_engine.ask_question(question)

                # Add to history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "source": result["sources"]
                })
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("Conversation History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                st.markdown(f"**Q{len(st.session_state.chat_history)-i}:** {chat['question']}")
                st.markdown(f"**A:** {chat['answer']}")

                # Show sources
                with st.expander("View Sources"):
                    for j, source in enumerate(chat['source'], 1):
                        st.markdown(f"**Source {j}:** (Page {source.metadata.get('page', 'N/A')}):")
                        st.text(f"{source.page_content[:300]}...")
                
                st.markdown("---")

else:
    # Instructions
    st.info("Upload a PDF and enter your Hugging Face token to get started")

    st.markdown("""
    ### How to use:
    1. Get a **free** Hugging Face token at [huggingface.co](https://huggingface.co) (Sign up -> Settings -> Access Tokens))
    2. Upload any PDF document
    3. Click "Process Document"
    4. Ask questions about the content!
                
    ### Example questions:
    - "What is this document about?"
    - "Summarize the main points"
    - "What are the key findings?"
    - "Who are the main people mentioned?"
    """)

# Footer
st.markdown("---")
st.markdown("**Tech:** LangChain, FAISS, Hugging Face, Sentence Transformers | **Powered by RAG**")

    
                                   