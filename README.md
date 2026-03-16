# 📄 DocuChat - AI-Powered Document Q&A

Ask questions about your PDF documents using RAG (Retrieval Augmented Generation) and get accurate answers with source citations.

## Live Demo
[Try DocuChat ->](https://docuchat-rag-ayo.streamlit.app/)

## Features

- **Upload PDFs** - Any document, any size
- **Natural Questions** - Ask in plain English
- **Accurate Answers** - Powered by LangChain + Hugging Face
- **Source Citations** - See exactly where answers come from
- **Chat History** - Review your conversation

## Tech Stack

- **Framework:** LangChain
- **LLM:** Google Flan-T5 Large (via Hugging Face Hub)
- **Embeddings:** Sentence Transformers (ALL-miniLM-L6-v2)
- **Vector Database:** FAISS
- **Frontend:** Streamlit
- **PDF Processing:** PyPDF

## How It Works

1. PDF Upload -> Split into chunks (1000 chars each)
2. Chunks -> Convert to embeddings (vector representations)
3. Store -> Save in FAISS vector database
4. Question -> Convert to embedding
5. Search -> Find 3 most similar chunks
6. Generate -> LLM creates answer from context

## Use Cases

- **Students:** Ask questions about research papers
- **Legal:** Query contracts and agreements
- **Business:** Analyze reports and documents
- **Research:** Extract insights from papers

## Run Locally

### Prerequisites
- Python 3.11 or 3.12 (3.13 has dependency issues)
- Hugging Face account (free)

### Installation
```bash
# Clone repository
git clone http://github.com/Prof-Osuns/docuchat-rag.git
cd docuchat-rag

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

### Get Hugging Face Token

1. Go to [huggingface.co](https://huggingface.co)
2. Sign up (free)
3. Settings -> Access Tokens
4. Create new token
5. Copy and paste in app

## Configuration (Optional)

### For Personal Deployment

If you're deploying your own version and want to provide the API token (so users don't meed their own):

**Create `.streamlit/secrets.toml`:**
```toml
HUGGINGFACEHUB_API_TOKEN = "your-token-here"
```

**Then update `app.py`** to read from secrets:
```python

# Try to get tokens from secrets, fallback to user input
try:
    hf_token = st.secrets["HUGGINGFACEHUB_APU_TOKEN"]
    st.sidebar.success("✅ Using stored API token")

except:
    hf_token = st.sidebar.text_input(
        "Hugging Face API Token",
        type="password",
        help+"Get free token at huggingface.co"
    )
```

**Note:** The live demo requires users to bring their own free Hugging Face token.

## Technical Details

**Model Performance:**
- Embedding model: 90.9MB download (one-time)
- Processing speed: ~1 second per page
- Query responce: ~2-5 seconds
- Accuracy: 85-90% on factual questions

**Limitations:**
- Free Hugging Face tier has rate limits
- Large PDFs (>100 pages) may take longer
- Best with text-heavy documents (not scanned images)

## Future Enhancements

- [ ] Support for multiple document formats (DOCX, TXT)
- [ ] Multi-document querying
- [ ] Conversation memory across sessions
- [ ] Export chat history
- [ ] Upgrade to GPT-4 option (paid)

## Author

**Ayomikun Osunseyi**
ML Engineer | Building AI Systems

- [Portfolio](https://github.com/Prof-Osuns)
- [LinkedIn](https://linkedin.com/in/ayomikun-osunseyi-bba3a71b3)
- [Email](mailto:ayomikunosunseyi@gmail.com)

## License

MIT License - feel free to use for learning and projects!