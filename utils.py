from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import logging
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Singleton LLM instance
_groq_llm = None
_gemini_llm = None

def load_environment():
    """Load environment variables from .env file and return a dictionary of required keys."""
    try:
        load_dotenv()
        required_keys = ["GROQ_API_KEY", "TAVILY_API_KEY", "GOOGLE_API_KEY"]
        env_vars = {key: os.getenv(key) for key in required_keys}
        
        # Check for missing keys
        missing_keys = [key for key, value in env_vars.items() if value is None]
        if missing_keys:
            logger.error(f"Missing environment variables: {', '.join(missing_keys)}")
            raise ValueError(f"Missing environment variables: {', '.join(missing_keys)}")
        
        logger.info("Environment variables loaded successfully")
        return env_vars
    except Exception as e:
        logger.error(f"Failed to load environment variables: {str(e)}")
        raise

def initialize_llm_groq():
    """Initialize and return a singleton ChatGroq LLM instance."""
    global _groq_llm
    if _groq_llm is None:
        env_vars = load_environment()
        _groq_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=env_vars["GROQ_API_KEY"]
        )
        logger.info("ChatGroq LLM initialized")
    return _groq_llm

def initialize_llm_gemini():
    """Initialize and return a singleton ChatGoogleGenerativeAI LLM instance."""
    global _gemini_llm
    if _gemini_llm is None:
        env_vars = load_environment()
        _gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            api_key=env_vars["GOOGLE_API_KEY"]
        )
        logger.info("ChatGoogleGenerativeAI LLM initialized")
    return _gemini_llm

def load_web_documents(urls=None):
    """Load documents from a list of URLs. Returns a list of documents."""
    default_urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    urls = urls or default_urls
    try:
        logger.info(f"Loading documents from URLs: {urls}")
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        logger.info(f"Loaded {len(docs_list)} documents")
        return docs_list
    except Exception as e:
        logger.error(f"Failed to load documents: {str(e)}")
        raise

def split_documents(documents, chunk_size=250, chunk_overlap=0):
    """Split documents into chunks for vectorstore processing."""
    try:
        logger.info(f"Splitting {len(documents)} documents with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        doc_splits = text_splitter.split_documents(documents)
        logger.info(f"Created {len(doc_splits)} document chunks")
        return doc_splits
    except Exception as e:
        logger.error(f"Failed to split documents: {str(e)}")
        raise

def get_embeddings(model_name="all-MiniLM-L6-v2"):
    """Initialize and return HuggingFace embeddings model."""
    try:
        logger.info(f"Initializing embeddings model: {model_name}")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        logger.info("Embeddings model initialized successfully")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {str(e)}")
        raise

def initialize_vectorstore(urls=None, chunk_size=250, chunk_overlap=0, embedding_model="all-MiniLM-L6-v2"):
    """Initialize Chroma vectorstore with documents from URLs."""
    try:
        logger.info("Initializing vectorstore")
        # Load documents
        documents = load_web_documents(urls)
        
        # Split documents
        doc_splits = split_documents(documents, chunk_size, chunk_overlap)
        
        # Initialize embeddings
        embeddings = get_embeddings(embedding_model)
        
        # Create vectorstore
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="adv-rag-chroma",
            embedding=embeddings,
        )
        logger.info("Vectorstore initialized successfully")
        return vectorstore.as_retriever()
    except Exception as e:
        logger.error(f"Failed to initialize vectorstore: {str(e)}")
        raise
