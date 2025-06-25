import streamlit as st
from graph import run_workflow
from utils import initialize_vectorstore
import logging
import validators

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(page_title="AI RAG Workflow", layout="wide")

# Initialize session state for storing results and URLs
if "result" not in st.session_state:
    st.session_state.result = None
if "documents" not in st.session_state:
    st.session_state.documents = []
if "web_search" not in st.session_state:
    st.session_state.web_search = "No"
if "hallucination_grade" not in st.session_state:
    st.session_state.hallucination_grade = None
if "answer_grade" not in st.session_state:
    st.session_state.answer_grade = None
if "urls" not in st.session_state:
    st.session_state.urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
if "source" not in st.session_state:
    st.session_state.source = []

# Initialize vectorstore (run once at startup or when URLs change)
@st.cache_resource
def setup_vectorstore(_urls):
    logger.info(f"Initializing vectorstore with URLs: {_urls}")
    return initialize_vectorstore(urls=_urls)

# Title and description
st.title("Agentic AI Retrieval-Augmented Generation (RAG) Workflow")
st.markdown("Enter URLs and a question to get an answer powered by a vectorstore or web search. View retrieved documents and grading results below.")

# Input form for URLs and question
with st.form(key="input_form"):
    st.subheader("Document URLs")
    url_input = st.text_area(
        "Enter URLs (one per line):",
        value="\n".join(st.session_state.urls),
        placeholder="e.g., https://lilianweng.github.io/posts/2023-06-23-agent/",
        height=100
    )
    question = st.text_input("Ask a question:", placeholder="e.g., What are the types of agent memory?", max_chars=200)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        submit_button = st.form_submit_button("Get Answer")
    with col2:
        clear_button = st.form_submit_button("Clear Results")
    with col3:
        update_urls_button = st.form_submit_button("Update URLs")

# Process URL updates
if update_urls_button:
    try:
        # Split and clean URLs
        new_urls = [url.strip() for url in url_input.split("\n") if url.strip()]
        # Validate URLs
        invalid_urls = [url for url in new_urls if not validators.url(url)]
        if invalid_urls:
            st.error(f"Invalid URLs: {', '.join(invalid_urls)}")
        elif not new_urls:
            st.error("Please provide at least one valid URL.")
        else:
            # Update session state and clear vectorstore cache
            if new_urls != st.session_state.urls:
                st.session_state.urls = new_urls
                setup_vectorstore.clear()  # Clear cache to force reinitialization
                logger.info(f"Updated URLs: {new_urls}")
                st.success("URLs updated successfully. Vectorstore will be reinitialized.")
    except Exception as e:
        logger.error(f"Error processing URLs: {str(e)}")
        st.error(f"Error processing URLs: {str(e)}")

# Clear session state if clear button is clicked
if clear_button:
    logger.info("Clearing session state")
    st.session_state.result = None
    st.session_state.documents = []
    st.session_state.web_search = "No"
    st.session_state.hallucination_grade = None
    st.session_state.answer_grade = None
    st.session_state.source = []

    st.rerun()

# Process the question and display results
if submit_button and question:
    with st.spinner("Processing your question..."):
        try:
            logger.info(f"Processing question: {question}")
            # Ensure vectorstore is initialized with current URLs
            retriever = setup_vectorstore(st.session_state.urls)
            # Run the LangGraph workflow
            result = run_workflow({"question": question})
            print("---WORKFLOW RESULT---", result)
            
            # Store results in session state
            st.session_state.result = result.get("generation", "No answer generated.")
            st.session_state.documents = result.get("documents", [])
            st.session_state.web_search = result.get("web_search", "No")
            st.session_state.hallucination_grade = result.get("hallucination_grade", "Not evaluated")
            st.session_state.answer_grade = result.get("answer_grade", "Not evaluated")
            logger.info("Workflow completed successfully")
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            st.error(f"An error occurred: {str(e)}. Please try again or check your API keys.")

# Display results
if st.session_state.result:
    st.subheader("Answer")
    st.write(st.session_state.result)

    st.subheader("Retrieved Documents")
    if st.session_state.documents:
        for i, doc in enumerate(st.session_state.documents):
            with st.expander(f"Document {i + 1}"):
                st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                st.write(f"**Source**: {doc.metadata.get('source', 'Unknown')}")
                st.session_state.source.append(doc.metadata.get('source', 'Unknown'))
                print("Doc metadata:", doc)
    else:
        st.write("No documents retrieved.")

    st.subheader("Grading Results")
    # st.write(f"**Web Search Used**: {st.session_state.web_search}")
    st.write(f"**Hallucination Check**: {st.session_state.hallucination_grade}")
    st.write(f"**Answer Relevance**: {st.session_state.answer_grade}")

    st.subheader("URLs Used")
    st.write("\n\n".join(set(st.session_state.source)))
