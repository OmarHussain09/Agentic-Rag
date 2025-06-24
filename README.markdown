# Agentic AI Retrieval-Augmented Generation (RAG) Workflow

This project implements a Retrieval-Augmented Generation (RAG) workflow using LangChain, LangGraph, and Streamlit. It allows users to input URLs to build a vectorstore, ask questions, and receive answers sourced from either the vectorstore or a web search (via Tavily). The system includes grading mechanisms to assess document relevance, answer grounding, and question resolution, ensuring high-quality responses.

## Features
- **Dynamic URL Input**: Users can provide URLs via the Streamlit UI to populate the vectorstore with custom documents.
- **Question Routing**: Queries are routed to a vectorstore (for topics like agents, prompt engineering, and adversarial attacks) or web search.
- **Document Retrieval**: Uses Chroma vectorstore with HuggingFace embeddings to retrieve relevant documents.
- **Answer Generation**: Generates concise answers (up to three sentences) using a Groq LLM.
- **Grading System**:
  - Checks document relevance to the question.
  - Evaluates if the answer is grounded in retrieved documents (hallucination check).
  - Assesses if the answer resolves the question.
- **Streamlit UI**: Interactive interface to input URLs, ask questions, view answers, retrieved documents, and grading results.

## Project Structure
- **`streamlit_app.py`**: Main entry point, providing the Streamlit UI for user interaction.
- **`graph.py`**: Defines the LangGraph workflow for question routing, retrieval, generation, and grading.
- **`utils.py`**: Utility functions for environment variable loading, vectorstore initialization, and singleton LLM management.
- **`grader.py`**: Implements grading functions for document relevance, hallucinations, and answer quality.
- **`generator.py`**: Defines the RAG chain for answer generation.
- **`router.py`**: Routes questions to vectorstore or web search based on topic.
- **`__init__.py`**: Exports key functions for the package.
- **Redundant File**:
  - `retriever.py`: Duplicates `initialize_vectorstore` from `utils.py` with hardcoded URLs. **Recommendation**: Remove and update `graph.py` to import `initialize_vectorstore` from `utils.py`.

**Note**: Commented-out code in several files (e.g., old LLM initializations) should be cleaned up for clarity.

## Prerequisites
- Python 3.8+
- A `.env` file with the following API keys:
  ```
  GROQ_API_KEY=your_groq_api_key
  TAVILY_API_KEY=your_tavily_api_key
  GOOGLE_API_KEY=your_google_api_key
  ```

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root directory with your API keys.

## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```
2. In the Streamlit UI:
   - Enter URLs (one per line) in the "Document URLs" text area and click "Update URLs" to initialize the vectorstore.
   - Enter a question (e.g., "What are the types of agent memory?") and click "Get Answer".
   - View the answer, retrieved documents, grading results, and used URLs.
   - Click "Clear Results" to reset the session.

## Dependencies
See `requirements.txt` for a full list. Key dependencies include:
- `langchain`, `langchain-core`, `langchain-groq`, `langchain-community`, `langchain-google-genai`
- `sentence-transformers` (for HuggingFace embeddings)
- `chromadb` (vectorstore)
- `tavily-python` (web search)
- `langgraph` (workflow orchestration)
- `streamlit` (UI)
- `validators` (URL validation)
- `python-dotenv` (environment variables)

## Development Notes
- **LLM Initialization**: Uses a singleton pattern in `utils.py` to initialize Groq and Gemini LLMs once, avoiding redundant instantiations.
- **Vectorstore**: Built with Chroma and HuggingFace embeddings (`all-MiniLM-L6-v2`), supporting user-provided URLs.
- **Web Search**: Integrates Tavily for queries outside the vectorstore’s scope.
- **To-Do**:
  - Remove `retriever.py` and update `graph.py` to use `utils.initialize_vectorstore`.
  - Clean up commented-out code for maintainability.
  - Enhance metadata handling to ensure consistent source attribution (addresses "Unknown" source issue).
  - Add unit tests using `pytest`.

## Troubleshooting
- **"Unknown" Source in Documents**: Ensure URLs are valid and accessible. Check logs for metadata issues during document loading (`utils.py`).
- **API Errors**: Verify API keys in `.env` are correct and active.
- **Vectorstore Issues**: Clear the Streamlit cache (`setup_vectorstore.clear()`) if URLs change but documents don’t update.

## License
This project is licensed under the MIT License. See `LICENSE` for details (if applicable).

## Acknowledgments
- Built with [LangChain](https://www.langchain.com/), [LangGraph](https://github.com/langchain-ai/langgraph), and [Streamlit](https://streamlit.io/).
- Inspired by RAG workflows for enhanced question-answering with source attribution.