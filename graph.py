from typing_extensions import TypedDict
from typing import List
from langgraph.graph import END, StateGraph
from langchain.schema import Document
from router import get_question_router
from retriever import initialize_vectorstore
from generator import get_rag_chain
from grader import get_document_grader, get_hallucination_grader, get_answer_grader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from utils import initialize_vectorstore, load_environment  # Import load_environment
# from grader import get_document_grader, get_hallucination_grader, get_answer_grader

import os


# class GraphState(TypedDict):
#     question: str
#     generation: str
#     web_search: str
#     documents: List[str]
#     hallucination_grade: str
#     answer_grade: str

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[str]
    hallucination_grade: str
    answer_grade: str
    urls: List[str]  # Add URLs to state

# def retrieve(state):
#     print("---RETRIEVE from Vector Store DB---")
#     question = state["question"]
#     retriever = initialize_vectorstore()
#     documents = retriever.invoke(question)
#     return {"documents": documents, "question": question}


def retrieve(state):
    print("---RETRIEVE from Vector Store DB---")
    question = state["question"]
    urls = state.get("urls", None)  # Get URLs from state
    retriever = initialize_vectorstore(urls=urls)
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question, "urls": urls}


def generate(state):
    print("---GENERATE Answer---")
    question = state["question"]
    documents = state["documents"]
    rag_chain = get_rag_chain()
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    retrieval_grader = get_document_grader()
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def web_search(state):
    print("---WEB SEARCH. Append to vector store db---")
    question = state["question"]
    documents = state.get("documents", [])
    print(documents)
    # web_search_tool = TavilySearchResults(max_results=3, tavily_api_key=os.getenv("TAVILY_API_KEY", "your-api-key"), include_answer=True, ) # here is the problem
    web_search_tool=TavilySearch(tavily_api_key=os.getenv("TAVILY_API_KEY", "your-api-key"), include_answer=True)
    docs = web_search_tool.invoke({"query": question})
    print(docs)
    results = docs.get("results", [])
    # if not results:
    #     print("No results found in web search.")
    #     return {"documents": documents, "question": question}
    final_content=""
    for d in results:
        if "content" not in d:
            print("No content found in web search result.")
            continue
        # Create Document object from web search result
        doc = Document(page_content=d["content"], metadata={"source": d.get("url", "unknown")})
        documents.append(doc)
    # web_results = "\n".join([d["content"] for d in docs])

    # web_results = docs.get("content", "no content found")
    # print(web_results)
    # web_results = Document(page_content=web_results)
    # print(web_results)
    # documents.append(web_results)
    return {"documents": documents, "question": question}

def route_question(state):
    print("---ROUTE QUESTION---")
    question = state["question"]
    question_router = get_question_router()
    source = question_router.invoke({"question": question})
    if source.datasource == 'websearch':
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source.datasource == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    web_search = state["web_search"]
    if web_search == "Yes":
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    hallucination_grader = get_hallucination_grader()
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        answer_grader = get_answer_grader()
        score = answer_grader.invoke({"question": question, "generation": generation})
        answer_grade = score.binary_score
        if answer_grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

# def run_workflow(inputs):
#     workflow = StateGraph(GraphState)
#     workflow.add_node("websearch", web_search)
#     workflow.add_node("retrieve", retrieve)
#     workflow.add_node("grade_documents", grade_documents)
#     workflow.add_node("generate", generate)
    
#     workflow.add_edge("websearch", "generate")
#     workflow.add_edge("retrieve", "grade_documents")
    
#     workflow.set_conditional_entry_point(
#         route_question,
#         {"websearch": "websearch", "vectorstore": "retrieve"}
#     )
    
#     workflow.add_conditional_edges(
#         "grade_documents",
#         decide_to_generate,
#         {"websearch": "websearch", "generate": "generate"}
#     )
    
#     workflow.add_conditional_edges(
#         "generate",
#         grade_generation_v_documents_and_question,
#         {"not supported": "generate", "useful": END, "not useful": "websearch"}
#     )
    
#     app = workflow.compile()
    
#     # Stream the workflow and collect final state
#     final_state = {}
#     for output in app.stream(inputs):
#         for key, value in output.items():
#             print(f"Finished running: {key}:")
#             final_state.update(value)
    
#     # Add grading results to the final state
#     if final_state.get("generation"):
#         hallucination_grader = get_hallucination_grader()
#         answer_grader = get_answer_grader()
#         final_state["hallucination_grade"] = hallucination_grader.invoke({
#             "documents": final_state["documents"],
#             "generation": final_state["generation"]
#         }).binary_score
#         final_state["answer_grade"] = answer_grader.invoke({
#             "question": final_state["question"],
#             "generation": final_state["generation"]
#         }).binary_score
    
#     return final_state



def run_workflow(inputs):
    # Ensure URLs are included in inputs
    workflow = StateGraph(GraphState)
    workflow.add_node("websearch", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    
    workflow.add_edge("websearch", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    
    workflow.set_conditional_entry_point(
        route_question,
        {"websearch": "websearch", "vectorstore": "retrieve"}
    )
    
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"websearch": "websearch", "generate": "generate"}
    )
    
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {"not supported": "generate", "useful": END, "not useful": "websearch"}
    )
    
    app = workflow.compile()
    
    # Stream the workflow and collect final state
    final_state = {}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Finished running: {key}:")
            final_state.update(value)
    
    # Add grading results to the final state
    if final_state.get("generation"):
        hallucination_grader = get_hallucination_grader()
        answer_grader = get_answer_grader()
        final_state["hallucination_grade"] = hallucination_grader.invoke({
            "documents": final_state["documents"],
            "generation": final_state["generation"]
        }).binary_score
        final_state["answer_grade"] = answer_grader.invoke({
            "question": final_state["question"],
            "generation": final_state["generation"]
        }).binary_score
    
    return final_state




