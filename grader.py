# from langchain_core.pydantic_v1 import BaseModel, Field
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# import os
# from utils import initilize_llm_gemini, initilize_llm_groq

# class GradeDocuments(BaseModel):
#     """Binary score for relevance check on retrieved documents."""
#     binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

# class GradeHallucinations(BaseModel):
#     """Binary score for hallucination present in generation answer."""
#     binary_score: str = Field(description="Answer is supported by the facts, 'yes' or 'no'.")

# class GradeAnswer(BaseModel):
#     """Binary score to assess answer addresses question."""
#     binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

# def get_document_grader():
#     llm = ChatGroq(
#         model="llama-3.3-70b-versatile",
#         temperature=0,
#         groq_api_key="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"
#     )
#     # llm = initilize_llm_groq
#     structured_llm_grader_docs = llm.with_structured_output(GradeDocuments)
    
#     system = """You are a grader assessing relevance of a retrieved document to a user question.
#     If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
#     Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
#     grade_prompt = ChatPromptTemplate.from_messages([
#         ("system", system),
#         ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
#     ])
    
#     return grade_prompt | structured_llm_grader_docs

# def get_hallucination_grader():
#     llm = ChatGroq(
#         model="llama-3.3-70b-versatile",
#         temperature=0,
#         groq_api_key="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"
#     )
#     # llm = initilize_llm_groq
#     structured_llm_grader_hallucination = llm.with_structured_output(GradeHallucinations)
    
#     system = """You are a grader assessing whether an LLM generation is supported by a set of retrieved facts.
#     Restrict yourself to give a binary score, either 'yes' or 'no'. If the answer is supported or partially supported by the set of facts, consider it a yes.
#     Don't consider calling external APIs for additional information as consistent with the facts."""
    
#     hallucination_prompt = ChatPromptTemplate.from_messages([
#         ("system", system),
#         ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
#     ])
    
#     return hallucination_prompt | structured_llm_grader_hallucination

# def get_answer_grader():
#     llm = ChatGroq(
#         model="llama-3.3-70b-versatile",
#         temperature=0,
#         groq_api_key="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"
#     )
#     # llm = initilize_llm_groq
#     structured_llm_grader_answer = llm.with_structured_output(GradeAnswer)
    
#     system = """You are a grader assessing whether an answer addresses / resolves a question.
#     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    
#     answer_prompt = ChatPromptTemplate.from_messages([
#         ("system", system),
#         ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
#     ])
    
#     return answer_prompt | structured_llm_grader_answer

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from utils import initialize_llm_groq

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(description="Answer is supported by the facts, 'yes' or 'no'.")

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

def get_document_grader():
    llm = initialize_llm_groq()
    structured_llm_grader_docs = llm.with_structured_output(GradeDocuments)
    
    system = """You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
    
    return grade_prompt | structured_llm_grader_docs

def get_hallucination_grader():
    llm = initialize_llm_groq()
    structured_llm_grader_hallucination = llm.with_structured_output(GradeHallucinations)
    
    system = """You are a grader assessing whether an LLM generation is supported by a set of retrieved facts.
    Restrict yourself to give a binary score, either 'yes' or 'no'. If the answer is supported or partially supported by the set of facts, consider it a yes.
    Don't consider calling external APIs for additional information as consistent with the facts."""
    
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ])
    
    return hallucination_prompt | structured_llm_grader_hallucination

def get_answer_grader():
    llm = initialize_llm_groq()
    structured_llm_grader_answer = llm.with_structured_output(GradeAnswer)
    
    system = """You are a grader assessing whether an answer addresses / resolves a question.
    Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ])
    
    return answer_prompt | structured_llm_grader_answer