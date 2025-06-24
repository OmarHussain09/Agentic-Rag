# from typing import Literal
# from langchain_core.pydantic_v1 import BaseModel, Field
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# import os
# from utils import initilize_llm_gemini, initilize_llm_groq

# class RouteQuery(BaseModel):
#     """Route a user query to the most relevant data source."""
#     datasource: Literal["vectorstore", "websearch"] = Field(
#         description="Given a user question, choose to route it to web search or a vectorstore."
#     )

# def get_question_router():
#     llm = ChatGroq(
#         model="llama-3.3-70b-versatile",
#         temperature=0,
#         groq_api_key="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"
#     )
#     # llm = initilize_llm_groq
#     structured_llm_router = llm.with_structured_output(RouteQuery)
    
#     system = """You are an expert at routing a user question to a vectorstore or websearch.
#     The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
#     Use the vectorstore for questions on these topics. For all else, use websearch."""
    
#     route_prompt = ChatPromptTemplate.from_messages([
#         ("system", system),
#         ("human", "{question}")
#     ])
    
#     return route_prompt | structured_llm_router

from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from utils import initialize_llm_groq

class RouteQuery(BaseModel):
    """Route a user query to the most relevant data source."""
    datasource: Literal["vectorstore", "websearch"] = Field(
        description="Given a user question, choose to route it to web search or a vectorstore."
    )

def get_question_router():
    llm = initialize_llm_groq()
    structured_llm_router = llm.with_structured_output(RouteQuery)
    
    system = """You are an expert at routing a user question to a vectorstore or websearch.
    The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
    Use the vectorstore for questions on these topics. For all else, use websearch."""
    
    route_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}")
    ])
    
    return route_prompt | structured_llm_router