# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_groq import ChatGroq
# import os
# from utils import initilize_llm_gemini, initilize_llm_groq

# def get_rag_chain():
#     llm = ChatGroq(
#         model="llama-3.3-70b-versatile",
#         temperature=0,
#         groq_api_key="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"
#     )
#     # llm = initilize_llm_gemini
#     # llm = initilize_llm_groq
    
#     prompt = ChatPromptTemplate.from_template(
#         """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
#         Question: {question}
#         Context: {context}
#         Answer:"""
#     )
    
#     return prompt | llm | StrOutputParser()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import initialize_llm_groq

def get_rag_chain():
    llm = initialize_llm_groq()
    
    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {question}
        Context: {context}
        Answer:"""
    )
    
    return prompt | llm | StrOutputParser()