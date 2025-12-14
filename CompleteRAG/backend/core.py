"""
RAG (Retrieval-Augmented Generation) system with chat history awareness.

This module implements a complete RAG pipeline that can understand questions
in the context of previous conversation history.
"""

from dotenv import load_dotenv

load_dotenv()

from typing import Any, Dict, List

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from consts import INDEX_NAME


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    """
    Execute RAG pipeline with chat history awareness.
    
    Process:
    1. Setup embeddings model and vector database connection
    2. Load prompt templates for query rephrasing and answer generation
    3. Create history-aware retriever (rephrases vague questions using context)
    4. Combine retrieval and generation into complete RAG chain
    5. Execute pipeline and return answer with source documents
    
    Args:
        query: The user's current question
        chat_history: Previous conversation messages for context
        
    Returns:
        dict: Contains AI's answer and source documents used
        
    Example:
        >>> chat_history = [{"role": "user", "content": "What is LangChain?"}]
        >>> result = run_llm("How do I install it?", chat_history)
        >>> print(result["answer"])  # Will understand "it" refers to LangChain
    """
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)

    """
    Load prompt templates from LangChain Hub:
    
    rephrase_prompt: Template that instructs LLM how to rephrase questions.
    Contains instructions like "Given chat history and follow-up question,
    rephrase to be standalone" with placeholders for {chat_history} and {question}.
    
    Example transformation:
    Input: "How do I install it?" + history about LangChain
    Output: "How do I install LangChain?"
    """
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    """
    Create history-aware retriever:
    
    Flow: query + chat_history → LLM (with rephrase_prompt) → clearer query → 
          vector search → relevant documents
          
    This solves the problem of vague follow-up questions by using conversation
    context to understand what the user is really asking about.
    """
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, 
        retriever=docsearch.as_retriever(), 
        prompt=rephrase_prompt
    )
    
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, 
        combine_docs_chain=stuff_documents_chain
    )

    """
    Execute complete RAG pipeline:
    Input: {"input": current_query, "chat_history": previous_messages}
    Output: {"answer": ai_response, "source_documents": docs_used}
    """
    result = qa.invoke(input={"input": query, "chat_history": chat_history})

    return result
