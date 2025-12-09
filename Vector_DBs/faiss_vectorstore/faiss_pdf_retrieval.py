import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    print("Retrieving from PDF using FAISS")

    # Initialize OpenAI embeddings and chat model
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    llm = ChatOpenAI()
    
    # Load the existing FAISS vector store from local directory
    vectorstore = FAISS.load_local(
        "faiss_index", 
        embeddings,
        allow_dangerous_deserialization=True  # Required for loading FAISS index
    )
    
    # Pull a pre-built QA prompt from LangChain Hub
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    
    # Create a chain that combines retrieved documents with the LLM
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    
    # Create the final retrieval chain that retrieves docs and generates answers
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain
    )
    
    # Example query to test the retrieval system
    query = "What is the main topic discussed in the PDF document?"
    result = retrieval_chain.invoke({"input": query})
    
    # Display the results
    print(f"Question: {query}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['context'])} documents retrieved")