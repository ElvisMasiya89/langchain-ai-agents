import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    print("Retrieving")

    # Initialize OpenAI embeddings and chat model
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    llm = ChatOpenAI()
    
    # Connect to existing Pinecone vector store with embedded documents
    vectorstore = PineconeVectorStore(
        embedding=embeddings,
        index_name=os.environ["INDEX_NAME"]
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
    query = "What are the main benefits mentioned in the document?"
    result = retrieval_chain.invoke({"input": query})
    
    # Display the results
    print(f"Question: {query}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['context'])} documents retrieved")
