import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from .env file
load_dotenv()

def format_docs(docs):
    """Helper function to format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    print("Simple LCEL Retrieval Example")

    # Step 1: Initialize embeddings and language model
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    llm = ChatOpenAI(temperature=0)  # Lower temperature for more consistent answers
    
    # Step 2: Connect to the Pinecone vector store
    vectorstore = PineconeVectorStore(
        embedding=embeddings,
        index_name=os.environ["INDEX_NAME"]
    )
    
    # Step 3: Create a retriever from the vector store
    # This will find the most similar documents to our query
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}  # Return top 3 most similar chunks
    )
    
    # Step 4: Create a custom prompt template
    # This is much simpler than using the hub prompt
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer based on the context, just say you don't know.

Context:
{context}

Question: {question}

Answer:""")
    
    # Step 5: Build the LCEL chain step by step
    # This shows exactly what happens at each step
    
    # The chain works like this:
    # 1. Input: {"question": "user's question"}
    # 2. Retrieve documents using the question
    # 3. Format the documents into readable text
    # 4. Pass both context and question to the prompt
    # 5. Send to LLM and parse the output
    
    rag_chain = (
        {
            "context": retriever | format_docs,  # Get docs and format them
            "question": RunnablePassthrough()     # Pass the question through unchanged
        }
        | prompt           # Create the prompt with context and question
        | llm              # Send to the language model
        | StrOutputParser() # Parse the response to a string
    )
    
    # Step 6: Test the chain with a question
    question = "What are vector databases and how do they differ from traditional databases?"
    
    print(f"\nQuestion: {question}")
    print("\n" + "="*50)
    
    # Invoke the chain - this runs through all the steps above
    answer = rag_chain.invoke(question)
    
    print(f"Answer: {answer}")
    
    print("\n" + "="*50)
    print("\nHow this works:")
    print("1. Your question is used to search the vector database")
    print("2. The most similar document chunks are retrieved")
    print("3. These chunks become the 'context' in our prompt")
    print("4. The LLM reads the context and answers your question")
    print("5. You get a response based on your specific documents!")
    
    # Step 7: Let's also show what documents were retrieved
    print("\n" + "="*50)
    print("\nRetrieved documents preview:")
    retrieved_docs = retriever.invoke(question)
    for i, doc in enumerate(retrieved_docs, 1):
        preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        print(f"\nChunk {i}: {preview}")