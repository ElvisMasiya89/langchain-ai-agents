import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    print("PDF Ingestion with FAISS")
    
    # Load the PDF document
    # Replace "sample.pdf" with your actual PDF file path
    loader = PyPDFLoader("the_react_paper.pdf")
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDF")

    print("Splitting document into chunks")

    # Split the document into smaller chunks for better retrieval
    # chunk_size=1000: Each chunk will be ~1000 characters
    # chunk_overlap=200: 200 characters overlap between chunks for context continuity
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} chunks")

    # Initialize OpenAI embeddings model
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    print("Creating FAISS vector store")
    # Create embeddings for all text chunks and store them in FAISS (local vector database)
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # Save the FAISS index locally for reuse
    vectorstore.save_local("faiss_index")
    print("FAISS index saved locally to 'faiss_index' directory")

    print("Finish")