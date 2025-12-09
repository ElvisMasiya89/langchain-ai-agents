import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    print("ingestion")
    
    # Load the text document from file
    loader = TextLoader("medium_blog1.txt")
    document = loader.load()

    print("Splitting")

    # Split the document into smaller chunks for better retrieval
    # chunk_size=1000: Each chunk will be ~1000 characters
    # chunk_overlap=0: No overlap between chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")

    # Initialize OpenAI embeddings model
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    print("Storing embeddings in Pinecone")
    # Create embeddings for all text chunks and store them in Pinecone vector database
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["INDEX_NAME"])

    print("Finish")

