"""
Configuration constants for WebCrawl system.

Contains API limits, batch sizes, and retry configurations for optimal
performance with Tavily and Pinecone rate limits.
"""

# Pinecone Configuration
INDEX_NAME = "webcrawl-index"
PINECONE_BATCH_SIZE = 100  # Optimal batch size for upserts
PINECONE_MAX_BATCH_SIZE = 1000  # Maximum vectors per batch
PINECONE_MAX_REQUEST_SIZE_MB = 2  # Maximum request size
PINECONE_WRITE_UNITS_LIMIT = 50000  # Write units per second per namespace

# Tavily Configuration  
TAVILY_MAX_DEPTH = 3  # Maximum crawl depth
TAVILY_DEFAULT_LIMIT = 50  # Default page limit per crawl
TAVILY_RATE_LIMIT = 100  # Requests per minute (adjust based on plan)

# Async Processing Configuration
MAX_CONCURRENT_CRAWLS = 5  # Maximum parallel crawl operations
MAX_CONCURRENT_UPLOADS = 10  # Maximum parallel Pinecone uploads
SEMAPHORE_LIMIT = 20  # Overall concurrency limit

# Retry Configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # Initial backoff time in seconds
MAX_BACKOFF = 60.0  # Maximum backoff time in seconds
BACKOFF_MULTIPLIER = 2  # Exponential backoff multiplier

# Chunk Configuration for Text Processing
CHUNK_SIZE = 1000  # Text chunk size for embeddings
CHUNK_OVERLAP = 200  # Overlap between chunks