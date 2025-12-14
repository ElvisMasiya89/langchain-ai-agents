# WebCrawl - Parallel Web Scraping with Tavily & Pinecone

Async web scraping system using TavilyCrawl for intelligent web crawling, with batch uploads to Pinecone and rate limiting for parallel processing.

## Features

- **Intelligent Web Crawling**: Uses Tavily's graph-based crawling for comprehensive site exploration
- **Async Processing**: Parallel crawling and processing with asyncio for maximum performance  
- **Rate Limiting**: Respects Pinecone rate limits with exponential backoff retry logic
- **Batch Processing**: Optimized batch uploads to Pinecone (up to 1000 vectors/2MB per request)
- **Clean Architecture**: Modular design with comprehensive documentation

## Environment Variables

Create a `.env` file with:
```
TAVILY_API_KEY=your_tavily_api_key
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
```

## Usage

```python
from backend.crawler import AsyncWebCrawler

# Initialize crawler
crawler = AsyncWebCrawler()

# Crawl and process multiple URLs in parallel
urls = ["https://example.com", "https://another-site.com"]
results = await crawler.crawl_and_store_batch(urls)
```

## Installation

```bash
pip install -r requirements.txt
```