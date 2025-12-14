"""
Async web crawler using Tavily with intelligent content extraction.

This module provides comprehensive web crawling capabilities using Tavily's
graph-based crawling technology with async processing for maximum performance.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import time

from tavily import TavilyClient
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from consts import (
    TAVILY_MAX_DEPTH, TAVILY_DEFAULT_LIMIT, MAX_CONCURRENT_CRAWLS,
    CHUNK_SIZE, CHUNK_OVERLAP, MAX_RETRIES, INITIAL_BACKOFF, BACKOFF_MULTIPLIER
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CrawlResult:
    """
    Structured result from a web crawl operation.
    
    Contains the extracted content, metadata, and processing information
    for a single URL or crawl session.
    """
    url: str
    title: str
    content: str
    metadata: Dict[str, Any]
    chunks: List[str]
    embeddings: Optional[List[List[float]]] = None
    crawl_time: Optional[float] = None
    error: Optional[str] = None


class AsyncTavilyCrawler:
    """
    Async web crawler using Tavily's intelligent crawling capabilities.
    
    Features:
    - Graph-based website traversal with parallel path exploration
    - Intelligent content extraction and cleaning
    - Async processing for maximum throughput
    - Built-in retry logic with exponential backoff
    - Respect for robots.txt and rate limits
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the async Tavily crawler.
        
        Args:
            api_key: Tavily API key for authentication
        """
        self.client = TavilyClient(api_key=api_key)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_CRAWLS)
        
    async def crawl_single_url(
        self, 
        url: str, 
        instructions: str = "Extract all useful content from the website",
        max_depth: int = TAVILY_MAX_DEPTH,
        limit: int = TAVILY_DEFAULT_LIMIT
    ) -> CrawlResult:
        """
        Crawl a single URL with Tavily's intelligent extraction.
        
        Process:
        1. Use semaphore to control concurrency
        2. Execute Tavily crawl with retry logic
        3. Extract and clean content from results
        4. Split content into optimally-sized chunks
        5. Generate embeddings for vector storage
        
        Args:
            url: Starting URL for crawling
            instructions: Crawl instructions for content filtering
            max_depth: Maximum crawl depth for link following
            limit: Maximum number of pages to crawl
            
        Returns:
            CrawlResult with extracted content, chunks, and embeddings
            
        Example:
            >>> crawler = AsyncTavilyCrawler(api_key="your_key")
            >>> result = await crawler.crawl_single_url(
            ...     "https://docs.python.org",
            ...     instructions="Find Python documentation and tutorials"
            ... )
            >>> print(f"Found {len(result.chunks)} content chunks")
        """
        start_time = time.time()
        
        async with self.semaphore:
            try:
                # Execute Tavily crawl with retry logic
                crawl_data = await self._crawl_with_retry(
                    url, instructions, max_depth, limit
                )
                
                # Extract content from crawl results
                content, title, metadata = self._extract_content(crawl_data)
                
                # Split content into chunks for vector processing
                chunks = self._create_chunks(content)
                
                # Generate embeddings for chunks
                embeddings = await self._generate_embeddings(chunks)
                
                crawl_time = time.time() - start_time
                
                logger.info(f"Successfully crawled {url} in {crawl_time:.2f}s - {len(chunks)} chunks")
                
                return CrawlResult(
                    url=url,
                    title=title,
                    content=content,
                    metadata=metadata,
                    chunks=chunks,
                    embeddings=embeddings,
                    crawl_time=crawl_time
                )
                
            except Exception as e:
                error_msg = f"Failed to crawl {url}: {str(e)}"
                logger.error(error_msg)
                return CrawlResult(
                    url=url,
                    title="",
                    content="",
                    metadata={},
                    chunks=[],
                    error=error_msg,
                    crawl_time=time.time() - start_time
                )

    async def crawl_multiple_urls(
        self, 
        urls: List[str],
        instructions: str = "Extract all useful content from the websites"
    ) -> List[CrawlResult]:
        """
        Crawl multiple URLs concurrently with optimal performance.
        
        Uses asyncio.gather() to process multiple URLs in parallel while
        respecting rate limits through semaphore control.
        
        Args:
            urls: List of URLs to crawl
            instructions: Common crawl instructions for all URLs
            
        Returns:
            List of CrawlResult objects for each URL
            
        Example:
            >>> urls = ["https://site1.com", "https://site2.com"]
            >>> results = await crawler.crawl_multiple_urls(urls)
            >>> successful = [r for r in results if not r.error]
        """
        logger.info(f"Starting concurrent crawl of {len(urls)} URLs")
        
        # Create crawl tasks for all URLs
        tasks = [
            self.crawl_single_url(url, instructions) 
            for url in urls
        ]
        
        # Execute all crawls concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = CrawlResult(
                    url=urls[i],
                    title="",
                    content="",
                    metadata={},
                    chunks=[],
                    error=f"Crawl failed: {str(result)}"
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        successful_crawls = sum(1 for r in processed_results if not r.error)
        logger.info(f"Completed crawling: {successful_crawls}/{len(urls)} successful")
        
        return processed_results

    async def _crawl_with_retry(
        self, 
        url: str, 
        instructions: str, 
        max_depth: int, 
        limit: int
    ) -> Dict[str, Any]:
        """
        Execute Tavily crawl with exponential backoff retry logic.
        
        Handles rate limiting and temporary failures with intelligent
        retry strategy to ensure reliable crawling.
        """
        backoff = INITIAL_BACKOFF
        
        for attempt in range(MAX_RETRIES):
            try:
                # Run Tavily crawl in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.client.crawl(
                        url=url,
                        max_depth=max_depth,
                        limit=limit,
                        instructions=instructions
                    )
                )
                return result
                
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise e
                
                logger.warning(f"Crawl attempt {attempt + 1} failed for {url}: {e}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * BACKOFF_MULTIPLIER, 60.0)
        
        raise Exception("Max retries exceeded")

    def _extract_content(self, crawl_data: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        """
        Extract and structure content from Tavily crawl response.
        
        Processes the raw crawl data to extract clean text content,
        page title, and relevant metadata for downstream processing.
        """
        if not crawl_data or 'results' not in crawl_data:
            return "", "Unknown", {}
        
        # Combine content from all crawled pages
        all_content = []
        titles = []
        metadata = {
            'total_pages': len(crawl_data.get('results', [])),
            'urls_found': []
        }
        
        for result in crawl_data['results']:
            if 'content' in result and result['content']:
                all_content.append(result['content'])
            
            if 'title' in result and result['title']:
                titles.append(result['title'])
                
            if 'url' in result:
                metadata['urls_found'].append(result['url'])
        
        # Create combined content and select best title
        combined_content = "\n\n".join(all_content)
        main_title = titles[0] if titles else "Crawled Content"
        
        metadata['content_length'] = len(combined_content)
        metadata['pages_with_content'] = len(all_content)
        
        return combined_content, main_title, metadata

    def _create_chunks(self, content: str) -> List[str]:
        """
        Split content into optimal chunks for vector embedding.
        
        Uses RecursiveCharacterTextSplitter to create semantically
        meaningful chunks while respecting token limits.
        """
        if not content.strip():
            return []
        
        # Split content into chunks
        documents = self.text_splitter.create_documents([content])
        chunks = [doc.page_content for doc in documents]
        
        # Filter out very short chunks
        filtered_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
        
        return filtered_chunks

    async def _generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for content chunks.
        
        Uses OpenAI embeddings model to create vector representations
        for each content chunk for vector database storage.
        """
        if not chunks:
            return []
        
        try:
            # Run embedding generation in thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.embeddings.embed_documents(chunks)
            )
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return []