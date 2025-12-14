"""
Async orchestrator for complete web crawling and vector storage pipeline.

This module coordinates the entire workflow from URLs to indexed vectors,
with comprehensive parallel processing and intelligent resource management.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import os
from dotenv import load_dotenv

from .crawler import AsyncTavilyCrawler, CrawlResult
from .uploader import AsyncPineconeUploader, UploadStats
from consts import (
    MAX_CONCURRENT_CRAWLS, MAX_CONCURRENT_UPLOADS, SEMAPHORE_LIMIT,
    TAVILY_DEFAULT_LIMIT, TAVILY_MAX_DEPTH
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """
    Comprehensive statistics for the entire crawling and upload pipeline.
    
    Provides detailed metrics for monitoring performance, identifying
    bottlenecks, and optimizing resource utilization.
    """
    urls_processed: int = 0
    successful_crawls: int = 0
    failed_crawls: int = 0
    total_chunks_created: int = 0
    total_vectors_uploaded: int = 0
    total_pipeline_time: float = 0.0
    crawl_time: float = 0.0
    upload_time: float = 0.0
    upload_stats: Optional[UploadStats] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class AsyncWebCrawlOrchestrator:
    """
    High-level orchestrator for parallel web crawling and vector storage.
    
    Coordinates the complete pipeline:
    1. Parallel web crawling with Tavily
    2. Content processing and chunking
    3. Embedding generation
    4. Batch upload to Pinecone with rate limiting
    5. Comprehensive monitoring and error handling
    
    Features:
    - Full async processing with optimal concurrency
    - Resource management with semaphores
    - Comprehensive error handling and recovery
    - Detailed performance monitoring
    - Environment-based configuration
    """
    
    def __init__(
        self,
        tavily_api_key: Optional[str] = None,
        pinecone_api_key: Optional[str] = None,
        index_name: Optional[str] = None
    ):
        """
        Initialize the web crawl orchestrator.
        
        Args:
            tavily_api_key: Tavily API key (defaults to TAVILY_API_KEY env var)
            pinecone_api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            index_name: Pinecone index name (defaults to PINECONE_INDEX_NAME env var)
        """
        self.tavily_api_key = tavily_api_key or os.getenv('TAVILY_API_KEY')
        self.pinecone_api_key = pinecone_api_key or os.getenv('PINECONE_API_KEY')
        self.index_name = index_name or os.getenv('PINECONE_INDEX_NAME', 'webcrawl-index')
        
        if not self.tavily_api_key:
            raise ValueError("Tavily API key required (TAVILY_API_KEY env var)")
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key required (PINECONE_API_KEY env var)")
        
        # Initialize components
        self.crawler = AsyncTavilyCrawler(self.tavily_api_key)
        self.uploader = AsyncPineconeUploader(self.pinecone_api_key, self.index_name)
        
        # Overall concurrency control
        self.pipeline_semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)

    async def crawl_and_store_urls(
        self,
        urls: List[str],
        crawl_instructions: str = "Extract all useful content and information",
        max_depth: int = TAVILY_MAX_DEPTH,
        limit_per_url: int = TAVILY_DEFAULT_LIMIT,
        initialize_index: bool = True
    ) -> PipelineStats:
        """
        Execute complete pipeline: crawl URLs and store in Pinecone.
        
        Full workflow:
        1. Initialize Pinecone index if needed
        2. Execute parallel crawling of all URLs
        3. Process and validate crawl results
        4. Generate embeddings for all content chunks
        5. Batch upload to Pinecone with rate limiting
        6. Return comprehensive pipeline statistics
        
        Args:
            urls: List of URLs to crawl and index
            crawl_instructions: Instructions for Tavily crawling
            max_depth: Maximum crawl depth per URL
            limit_per_url: Maximum pages to crawl per URL
            initialize_index: Whether to initialize Pinecone index
            
        Returns:
            PipelineStats with comprehensive metrics
            
        Example:
            >>> orchestrator = AsyncWebCrawlOrchestrator()
            >>> urls = ["https://docs.python.org", "https://fastapi.tiangolo.com"]
            >>> stats = await orchestrator.crawl_and_store_urls(urls)
            >>> print(f"Processed {stats.urls_processed} URLs, "
            ...       f"uploaded {stats.total_vectors_uploaded} vectors")
        """
        start_time = time.time()
        stats = PipelineStats(urls_processed=len(urls))
        
        logger.info(f"Starting pipeline for {len(urls)} URLs")
        
        try:
            # Step 1: Initialize Pinecone index
            if initialize_index:
                logger.info("Initializing Pinecone index...")
                await self.uploader.initialize_index()
            
            # Step 2: Execute parallel crawling
            logger.info("Starting parallel web crawling...")
            crawl_start = time.time()
            
            crawl_results = await self._execute_parallel_crawls(
                urls, crawl_instructions, max_depth, limit_per_url
            )
            
            stats.crawl_time = time.time() - crawl_start
            stats.successful_crawls = sum(1 for r in crawl_results if not r.error)
            stats.failed_crawls = len(crawl_results) - stats.successful_crawls
            
            # Collect crawl errors
            for result in crawl_results:
                if result.error:
                    stats.errors.append(f"Crawl failed for {result.url}: {result.error}")
            
            logger.info(
                f"Crawling completed: {stats.successful_crawls}/{len(urls)} successful "
                f"in {stats.crawl_time:.2f}s"
            )
            
            # Step 3: Process successful crawls
            successful_results = [r for r in crawl_results if not r.error and r.chunks]
            
            if not successful_results:
                logger.warning("No successful crawls with content found")
                stats.total_pipeline_time = time.time() - start_time
                return stats
            
            stats.total_chunks_created = sum(len(r.chunks) for r in successful_results)
            
            # Step 4: Upload to Pinecone
            logger.info(f"Uploading {stats.total_chunks_created} chunks to Pinecone...")
            upload_start = time.time()
            
            upload_stats = await self.uploader.upload_crawl_results(successful_results)
            
            stats.upload_time = time.time() - upload_start
            stats.upload_stats = upload_stats
            stats.total_vectors_uploaded = upload_stats.successful_uploads
            
            # Step 5: Final statistics
            stats.total_pipeline_time = time.time() - start_time
            
            logger.info(
                f"Pipeline completed successfully:\n"
                f"  • URLs processed: {stats.urls_processed}\n"
                f"  • Successful crawls: {stats.successful_crawls}\n"
                f"  • Content chunks: {stats.total_chunks_created}\n"
                f"  • Vectors uploaded: {stats.total_vectors_uploaded}\n"
                f"  • Total time: {stats.total_pipeline_time:.2f}s\n"
                f"  • Crawl time: {stats.crawl_time:.2f}s\n"
                f"  • Upload time: {stats.upload_time:.2f}s"
            )
            
            return stats
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            logger.error(error_msg)
            stats.errors.append(error_msg)
            stats.total_pipeline_time = time.time() - start_time
            return stats

    async def crawl_single_domain(
        self,
        base_url: str,
        crawl_instructions: str = "Extract all useful content and documentation",
        max_depth: int = 3,
        limit: int = 100
    ) -> PipelineStats:
        """
        Crawl a single domain comprehensively.
        
        Optimized for crawling entire documentation sites or knowledge bases
        with intelligent content extraction and deduplication.
        
        Args:
            base_url: Starting URL for domain crawling
            crawl_instructions: Specific instructions for content extraction
            max_depth: Maximum crawl depth
            limit: Maximum pages to crawl
            
        Returns:
            PipelineStats for the domain crawl
        """
        logger.info(f"Starting comprehensive domain crawl: {base_url}")
        
        # Execute single-URL pipeline with higher limits
        return await self.crawl_and_store_urls(
            urls=[base_url],
            crawl_instructions=crawl_instructions,
            max_depth=max_depth,
            limit_per_url=limit
        )

    async def _execute_parallel_crawls(
        self,
        urls: List[str],
        instructions: str,
        max_depth: int,
        limit_per_url: int
    ) -> List[CrawlResult]:
        """
        Execute crawls for all URLs with optimal parallelization.
        
        Manages concurrency to balance throughput with resource constraints
        and API rate limits.
        """
        async with self.pipeline_semaphore:
            # Create customized crawl tasks for each URL
            tasks = []
            for url in urls:
                task = self.crawler.crawl_single_url(
                    url=url,
                    instructions=instructions,
                    max_depth=max_depth,
                    limit=limit_per_url
                )
                tasks.append(task)
            
            # Execute all crawls concurrently
            logger.info(f"Executing {len(tasks)} parallel crawls...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Create error result for failed crawl
                    error_result = CrawlResult(
                        url=urls[i],
                        title="",
                        content="",
                        metadata={},
                        chunks=[],
                        error=f"Crawl exception: {str(result)}"
                    )
                    processed_results.append(error_result)
                else:
                    processed_results.append(result)
            
            return processed_results

    def get_pipeline_summary(self, stats: PipelineStats) -> Dict[str, Any]:
        """
        Generate comprehensive pipeline summary for reporting.
        
        Creates structured summary suitable for logging, monitoring,
        or user interfaces.
        
        Args:
            stats: Pipeline statistics to summarize
            
        Returns:
            Dictionary with formatted pipeline summary
        """
        summary = {
            'pipeline_overview': {
                'total_urls': stats.urls_processed,
                'successful_crawls': stats.successful_crawls,
                'failed_crawls': stats.failed_crawls,
                'success_rate': f"{(stats.successful_crawls / max(stats.urls_processed, 1) * 100):.1f}%"
            },
            'content_processing': {
                'total_chunks_created': stats.total_chunks_created,
                'vectors_uploaded': stats.total_vectors_uploaded,
                'upload_success_rate': f"{(stats.total_vectors_uploaded / max(stats.total_chunks_created, 1) * 100):.1f}%"
            },
            'performance_metrics': {
                'total_time': f"{stats.total_pipeline_time:.2f}s",
                'crawl_time': f"{stats.crawl_time:.2f}s",
                'upload_time': f"{stats.upload_time:.2f}s",
                'throughput': f"{stats.total_vectors_uploaded / max(stats.total_pipeline_time, 1):.1f} vectors/sec"
            }
        }
        
        # Add upload details if available
        if stats.upload_stats:
            summary['upload_details'] = {
                'batches_processed': stats.upload_stats.batches_processed,
                'rate_limit_hits': stats.upload_stats.rate_limit_hits,
                'retries_performed': stats.upload_stats.retries_performed,
                'failed_uploads': stats.upload_stats.failed_uploads
            }
        
        # Add errors if any
        if stats.errors:
            summary['errors'] = stats.errors[:5]  # Limit to first 5 errors
            if len(stats.errors) > 5:
                summary['errors'].append(f"... and {len(stats.errors) - 5} more errors")
        
        return summary