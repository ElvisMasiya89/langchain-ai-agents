"""
Async Pinecone uploader with rate limiting and batch processing.

This module provides optimized batch uploads to Pinecone with intelligent
rate limiting, retry logic, and parallel processing for maximum throughput.
"""

import asyncio
import logging
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import backoff

import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from .crawler import CrawlResult
from consts import (
    INDEX_NAME, PINECONE_BATCH_SIZE, PINECONE_MAX_BATCH_SIZE,
    PINECONE_MAX_REQUEST_SIZE_MB, PINECONE_WRITE_UNITS_LIMIT,
    MAX_CONCURRENT_UPLOADS, MAX_RETRIES, INITIAL_BACKOFF, BACKOFF_MULTIPLIER
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UploadStats:
    """
    Statistics tracking for batch upload operations.
    
    Provides comprehensive metrics for monitoring upload performance
    and identifying optimization opportunities.
    """
    total_vectors: int = 0
    successful_uploads: int = 0
    failed_uploads: int = 0
    batches_processed: int = 0
    total_upload_time: float = 0.0
    rate_limit_hits: int = 0
    retries_performed: int = 0


@dataclass
class VectorRecord:
    """
    Individual vector record for Pinecone storage.
    
    Represents a single chunk of content with its embedding and metadata
    optimized for vector similarity search and retrieval.
    """
    id: str
    values: List[float]
    metadata: Dict[str, Any]


class AsyncPineconeUploader:
    """
    High-performance async Pinecone uploader with intelligent batching.
    
    Features:
    - Optimal batch sizing (up to 1000 vectors or 2MB per request)
    - Rate limiting with exponential backoff (50,000 write units/sec limit)
    - Parallel uploads with concurrency control
    - Automatic retry logic for transient failures
    - Comprehensive upload statistics and monitoring
    """
    
    def __init__(self, api_key: str, index_name: str = INDEX_NAME):
        """
        Initialize the async Pinecone uploader.
        
        Args:
            api_key: Pinecone API key for authentication
            index_name: Target Pinecone index name
        """
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.index = None
        self.upload_semaphore = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)
        self.rate_limiter = asyncio.Semaphore(PINECONE_WRITE_UNITS_LIMIT)
        
    async def initialize_index(self, dimension: int = 1536) -> None:
        """
        Initialize or connect to Pinecone index.
        
        Creates index if it doesn't exist, otherwise connects to existing index.
        Uses serverless specification for cost-effective scaling.
        
        Args:
            dimension: Vector dimension (1536 for OpenAI text-embedding-3-small)
        """
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            
            if self.index_name not in [idx['name'] for idx in existing_indexes]:
                logger.info(f"Creating new index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                
                # Wait for index to be ready
                while not self.pc.describe_index(self.index_name).status['ready']:
                    await asyncio.sleep(1)
                    
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize index: {e}")
            raise

    async def upload_crawl_results(
        self, 
        crawl_results: List[CrawlResult]
    ) -> UploadStats:
        """
        Upload multiple crawl results to Pinecone with optimal batching.
        
        Process:
        1. Convert crawl results to vector records
        2. Create optimally-sized batches (respecting size and count limits)
        3. Upload batches in parallel with rate limiting
        4. Track statistics and handle retries
        5. Return comprehensive upload metrics
        
        Args:
            crawl_results: List of crawl results to upload
            
        Returns:
            UploadStats with detailed upload metrics
            
        Example:
            >>> uploader = AsyncPineconeUploader(api_key="your_key")
            >>> await uploader.initialize_index()
            >>> stats = await uploader.upload_crawl_results(crawl_results)
            >>> print(f"Uploaded {stats.successful_uploads} vectors")
        """
        if not self.index:
            await self.initialize_index()
            
        start_time = time.time()
        stats = UploadStats()
        
        # Convert crawl results to vector records
        vector_records = self._create_vector_records(crawl_results)
        stats.total_vectors = len(vector_records)
        
        if not vector_records:
            logger.warning("No vector records to upload")
            return stats
        
        logger.info(f"Uploading {len(vector_records)} vectors to Pinecone")
        
        # Create optimally-sized batches
        batches = self._create_batches(vector_records)
        stats.batches_processed = len(batches)
        
        # Upload all batches concurrently
        batch_results = await asyncio.gather(
            *[self._upload_batch_with_retry(batch, stats) for batch in batches],
            return_exceptions=True
        )
        
        # Process batch results
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Batch upload failed: {result}")
                stats.failed_uploads += PINECONE_BATCH_SIZE  # Estimate
            else:
                stats.successful_uploads += result
        
        stats.total_upload_time = time.time() - start_time
        
        logger.info(
            f"Upload completed: {stats.successful_uploads}/{stats.total_vectors} vectors "
            f"in {stats.total_upload_time:.2f}s "
            f"({stats.batches_processed} batches, {stats.rate_limit_hits} rate limits)"
        )
        
        return stats

    async def _upload_batch_with_retry(
        self, 
        batch: List[VectorRecord], 
        stats: UploadStats
    ) -> int:
        """
        Upload a single batch with retry logic and rate limiting.
        
        Implements exponential backoff for rate limit handling and
        respects Pinecone's write unit limitations.
        """
        async with self.upload_semaphore:
            
            @backoff.on_exception(
                backoff.expo,
                Exception,
                max_tries=MAX_RETRIES,
                base=INITIAL_BACKOFF,
                max_value=60.0,
                on_backoff=lambda details: self._on_retry(details, stats)
            )
            async def upload_with_backoff():
                # Rate limiting - acquire permits for write units
                async with self.rate_limiter:
                    loop = asyncio.get_event_loop()
                    
                    # Prepare vectors for Pinecone format
                    vectors = [
                        (record.id, record.values, record.metadata)
                        for record in batch
                    ]
                    
                    # Execute upload in thread pool to avoid blocking
                    await loop.run_in_executor(
                        None,
                        lambda: self.index.upsert(vectors=vectors)
                    )
                    
                    return len(batch)
            
            try:
                return await upload_with_backoff()
            except Exception as e:
                logger.error(f"Batch upload failed after retries: {e}")
                stats.failed_uploads += len(batch)
                return 0

    def _create_vector_records(self, crawl_results: List[CrawlResult]) -> List[VectorRecord]:
        """
        Convert crawl results into Pinecone vector records.
        
        Creates unique IDs for each chunk and structures metadata
        for optimal retrieval and filtering capabilities.
        """
        vector_records = []
        
        for result in crawl_results:
            if result.error or not result.embeddings or not result.chunks:
                continue
                
            # Create vector record for each chunk
            for i, (chunk, embedding) in enumerate(zip(result.chunks, result.embeddings)):
                # Generate unique ID for this chunk
                chunk_id = self._generate_chunk_id(result.url, i, chunk)
                
                # Create comprehensive metadata
                metadata = {
                    'source_url': result.url,
                    'title': result.title,
                    'chunk_index': i,
                    'total_chunks': len(result.chunks),
                    'content_length': len(chunk),
                    'crawl_time': result.crawl_time,
                    'text': chunk[:1000],  # Truncate for metadata limits
                    **result.metadata  # Include additional crawl metadata
                }
                
                # Clean metadata (remove None values and ensure serializable)
                cleaned_metadata = {
                    k: v for k, v in metadata.items() 
                    if v is not None and isinstance(v, (str, int, float, bool, list))
                }
                
                vector_records.append(VectorRecord(
                    id=chunk_id,
                    values=embedding,
                    metadata=cleaned_metadata
                ))
        
        return vector_records

    def _create_batches(self, vector_records: List[VectorRecord]) -> List[List[VectorRecord]]:
        """
        Create optimally-sized batches respecting Pinecone limits.
        
        Balances batch size and request size to maximize throughput
        while staying within Pinecone's 2MB/1000 vector limits.
        """
        batches = []
        current_batch = []
        current_size = 0
        
        for record in vector_records:
            # Estimate record size (embedding + metadata)
            record_size = len(record.values) * 4  # 4 bytes per float
            record_size += len(str(record.metadata).encode('utf-8'))
            
            # Check if adding this record would exceed limits
            if (len(current_batch) >= PINECONE_MAX_BATCH_SIZE or 
                current_size + record_size > PINECONE_MAX_REQUEST_SIZE_MB * 1024 * 1024):
                
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_size = 0
            
            current_batch.append(record)
            current_size += record_size
        
        # Add final batch if not empty
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"Created {len(batches)} batches for {len(vector_records)} vectors")
        return batches

    def _generate_chunk_id(self, url: str, chunk_index: int, content: str) -> str:
        """
        Generate unique, deterministic ID for content chunks.
        
        Creates hash-based IDs that are reproducible and collision-resistant
        for reliable deduplication and updates.
        """
        # Create deterministic hash from URL, index, and content
        content_hash = hashlib.sha256(
            f"{url}:{chunk_index}:{content[:200]}".encode('utf-8')
        ).hexdigest()[:16]
        
        return f"{urlparse(url).netloc}_{chunk_index}_{content_hash}"

    def _on_retry(self, details: Dict[str, Any], stats: UploadStats) -> None:
        """
        Handle retry events for monitoring and statistics.
        
        Tracks retry attempts and rate limit hits for performance monitoring.
        """
        stats.retries_performed += 1
        
        if "rate limit" in str(details.get('exception', '')).lower():
            stats.rate_limit_hits += 1
            
        logger.warning(
            f"Retrying upload (attempt {details['tries']}) after {details['wait']:.2f}s: "
            f"{details.get('exception', 'Unknown error')}"
        )