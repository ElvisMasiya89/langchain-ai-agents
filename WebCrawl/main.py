"""
Main entry point for WebCrawl - Parallel Web Scraping with Tavily & Pinecone.

Provides both CLI interface and programmatic access to the complete
web crawling and vector storage pipeline.
"""

import asyncio
import argparse
import json
import sys
from typing import List, Optional
from dotenv import load_dotenv

from backend.orchestrator import AsyncWebCrawlOrchestrator, PipelineStats

# Load environment variables
load_dotenv()


async def crawl_urls(
    urls: List[str],
    instructions: str = "Extract all useful content and information",
    max_depth: int = 3,
    limit_per_url: int = 50,
    output_file: Optional[str] = None
) -> PipelineStats:
    """
    Main function to crawl URLs and store in Pinecone.
    
    Args:
        urls: List of URLs to crawl
        instructions: Crawl instructions for content filtering
        max_depth: Maximum crawl depth
        limit_per_url: Maximum pages per URL
        output_file: Optional file to save results summary
        
    Returns:
        Pipeline statistics
        
    Example:
        >>> urls = ["https://docs.python.org/3/", "https://fastapi.tiangolo.com/"]
        >>> stats = await crawl_urls(urls, instructions="Find Python documentation")
        >>> print(f"Uploaded {stats.total_vectors_uploaded} vectors")
    """
    print(f"🚀 Starting WebCrawl pipeline for {len(urls)} URLs...")
    print(f"📋 Instructions: {instructions}")
    print(f"🔍 Max depth: {max_depth}, Limit per URL: {limit_per_url}")
    print("-" * 60)
    
    # Initialize orchestrator
    orchestrator = AsyncWebCrawlOrchestrator()
    
    # Execute pipeline
    stats = await orchestrator.crawl_and_store_urls(
        urls=urls,
        crawl_instructions=instructions,
        max_depth=max_depth,
        limit_per_url=limit_per_url
    )
    
    # Generate and display summary
    summary = orchestrator.get_pipeline_summary(stats)
    
    print("\n" + "="*60)
    print("📊 PIPELINE SUMMARY")
    print("="*60)
    
    # Pipeline Overview
    overview = summary['pipeline_overview']
    print(f"📈 URLs Processed: {overview['total_urls']}")
    print(f"✅ Successful Crawls: {overview['successful_crawls']}")
    print(f"❌ Failed Crawls: {overview['failed_crawls']}")
    print(f"🎯 Success Rate: {overview['success_rate']}")
    
    # Content Processing
    content = summary['content_processing']
    print(f"📄 Content Chunks: {content['total_chunks_created']}")
    print(f"⬆️  Vectors Uploaded: {content['vectors_uploaded']}")
    print(f"💾 Upload Success: {content['upload_success_rate']}")
    
    # Performance Metrics
    perf = summary['performance_metrics']
    print(f"⏱️  Total Time: {perf['total_time']}")
    print(f"🕷️  Crawl Time: {perf['crawl_time']}")
    print(f"📤 Upload Time: {perf['upload_time']}")
    print(f"⚡ Throughput: {perf['throughput']}")
    
    # Upload Details
    if 'upload_details' in summary:
        upload = summary['upload_details']
        print(f"📦 Batches: {upload['batches_processed']}")
        print(f"⏳ Rate Limits Hit: {upload['rate_limit_hits']}")
        print(f"🔄 Retries: {upload['retries_performed']}")
    
    # Errors
    if 'errors' in summary and summary['errors']:
        print(f"\n⚠️  Errors ({len(summary['errors'])}):")
        for error in summary['errors']:
            print(f"   • {error}")
    
    print("="*60)
    
    # Save results to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"💾 Results saved to: {output_file}")
    
    return stats


async def crawl_domain(
    domain_url: str,
    instructions: str = "Extract all documentation and useful content",
    max_depth: int = 3,
    limit: int = 100,
    output_file: Optional[str] = None
) -> PipelineStats:
    """
    Crawl an entire domain comprehensively.
    
    Optimized for documentation sites and knowledge bases.
    
    Args:
        domain_url: Base URL for domain crawling
        instructions: Specific crawl instructions
        max_depth: Maximum crawl depth
        limit: Maximum pages to crawl
        output_file: Optional results file
        
    Returns:
        Pipeline statistics
    """
    print(f"🌐 Starting comprehensive domain crawl: {domain_url}")
    print(f"📋 Instructions: {instructions}")
    print(f"🔍 Max depth: {max_depth}, Page limit: {limit}")
    print("-" * 60)
    
    orchestrator = AsyncWebCrawlOrchestrator()
    
    stats = await orchestrator.crawl_single_domain(
        base_url=domain_url,
        crawl_instructions=instructions,
        max_depth=max_depth,
        limit=limit
    )
    
    # Display results (reuse the display logic from crawl_urls)
    summary = orchestrator.get_pipeline_summary(stats)
    
    print("\n" + "="*60)
    print("📊 DOMAIN CRAWL SUMMARY")
    print("="*60)
    print(f"🌐 Domain: {domain_url}")
    print(f"📄 Content Chunks: {summary['content_processing']['total_chunks_created']}")
    print(f"⬆️  Vectors Uploaded: {summary['content_processing']['vectors_uploaded']}")
    print(f"⏱️  Total Time: {summary['performance_metrics']['total_time']}")
    print(f"⚡ Throughput: {summary['performance_metrics']['throughput']}")
    print("="*60)
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"💾 Results saved to: {output_file}")
    
    return stats


def main():
    """CLI interface for WebCrawl."""
    parser = argparse.ArgumentParser(
        description="WebCrawl - Parallel Web Scraping with Tavily & Pinecone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crawl multiple URLs
  python main.py --urls https://docs.python.org https://fastapi.tiangolo.com
  
  # Crawl a single domain comprehensively
  python main.py --domain https://docs.python.org --limit 200
  
  # Crawl with custom instructions
  python main.py --urls https://example.com --instructions "Find API documentation"
  
  # Save results to file
  python main.py --domain https://docs.python.org --output results.json
        """
    )
    
    # URL input options
    url_group = parser.add_mutually_exclusive_group(required=True)
    url_group.add_argument(
        '--urls', 
        nargs='+', 
        help='List of URLs to crawl'
    )
    url_group.add_argument(
        '--domain', 
        help='Single domain to crawl comprehensively'
    )
    
    # Crawl configuration
    parser.add_argument(
        '--instructions', 
        default='Extract all useful content and information',
        help='Instructions for crawl content filtering'
    )
    parser.add_argument(
        '--max-depth', 
        type=int, 
        default=3,
        help='Maximum crawl depth (default: 3)'
    )
    parser.add_argument(
        '--limit', 
        type=int, 
        default=50,
        help='Maximum pages per URL (default: 50)'
    )
    parser.add_argument(
        '--output', 
        help='Output file for results summary (JSON)'
    )
    
    args = parser.parse_args()
    
    # Validate environment variables
    import os
    required_vars = ['TAVILY_API_KEY', 'PINECONE_API_KEY', 'OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   • {var}")
        print("\nPlease set these in your .env file.")
        sys.exit(1)
    
    # Execute crawling
    try:
        if args.urls:
            # Multi-URL crawling
            asyncio.run(crawl_urls(
                urls=args.urls,
                instructions=args.instructions,
                max_depth=args.max_depth,
                limit_per_url=args.limit,
                output_file=args.output
            ))
        else:
            # Single domain crawling
            asyncio.run(crawl_domain(
                domain_url=args.domain,
                instructions=args.instructions,
                max_depth=args.max_depth,
                limit=args.limit,
                output_file=args.output
            ))
            
        print("✅ WebCrawl completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Crawling interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()