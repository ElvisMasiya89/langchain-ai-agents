"""
Example usage of WebCrawl for different scenarios.

Demonstrates various ways to use the async web crawling and vector
storage pipeline for different types of content and use cases.
"""

import asyncio
from backend.orchestrator import AsyncWebCrawlOrchestrator


async def example_documentation_crawl():
    """
    Example: Crawl technical documentation sites.
    
    Demonstrates crawling multiple documentation sites with
    specific instructions for extracting technical content.
    """
    print("📚 Example: Documentation Crawl")
    print("="*50)
    
    orchestrator = AsyncWebCrawlOrchestrator()
    
    urls = [
        "https://docs.python.org/3/tutorial/",
        "https://fastapi.tiangolo.com/tutorial/",
        "https://docs.pydantic.dev/latest/",
    ]
    
    instructions = (
        "Extract all documentation content including code examples, "
        "API references, tutorials, and explanations. Focus on "
        "technical information that would be useful for developers."
    )
    
    stats = await orchestrator.crawl_and_store_urls(
        urls=urls,
        crawl_instructions=instructions,
        max_depth=2,
        limit_per_url=30
    )
    
    print(f"✅ Crawled {stats.successful_crawls} documentation sites")
    print(f"📊 Created {stats.total_chunks_created} content chunks")
    print(f"⬆️  Uploaded {stats.total_vectors_uploaded} vectors")
    print()


async def example_news_and_blog_crawl():
    """
    Example: Crawl news articles and blog posts.
    
    Shows how to crawl content-heavy sites with instructions
    focused on extracting article content and metadata.
    """
    print("📰 Example: News & Blog Crawl")
    print("="*50)
    
    orchestrator = AsyncWebCrawlOrchestrator()
    
    urls = [
        "https://realpython.com/",
        "https://towardsdatascience.com/",
        "https://stackoverflow.blog/",
    ]
    
    instructions = (
        "Extract article content, blog posts, and educational materials. "
        "Focus on the main content of articles, including code snippets, "
        "explanations, and practical examples. Skip navigation and ads."
    )
    
    stats = await orchestrator.crawl_and_store_urls(
        urls=urls,
        crawl_instructions=instructions,
        max_depth=2,
        limit_per_url=25
    )
    
    print(f"✅ Crawled {stats.successful_crawls} content sites")
    print(f"📊 Created {stats.total_chunks_created} content chunks")
    print(f"⬆️  Uploaded {stats.total_vectors_uploaded} vectors")
    print()


async def example_single_domain_comprehensive():
    """
    Example: Comprehensive single-domain crawl.
    
    Demonstrates deep crawling of a single domain to build
    a comprehensive knowledge base from one source.
    """
    print("🌐 Example: Comprehensive Domain Crawl")
    print("="*50)
    
    orchestrator = AsyncWebCrawlOrchestrator()
    
    # Crawl a comprehensive documentation site
    stats = await orchestrator.crawl_single_domain(
        base_url="https://docs.langchain.com/",
        crawl_instructions=(
            "Extract all LangChain documentation including tutorials, "
            "API references, examples, and guides. Focus on technical "
            "content that would help developers learn and use LangChain."
        ),
        max_depth=3,
        limit=100
    )
    
    print(f"✅ Comprehensively crawled LangChain docs")
    print(f"📊 Created {stats.total_chunks_created} content chunks")
    print(f"⬆️  Uploaded {stats.total_vectors_uploaded} vectors")
    print(f"⏱️  Total time: {stats.total_pipeline_time:.2f}s")
    print()


async def example_parallel_processing():
    """
    Example: Demonstrate parallel processing capabilities.
    
    Shows how the system handles multiple large crawl jobs
    concurrently with proper resource management.
    """
    print("⚡ Example: Parallel Processing Power")
    print("="*50)
    
    orchestrator = AsyncWebCrawlOrchestrator()
    
    # Multiple domain crawls in parallel
    tasks = [
        orchestrator.crawl_single_domain(
            "https://docs.python.org/3/",
            "Extract Python documentation and tutorials",
            max_depth=2, limit=20
        ),
        orchestrator.crawl_single_domain(
            "https://docs.pydantic.dev/",
            "Extract Pydantic documentation and examples", 
            max_depth=2, limit=20
        ),
        orchestrator.crawl_single_domain(
            "https://fastapi.tiangolo.com/",
            "Extract FastAPI documentation and tutorials",
            max_depth=2, limit=20
        )
    ]
    
    print("🚀 Starting 3 parallel domain crawls...")
    results = await asyncio.gather(*tasks)
    
    total_vectors = sum(stats.total_vectors_uploaded for stats in results)
    total_time = max(stats.total_pipeline_time for stats in results)
    
    print(f"✅ Completed 3 parallel crawls")
    print(f"📊 Total vectors uploaded: {total_vectors}")
    print(f"⏱️  Total time: {total_time:.2f}s (parallel execution)")
    print(f"⚡ Effective throughput: {total_vectors/total_time:.1f} vectors/sec")
    print()


async def example_custom_filtering():
    """
    Example: Custom content filtering with specific instructions.
    
    Shows how to use targeted instructions to extract only
    specific types of content from websites.
    """
    print("🎯 Example: Custom Content Filtering")
    print("="*50)
    
    orchestrator = AsyncWebCrawlOrchestrator()
    
    urls = [
        "https://github.com/python/cpython",
        "https://github.com/microsoft/TypeScript",
        "https://github.com/facebook/react"
    ]
    
    instructions = (
        "Focus exclusively on README files, documentation, and "
        "getting started guides. Extract setup instructions, "
        "installation guides, and basic usage examples. "
        "Skip issue discussions and code files."
    )
    
    stats = await orchestrator.crawl_and_store_urls(
        urls=urls,
        crawl_instructions=instructions,
        max_depth=1,  # Shallow crawl for specific content
        limit_per_url=15
    )
    
    print(f"✅ Targeted crawl of {len(urls)} GitHub repos")
    print(f"📊 Created {stats.total_chunks_created} content chunks")
    print(f"🎯 Focused on README and documentation content")
    print()


async def run_all_examples():
    """Run all example scenarios."""
    print("🚀 WebCrawl Examples - Comprehensive Demo")
    print("="*60)
    print()
    
    # Run examples in sequence
    await example_documentation_crawl()
    await example_news_and_blog_crawl() 
    await example_single_domain_comprehensive()
    await example_custom_filtering()
    await example_parallel_processing()
    
    print("🎉 All examples completed successfully!")
    print("="*60)


if __name__ == "__main__":
    # Run specific example or all examples
    import sys
    
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        examples = {
            'docs': example_documentation_crawl,
            'news': example_news_and_blog_crawl, 
            'domain': example_single_domain_comprehensive,
            'filter': example_custom_filtering,
            'parallel': example_parallel_processing,
        }
        
        if example_name in examples:
            asyncio.run(examples[example_name]())
        else:
            print(f"Unknown example: {example_name}")
            print(f"Available examples: {', '.join(examples.keys())}")
    else:
        # Run all examples
        asyncio.run(run_all_examples())