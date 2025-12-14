"""
Basic structure test for WebCrawl components.

Verifies that all modules can be imported and basic functionality
is accessible without running full crawling operations.
"""

def test_imports():
    """Test that all core modules can be imported successfully."""
    try:
        from backend.crawler import AsyncTavilyCrawler, CrawlResult
        from backend.uploader import AsyncPineconeUploader, UploadStats, VectorRecord
        from backend.orchestrator import AsyncWebCrawlOrchestrator, PipelineStats
        from consts import INDEX_NAME, PINECONE_BATCH_SIZE
        
        print("✅ All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_configuration():
    """Test configuration constants are properly defined."""
    try:
        from consts import (
            INDEX_NAME, PINECONE_BATCH_SIZE, TAVILY_MAX_DEPTH,
            MAX_CONCURRENT_CRAWLS, MAX_CONCURRENT_UPLOADS
        )
        
        assert isinstance(INDEX_NAME, str)
        assert isinstance(PINECONE_BATCH_SIZE, int)
        assert isinstance(TAVILY_MAX_DEPTH, int)
        assert isinstance(MAX_CONCURRENT_CRAWLS, int)
        assert isinstance(MAX_CONCURRENT_UPLOADS, int)
        
        print("✅ Configuration constants are properly defined")
        return True
    except (ImportError, AssertionError) as e:
        print(f"❌ Configuration error: {e}")
        return False


def test_dataclasses():
    """Test that dataclasses can be instantiated."""
    try:
        from backend.crawler import CrawlResult
        from backend.uploader import UploadStats, VectorRecord
        from backend.orchestrator import PipelineStats
        
        # Test CrawlResult
        crawl_result = CrawlResult(
            url="https://example.com",
            title="Test Page",
            content="Test content",
            metadata={"test": "data"},
            chunks=["chunk1", "chunk2"]
        )
        
        # Test UploadStats
        upload_stats = UploadStats()
        
        # Test VectorRecord
        vector_record = VectorRecord(
            id="test_id",
            values=[0.1, 0.2, 0.3],
            metadata={"test": "metadata"}
        )
        
        # Test PipelineStats
        pipeline_stats = PipelineStats()
        
        print("✅ All dataclasses can be instantiated")
        return True
    except Exception as e:
        print(f"❌ Dataclass error: {e}")
        return False


def test_environment_setup():
    """Test environment variable setup guidance."""
    import os
    
    required_vars = ['TAVILY_API_KEY', 'PINECONE_API_KEY', 'OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("   Please copy .env.example to .env and fill in your API keys")
        print("   This is expected for the structure test")
    else:
        print("✅ All required environment variables are set")
    
    return True


def main():
    """Run all structure tests."""
    print("🧪 WebCrawl Structure Tests")
    print("="*40)
    
    tests = [
        test_imports,
        test_configuration,
        test_dataclasses,
        test_environment_setup
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            results.append(False)
        print()
    
    passed = sum(results)
    total = len(results)
    
    print("="*40)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All structure tests passed! WebCrawl is ready to use.")
        print("   Next steps:")
        print("   1. Copy .env.example to .env")
        print("   2. Fill in your API keys")
        print("   3. Install dependencies: pip install -r requirements.txt")
        print("   4. Run: python main.py --help")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    main()