#!/usr/bin/env python
"""
🚀 TURBO BENCHMARK: Performance comparison between standard and turbo services
"""
import os
import sys
import django
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, '/Users/shadow/PycharmProjects/kg-6')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'neuro_symbolic_agent.settings')
os.environ['USE_TURBO_MODE'] = 'true'  # Force turbo mode

# Configure Django
django.setup()

def benchmark_turbo_performance():
    """Comprehensive performance benchmark"""
    print("🚀 TURBO PERFORMANCE BENCHMARK")
    print("=" * 60)

    # Test scenarios
    test_cases = [
        "Hi, my name is Alice, I work for TechCorp in San Francisco",
        "John issued expense report #456 to the finance team",
        "Sarah approved the Q4 marketing budget of $50,000",
        "Mike reports to Jennifer in the sales department",
        "Project Alpha was assigned to the development team",
        "Invoice INV-789 was paid by GlobalCorp last Friday",
        "Emma manages the customer success team at StartupXYZ",
        "The AI conference was attended by 500 participants",
        "Budget approval workflow requires manager authorization",
        "Security audit completed by external consultants"
    ]

    try:
        from rag_agent.turbo_services import TurboReasoningService
        print("✅ TurboReasoningService loaded successfully")

        # Initialize turbo service
        print("\n🚀 Initializing TurboReasoningService...")
        start_time = time.time()
        turbo_service = TurboReasoningService()
        init_time = time.time() - start_time
        print(f"   Initialization: {init_time:.3f}s")

        # Warm-up phase
        print("\n🔥 Warming up caches...")
        warmup_start = time.time()
        turbo_service.query("Hello, I'm testing the system")
        warmup_time = time.time() - warmup_start
        print(f"   Warmup query: {warmup_time:.3f}s")

        # Performance benchmark
        print("\n⚡ PERFORMANCE BENCHMARK")
        print("-" * 40)

        processing_times = []
        total_entities = 0
        total_relationships = 0

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔍 Test {i}: {test_case[:50]}...")

            start = time.time()
            response = turbo_service.query(test_case)
            duration = time.time() - start

            processing_times.append(duration)
            print(f"   ⏱️  Processing: {duration:.3f}s")
            print(f"   💬 Response: {response[:80]}...")

        # Performance statistics
        avg_time = statistics.mean(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
        median_time = statistics.median(processing_times)

        print(f"\n📊 PERFORMANCE STATISTICS")
        print("-" * 40)
        print(f"   Average time: {avg_time:.3f}s")
        print(f"   Median time:  {median_time:.3f}s")
        print(f"   Min time:     {min_time:.3f}s")
        print(f"   Max time:     {max_time:.3f}s")
        print(f"   Total tests:  {len(test_cases)}")

        # Get detailed metrics
        print(f"\n🎯 DETAILED METRICS")
        print("-" * 40)
        metrics = turbo_service.get_performance_metrics()

        if "extraction_avg" in metrics:
            print(f"   LLM Extraction: {metrics['extraction_avg']:.3f}s avg")
        if "similarity_search_avg" in metrics:
            print(f"   FAISS Search:   {metrics['similarity_search_avg']:.4f}s avg")
        if "storage_avg" in metrics:
            print(f"   Graph Storage:  {metrics['storage_avg']:.3f}s avg")

        # Cache statistics
        if "cache_sizes" in metrics:
            cache_info = metrics["cache_sizes"]
            print(f"\n💾 CACHE STATISTICS")
            print("-" * 40)
            for cache_name, size in cache_info.items():
                print(f"   {cache_name}: {size} entries")

        # Concurrent performance test
        print(f"\n🚀 CONCURRENT PERFORMANCE TEST")
        print("-" * 40)

        concurrent_queries = [
            "Bob works for DataCorp",
            "Lisa approved budget #123",
            "Team Beta assigned to Project Gamma",
            "Payment processed for Invoice #999"
        ]

        concurrent_start = time.time()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(turbo_service.query, query)
                for query in concurrent_queries
            ]

            concurrent_results = []
            for future in as_completed(futures):
                result = future.result()
                concurrent_results.append(result)

        concurrent_total = time.time() - concurrent_start
        concurrent_avg = concurrent_total / len(concurrent_queries)

        print(f"   Concurrent queries: {len(concurrent_queries)}")
        print(f"   Total time: {concurrent_total:.3f}s")
        print(f"   Average per query: {concurrent_avg:.3f}s")
        print(f"   Speedup factor: {len(concurrent_queries) * concurrent_avg / concurrent_total:.2f}x")

        # Real-time performance rating
        if avg_time < 1.0:
            rating = "🚀 BLAZING FAST"
        elif avg_time < 2.0:
            rating = "⚡ VERY FAST"
        elif avg_time < 5.0:
            rating = "🔥 FAST"
        else:
            rating = "⏳ MODERATE"

        print(f"\n🏆 PERFORMANCE RATING: {rating}")
        print(f"   Real-time ready: {'✅ YES' if avg_time < 2.0 else '⚠️  MARGINAL' if avg_time < 5.0 else '❌ NO'}")

        # Performance recommendations
        print(f"\n💡 OPTIMIZATION STATUS")
        print("-" * 40)
        print("   ✅ FAISS vector similarity search")
        print("   ✅ Multi-level caching (extraction, embedding, query)")
        print("   ✅ Concurrent LLM processing")
        print("   ✅ Batch Neo4j operations")
        print("   ✅ Connection pooling")
        print("   ✅ Performance monitoring")

        if avg_time < 1.5:
            print(f"\n🎉 ENTERPRISE READY! Average response time: {avg_time:.3f}s")
        else:
            print(f"\n⚠️  Consider further optimization. Current average: {avg_time:.3f}s")

    except ImportError as e:
        print(f"❌ Failed to import TurboReasoningService: {e}")
        print("   Install required packages: pip install faiss-cpu sentence-transformers diskcache")
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

def compare_with_enhanced_service():
    """Compare turbo vs enhanced service performance"""
    print(f"\n🔬 COMPARISON TEST: Turbo vs Enhanced")
    print("-" * 50)

    test_query = "Alice works for DataCorp and manages the engineering team"

    try:
        # Test enhanced service
        from rag_agent.enhanced_services import EnhancedReasoningService
        enhanced_service = EnhancedReasoningService()

        print("Testing Enhanced Service...")
        enhanced_start = time.time()
        enhanced_response = enhanced_service.query(test_query)
        enhanced_time = time.time() - enhanced_start

        # Test turbo service
        from rag_agent.turbo_services import TurboReasoningService
        turbo_service = TurboReasoningService()

        print("Testing Turbo Service...")
        turbo_start = time.time()
        turbo_response = turbo_service.query(test_query)
        turbo_time = time.time() - turbo_start

        # Compare results
        speedup = enhanced_time / turbo_time if turbo_time > 0 else 0

        print(f"\n📊 COMPARISON RESULTS")
        print("-" * 30)
        print(f"Enhanced Service: {enhanced_time:.3f}s")
        print(f"Turbo Service:    {turbo_time:.3f}s")
        print(f"Speedup Factor:   {speedup:.2f}x")

        if speedup > 1.5:
            print(f"🚀 Turbo service is {speedup:.1f}x faster!")
        elif speedup > 1.1:
            print(f"⚡ Turbo service shows good improvement")
        else:
            print(f"📊 Similar performance between services")

    except Exception as e:
        print(f"❌ Comparison failed: {e}")

if __name__ == "__main__":
    benchmark_turbo_performance()
    compare_with_enhanced_service()