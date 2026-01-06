/// @file bench_batch.cpp
/// @brief Benchmarks for batched query processing
///
/// This benchmark measures the novel contribution: batched query processing
/// with segment coalescing and prediction-guided optimization.
///
/// Key hypothesis: Batched processing with query reordering and segment
/// coalescing should provide significant speedups (2-5x) over sequential
/// processing, especially for workloads with spatial locality.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "simdex/simdex.hpp"

namespace {

// ============================================================================
// Data generation utilities
// ============================================================================

std::vector<std::uint64_t> generate_sorted_data(std::size_t size, std::uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::vector<std::uint64_t> data(size);

    std::uint64_t val = 0;
    for (auto& d : data) {
        std::uniform_int_distribution<std::uint64_t> gap(1, 10);
        val += gap(rng);
        d = val;
    }

    return data;
}

/// Generate queries with spatial locality (clustered)
/// Real-world queries often have locality - queries near each other in time
/// tend to access nearby data
std::vector<std::uint64_t> generate_clustered_queries(
    const std::vector<std::uint64_t>& data,
    std::size_t count,
    std::size_t num_clusters = 10,
    std::uint64_t seed = 123
) {
    std::mt19937_64 rng(seed);
    std::vector<std::uint64_t> queries;
    queries.reserve(count);

    // Pick cluster centers
    std::vector<std::size_t> cluster_centers;
    std::uniform_int_distribution<std::size_t> center_dist(0, data.size() - 1);
    for (std::size_t i = 0; i < num_clusters; ++i) {
        cluster_centers.push_back(center_dist(rng));
    }

    // Generate queries around cluster centers
    std::uniform_int_distribution<std::size_t> cluster_pick(0, num_clusters - 1);
    for (std::size_t i = 0; i < count; ++i) {
        std::size_t center = cluster_centers[cluster_pick(rng)];

        // Query within ±100 elements of center (simulates spatial locality)
        std::uniform_int_distribution<int> offset(-100, 100);
        std::size_t idx = static_cast<std::size_t>(
            std::max<int>(0, std::min<int>(static_cast<int>(data.size() - 1),
                                           static_cast<int>(center) + offset(rng))));

        // 80% exact hits, 20% near misses
        std::bernoulli_distribution hit(0.8);
        if (hit(rng)) {
            queries.push_back(data[idx]);
        } else {
            queries.push_back(data[idx] + 1);  // Near miss
        }
    }

    return queries;
}

/// Generate uniformly random queries (no locality)
std::vector<std::uint64_t> generate_random_queries(
    const std::vector<std::uint64_t>& data,
    std::size_t count,
    std::uint64_t seed = 456
) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t> idx_dist(0, data.size() - 1);

    std::vector<std::uint64_t> queries;
    queries.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        queries.push_back(data[idx_dist(rng)]);
    }

    return queries;
}

// ============================================================================
// Baseline: Sequential processing (one query at a time)
// ============================================================================

static void BM_Sequential_SingleQuery(benchmark::State& state) {
    const std::size_t data_size = 1'000'000;
    const std::size_t batch_size = static_cast<std::size_t>(state.range(0));

    auto data = generate_sorted_data(data_size);
    auto queries = generate_clustered_queries(data, batch_size);

    simdex::SIMDPGMIndex<std::uint64_t, 64> index(std::move(data));

    for (auto _ : state) {
        std::size_t found_count = 0;
        for (const auto& key : queries) {
            auto it = index.lower_bound(key);
            if (it != index.end() && *it == key) {
                found_count++;
            }
        }
        benchmark::DoNotOptimize(found_count);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * batch_size));
    state.SetLabel("sequential");
}

// ============================================================================
// Batched processing with different strategies
// ============================================================================

template<simdex::BatchStrategy Strategy>
static void BM_Batched(benchmark::State& state) {
    const std::size_t data_size = 1'000'000;
    const std::size_t batch_size = static_cast<std::size_t>(state.range(0));

    auto data = generate_sorted_data(data_size);
    auto queries = generate_clustered_queries(data, batch_size);

    simdex::SIMDPGMIndex<std::uint64_t, 64> index(std::move(data));
    auto processor = simdex::make_batch_processor(index);

    for (auto _ : state) {
        auto result = processor.process(queries, Strategy);
        benchmark::DoNotOptimize(result.positions);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * batch_size));

    const char* strategy_name = "unknown";
    if constexpr (Strategy == simdex::BatchStrategy::Sequential) strategy_name = "batch_sequential";
    if constexpr (Strategy == simdex::BatchStrategy::SortedPrediction) strategy_name = "batch_sorted";
    if constexpr (Strategy == simdex::BatchStrategy::SegmentCoalescing) strategy_name = "batch_coalesced";
    if constexpr (Strategy == simdex::BatchStrategy::Prefetched) strategy_name = "batch_prefetched";
    if constexpr (Strategy == simdex::BatchStrategy::FullyOptimized) strategy_name = "batch_optimized";
    state.SetLabel(strategy_name);
}

// ============================================================================
// Compare with vanilla PGM-Index batched processing
// ============================================================================

static void BM_VanillaPGM_Batch(benchmark::State& state) {
    const std::size_t data_size = 1'000'000;
    const std::size_t batch_size = static_cast<std::size_t>(state.range(0));

    auto data = generate_sorted_data(data_size);
    auto queries = generate_clustered_queries(data, batch_size);

    pgm::PGMIndex<std::uint64_t, 64> index(data.begin(), data.end());

    for (auto _ : state) {
        std::size_t found_count = 0;
        for (const auto& key : queries) {
            auto approx = index.search(key);
            auto lo = data.begin() + static_cast<std::ptrdiff_t>(approx.lo);
            auto hi = data.begin() + static_cast<std::ptrdiff_t>(std::min(approx.hi, data.size()));
            auto it = std::lower_bound(lo, hi, key);
            if (it != data.end() && *it == key) {
                found_count++;
            }
        }
        benchmark::DoNotOptimize(found_count);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * batch_size));
    state.SetLabel("vanilla_pgm");
}

// ============================================================================
// Locality analysis: Compare clustered vs random queries
// ============================================================================

template<simdex::BatchStrategy Strategy>
static void BM_Batched_RandomQueries(benchmark::State& state) {
    const std::size_t data_size = 1'000'000;
    const std::size_t batch_size = static_cast<std::size_t>(state.range(0));

    auto data = generate_sorted_data(data_size);
    auto queries = generate_random_queries(data, batch_size);  // Random, not clustered

    simdex::SIMDPGMIndex<std::uint64_t, 64> index(std::move(data));
    auto processor = simdex::make_batch_processor(index);

    for (auto _ : state) {
        auto result = processor.process(queries, Strategy);
        benchmark::DoNotOptimize(result.positions);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * batch_size));

    const char* strategy_name = "unknown";
    if constexpr (Strategy == simdex::BatchStrategy::Sequential) strategy_name = "random_sequential";
    if constexpr (Strategy == simdex::BatchStrategy::FullyOptimized) strategy_name = "random_optimized";
    state.SetLabel(strategy_name);
}

// ============================================================================
// Efficiency metrics: Cache hits and segments loaded
// ============================================================================

static void BM_Efficiency_Analysis(benchmark::State& state) {
    const std::size_t data_size = 1'000'000;
    const std::size_t batch_size = static_cast<std::size_t>(state.range(0));

    auto data = generate_sorted_data(data_size);
    auto queries = generate_clustered_queries(data, batch_size);

    simdex::SIMDPGMIndex<std::uint64_t, 64> index(std::move(data));
    auto processor = simdex::make_batch_processor(index);

    simdex::BatchResult<std::uint64_t> last_result;

    for (auto _ : state) {
        last_result = processor.process(queries, simdex::BatchStrategy::FullyOptimized);
        benchmark::DoNotOptimize(last_result.positions);
    }

    // Report efficiency metrics
    state.counters["cache_hits"] = benchmark::Counter(
        static_cast<double>(last_result.cache_hits),
        benchmark::Counter::kDefaults
    );
    state.counters["segments"] = benchmark::Counter(
        static_cast<double>(last_result.segments_loaded),
        benchmark::Counter::kDefaults
    );
    state.counters["coalesce_ratio"] = benchmark::Counter(
        static_cast<double>(batch_size) / static_cast<double>(last_result.segments_loaded),
        benchmark::Counter::kDefaults
    );

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * batch_size));
    state.SetLabel("efficiency_analysis");
}

// ============================================================================
// Throughput comparison at different batch sizes
// ============================================================================

static void BM_Throughput_Scaling(benchmark::State& state) {
    const std::size_t data_size = 1'000'000;
    const std::size_t batch_size = static_cast<std::size_t>(state.range(0));

    auto data = generate_sorted_data(data_size);

    simdex::SIMDPGMIndex<std::uint64_t, 64> index(data);
    auto processor = simdex::make_batch_processor(index);

    // Run multiple batches per iteration for more stable measurement
    const std::size_t batches_per_iter = 100;
    std::vector<std::vector<std::uint64_t>> all_queries;
    for (std::size_t i = 0; i < batches_per_iter; ++i) {
        all_queries.push_back(generate_clustered_queries(data, batch_size, 10, 42 + i));
    }

    for (auto _ : state) {
        std::size_t total_found = 0;
        for (const auto& queries : all_queries) {
            auto result = processor.process(queries, simdex::BatchStrategy::FullyOptimized);
            total_found += std::count(result.found.begin(), result.found.end(), true);
        }
        benchmark::DoNotOptimize(total_found);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * batches_per_iter * batch_size));
    state.SetLabel("throughput_scaling");
}

// ============================================================================
// Register benchmarks
// ============================================================================

// Batch sizes to test
#define BATCH_SIZES Arg(64)->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384)

// Main comparison: sequential vs batched strategies
BENCHMARK(BM_Sequential_SingleQuery)->BATCH_SIZES->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_VanillaPGM_Batch)->BATCH_SIZES->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Batched<simdex::BatchStrategy::Sequential>)->BATCH_SIZES->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Batched<simdex::BatchStrategy::SortedPrediction>)->BATCH_SIZES->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Batched<simdex::BatchStrategy::SegmentCoalescing>)->BATCH_SIZES->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Batched<simdex::BatchStrategy::Prefetched>)->BATCH_SIZES->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Batched<simdex::BatchStrategy::FullyOptimized>)->BATCH_SIZES->Unit(benchmark::kMicrosecond);

// Locality comparison
BENCHMARK(BM_Batched_RandomQueries<simdex::BatchStrategy::Sequential>)->BATCH_SIZES->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Batched_RandomQueries<simdex::BatchStrategy::FullyOptimized>)->BATCH_SIZES->Unit(benchmark::kMicrosecond);

// Efficiency analysis
BENCHMARK(BM_Efficiency_Analysis)->BATCH_SIZES->Unit(benchmark::kMicrosecond);

// Throughput scaling
BENCHMARK(BM_Throughput_Scaling)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096)->Unit(benchmark::kMillisecond);

} // namespace
