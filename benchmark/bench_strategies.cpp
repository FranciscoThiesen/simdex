/// @file bench_strategies.cpp
/// @brief Micro-benchmarks for search strategies

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "simdex/simdex.hpp"

namespace {

// Generate sorted test data
std::vector<std::uint64_t> generate_sorted_data(std::size_t size) {
    std::vector<std::uint64_t> data(size);
    std::iota(data.begin(), data.end(), 0);
    return data;
}

// Generate random query keys
std::vector<std::uint64_t> generate_queries(std::size_t count, std::uint64_t max_val) {
    std::vector<std::uint64_t> queries(count);
    std::mt19937_64 rng(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<std::uint64_t> dist(0, max_val);

    for (auto& q : queries) {
        q = dist(rng);
    }
    return queries;
}

// Benchmark scalar baseline with varying range sizes
template<typename Strategy>
static void BM_SearchStrategy(benchmark::State& state) {
    const std::size_t data_size = 1'000'000;
    const std::size_t range_size = static_cast<std::size_t>(state.range(0));
    const std::size_t num_queries = 10000;

    auto data = generate_sorted_data(data_size);
    auto queries = generate_queries(num_queries, data_size - 1);

    Strategy strategy;
    std::size_t query_idx = 0;

    for (auto _ : state) {
        // Simulate last-mile search within a predicted range
        std::uint64_t key = queries[query_idx++ % num_queries];

        // Clamp key to valid range start
        std::size_t approx_pos = std::min(key, data_size - range_size);
        std::size_t lo = approx_pos;
        std::size_t hi = std::min(approx_pos + range_size, data_size);

        auto result = strategy.lower_bound(data.data(), lo, hi, key);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
    state.SetLabel(Strategy::name());
}

// Register benchmarks for different range sizes (epsilon effect)

// Scalar baseline (std::lower_bound)
BENCHMARK(BM_SearchStrategy<simdex::ScalarBaseline<std::uint64_t>>)
    ->Arg(16)->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Arg(512)
    ->Unit(benchmark::kNanosecond);

// SIMD Linear Scan - best for small ranges
BENCHMARK(BM_SearchStrategy<simdex::SIMDLinearScan<std::uint64_t>>)
    ->Arg(16)->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Arg(512)
    ->Unit(benchmark::kNanosecond);

// SIMD K-ary Search - best for medium ranges
BENCHMARK(BM_SearchStrategy<simdex::SIMDKarySearch<std::uint64_t>>)
    ->Arg(16)->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Arg(512)
    ->Unit(benchmark::kNanosecond);

// ============================================================================
// Throughput benchmark (batch queries)
// ============================================================================

template<typename Strategy>
static void BM_Throughput(benchmark::State& state) {
    const std::size_t data_size = 1'000'000;
    const std::size_t range_size = 64;  // Fixed epsilon
    const std::size_t batch_size = 1000;

    auto data = generate_sorted_data(data_size);
    auto queries = generate_queries(batch_size, data_size - 1);

    Strategy strategy;

    for (auto _ : state) {
        for (std::size_t i = 0; i < batch_size; ++i) {
            std::uint64_t key = queries[i];
            std::size_t approx_pos = std::min(key, data_size - range_size);
            std::size_t lo = approx_pos;
            std::size_t hi = std::min(approx_pos + range_size, data_size);

            auto result = strategy.lower_bound(data.data(), lo, hi, key);
            benchmark::DoNotOptimize(result);
        }
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * batch_size));
    state.SetLabel(Strategy::name());
}

BENCHMARK(BM_Throughput<simdex::ScalarBaseline<std::uint64_t>>)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Throughput<simdex::SIMDLinearScan<std::uint64_t>>)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Throughput<simdex::SIMDKarySearch<std::uint64_t>>)
    ->Unit(benchmark::kMicrosecond);

} // namespace
