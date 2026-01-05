/// @file bench_pgm_e2e.cpp
/// @brief End-to-end benchmarks for SIMDPGMIndex vs vanilla PGM-Index

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include <pgm/pgm_index.hpp>
#include "simdex/simdex.hpp"

namespace {

// ============================================================================
// Data generation utilities
// ============================================================================

/// Generate contiguous sorted data (best case for learned index)
std::vector<std::uint64_t> generate_contiguous(std::size_t size) {
    std::vector<std::uint64_t> data(size);
    std::iota(data.begin(), data.end(), 0);
    return data;
}

/// Generate random unique sorted data
std::vector<std::uint64_t> generate_random_unique(std::size_t size, std::uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::uint64_t> dist(0, size * 100);

    std::set<std::uint64_t> unique_values;
    while (unique_values.size() < size) {
        unique_values.insert(dist(rng));
    }

    return std::vector<std::uint64_t>(unique_values.begin(), unique_values.end());
}

/// Generate dense clustered data (realistic for many workloads)
std::vector<std::uint64_t> generate_clustered(std::size_t size, std::uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::vector<std::uint64_t> data(size);

    std::uint64_t base = 0;
    for (std::size_t i = 0; i < size; ++i) {
        // Mostly sequential with occasional gaps
        std::uniform_int_distribution<std::uint64_t> gap(1, 10);
        base += gap(rng);
        data[i] = base;
    }

    return data;
}

/// Generate query keys (mix of existing and non-existing)
std::vector<std::uint64_t> generate_queries(const std::vector<std::uint64_t>& data,
                                             std::size_t count, double hit_ratio = 0.8) {
    std::mt19937_64 rng(123);
    std::uniform_int_distribution<std::size_t> idx_dist(0, data.size() - 1);
    std::uniform_int_distribution<std::uint64_t> miss_dist(0, data.back() * 2);
    std::bernoulli_distribution hit_dist(hit_ratio);

    std::vector<std::uint64_t> queries(count);
    for (auto& q : queries) {
        if (hit_dist(rng)) {
            // Hit: use existing key
            q = data[idx_dist(rng)];
        } else {
            // Miss: use random key (may or may not exist)
            q = miss_dist(rng);
        }
    }
    return queries;
}

// ============================================================================
// Vanilla PGM-Index benchmark (baseline)
// ============================================================================

template<std::size_t Epsilon>
static void BM_VanillaPGM(benchmark::State& state) {
    const std::size_t data_size = static_cast<std::size_t>(state.range(0));
    auto data = generate_random_unique(data_size);
    auto queries = generate_queries(data, 10000);

    pgm::PGMIndex<std::uint64_t, Epsilon> index(data.begin(), data.end());

    std::size_t query_idx = 0;
    for (auto _ : state) {
        auto key = queries[query_idx++ % queries.size()];

        // PGM search + std::lower_bound for last-mile
        auto approx = index.search(key);
        auto lo = data.begin() + approx.lo;
        auto hi = data.begin() + std::min(approx.hi, data.size());
        auto result = std::lower_bound(lo, hi, key);

        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
    state.SetLabel("vanilla_pgm_eps" + std::to_string(Epsilon));
}

// ============================================================================
// SIMD PGM-Index benchmarks
// ============================================================================

template<std::size_t Epsilon, simdex::StrategyPolicy Policy>
static void BM_SIMDPGM(benchmark::State& state) {
    const std::size_t data_size = static_cast<std::size_t>(state.range(0));
    auto data = generate_random_unique(data_size);
    auto queries = generate_queries(data, 10000);

    simdex::SIMDPGMIndex<std::uint64_t, Epsilon> index(std::move(data));

    std::size_t query_idx = 0;
    for (auto _ : state) {
        auto key = queries[query_idx++ % queries.size()];
        auto result = index.lower_bound(key, Policy);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));

    const char* policy_name = "auto";
    if constexpr (Policy == simdex::StrategyPolicy::AlwaysLinear) policy_name = "linear";
    if constexpr (Policy == simdex::StrategyPolicy::AlwaysKary) policy_name = "kary";
    if constexpr (Policy == simdex::StrategyPolicy::AlwaysScalar) policy_name = "scalar";

    state.SetLabel(std::string("simd_pgm_eps") + std::to_string(Epsilon) + "_" + policy_name);
}

// ============================================================================
// Register benchmarks
// ============================================================================

// Data sizes to test
#define DATA_SIZES Arg(100000)->Arg(1000000)->Arg(10000000)

// Epsilon = 32 (small ranges, ~64 elements to search)
BENCHMARK(BM_VanillaPGM<32>)->DATA_SIZES->Unit(benchmark::kNanosecond);
BENCHMARK(BM_SIMDPGM<32, simdex::StrategyPolicy::Auto>)->DATA_SIZES->Unit(benchmark::kNanosecond);
BENCHMARK(BM_SIMDPGM<32, simdex::StrategyPolicy::AlwaysLinear>)->DATA_SIZES->Unit(benchmark::kNanosecond);
BENCHMARK(BM_SIMDPGM<32, simdex::StrategyPolicy::AlwaysKary>)->DATA_SIZES->Unit(benchmark::kNanosecond);

// Epsilon = 64 (default, ~128 elements to search)
BENCHMARK(BM_VanillaPGM<64>)->DATA_SIZES->Unit(benchmark::kNanosecond);
BENCHMARK(BM_SIMDPGM<64, simdex::StrategyPolicy::Auto>)->DATA_SIZES->Unit(benchmark::kNanosecond);
BENCHMARK(BM_SIMDPGM<64, simdex::StrategyPolicy::AlwaysLinear>)->DATA_SIZES->Unit(benchmark::kNanosecond);
BENCHMARK(BM_SIMDPGM<64, simdex::StrategyPolicy::AlwaysKary>)->DATA_SIZES->Unit(benchmark::kNanosecond);

// Epsilon = 128 (larger ranges, ~256 elements to search)
BENCHMARK(BM_VanillaPGM<128>)->DATA_SIZES->Unit(benchmark::kNanosecond);
BENCHMARK(BM_SIMDPGM<128, simdex::StrategyPolicy::Auto>)->DATA_SIZES->Unit(benchmark::kNanosecond);
BENCHMARK(BM_SIMDPGM<128, simdex::StrategyPolicy::AlwaysLinear>)->DATA_SIZES->Unit(benchmark::kNanosecond);
BENCHMARK(BM_SIMDPGM<128, simdex::StrategyPolicy::AlwaysKary>)->DATA_SIZES->Unit(benchmark::kNanosecond);

// ============================================================================
// Throughput benchmark (batch queries)
// ============================================================================

template<std::size_t Epsilon>
static void BM_Throughput_Vanilla(benchmark::State& state) {
    auto data = generate_random_unique(1'000'000);
    auto queries = generate_queries(data, 10000);

    pgm::PGMIndex<std::uint64_t, Epsilon> index(data.begin(), data.end());

    for (auto _ : state) {
        for (const auto& key : queries) {
            auto approx = index.search(key);
            auto lo = data.begin() + approx.lo;
            auto hi = data.begin() + std::min(approx.hi, data.size());
            auto result = std::lower_bound(lo, hi, key);
            benchmark::DoNotOptimize(result);
        }
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * queries.size()));
    state.SetLabel("vanilla_throughput_eps" + std::to_string(Epsilon));
}

template<std::size_t Epsilon>
static void BM_Throughput_SIMD(benchmark::State& state) {
    auto data = generate_random_unique(1'000'000);
    auto queries = generate_queries(data, 10000);

    simdex::SIMDPGMIndex<std::uint64_t, Epsilon> index(std::move(data));

    for (auto _ : state) {
        for (const auto& key : queries) {
            auto result = index.lower_bound(key);
            benchmark::DoNotOptimize(result);
        }
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * queries.size()));
    state.SetLabel("simd_throughput_eps" + std::to_string(Epsilon));
}

BENCHMARK(BM_Throughput_Vanilla<64>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Throughput_SIMD<64>)->Unit(benchmark::kMillisecond);

// ============================================================================
// Data distribution comparison
// ============================================================================

template<typename DataGenerator>
static void BM_Distribution(benchmark::State& state, DataGenerator gen) {
    auto data = gen(1'000'000);
    auto queries = generate_queries(data, 10000);

    simdex::SIMDPGMIndex<std::uint64_t, 64> index(std::move(data));

    for (auto _ : state) {
        for (const auto& key : queries) {
            auto result = index.lower_bound(key);
            benchmark::DoNotOptimize(result);
        }
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * queries.size()));
}

BENCHMARK_CAPTURE(BM_Distribution, contiguous, generate_contiguous)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_Distribution, random, [](std::size_t s) { return generate_random_unique(s); })->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_Distribution, clustered, [](std::size_t s) { return generate_clustered(s); })->Unit(benchmark::kMillisecond);

} // namespace
