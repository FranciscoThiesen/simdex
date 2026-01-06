/// @file bench_neon_opt.cpp
/// @brief Benchmark optimized NEON implementations against baseline
///
/// This benchmark specifically targets ARM NEON to find the best strategy.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "simdex/simdex.hpp"

#if defined(SIMDEX_HAS_NEON)
#include "simdex/search/linear_scan_neon_opt.hpp"
#endif

namespace {

// Pre-generate data and queries
struct TestData {
    std::vector<std::uint64_t> data;
    std::vector<std::uint64_t> queries;
    std::vector<std::size_t> positions;

    TestData() : data(100000), queries(10000), positions(10000) {
        std::iota(data.begin(), data.end(), 0);

        std::mt19937_64 rng(42);
        for (auto& q : queries) {
            q = rng() % data.size();
        }
        for (auto& p : positions) {
            p = rng() % (data.size() - 1000);
        }
    }
};

static TestData& get_test_data() {
    static TestData td;
    return td;
}

// Benchmark a strategy at a given range size
template<typename Strategy>
static void BM_Strategy(benchmark::State& state) {
    auto& td = get_test_data();
    const std::size_t range_size = static_cast<std::size_t>(state.range(0));

    Strategy strategy;
    std::size_t idx = 0;

    for (auto _ : state) {
        std::size_t pos = td.positions[idx % td.positions.size()];
        std::uint64_t key = td.queries[idx % td.queries.size()];
        ++idx;

        // Ensure key is in range for fair comparison
        key = td.data[pos + (key % range_size)];

        auto result = strategy.lower_bound(td.data.data(), pos, pos + range_size, key);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

// Range sizes to test
#define RANGE_SIZES Arg(8)->Arg(16)->Arg(32)->Arg(64)->Arg(128)->Arg(256)

// Register benchmarks

// Scalar baseline (std::lower_bound wrapper)
BENCHMARK(BM_Strategy<simdex::ScalarBaseline<std::uint64_t>>)
    ->Name("Scalar_Baseline")
    ->RANGE_SIZES
    ->Unit(benchmark::kNanosecond);

// Original SIMD linear scan
BENCHMARK(BM_Strategy<simdex::SIMDLinearScan<std::uint64_t>>)
    ->Name("SIMD_Linear_Original")
    ->RANGE_SIZES
    ->Unit(benchmark::kNanosecond);

// Original K-ary search
BENCHMARK(BM_Strategy<simdex::SIMDKarySearch<std::uint64_t>>)
    ->Name("SIMD_Kary_Original")
    ->RANGE_SIZES
    ->Unit(benchmark::kNanosecond);

#if defined(SIMDEX_HAS_NEON)
// Optimized NEON linear scan (multi-vector with deferred extraction)
BENCHMARK(BM_Strategy<simdex::NEONLinearScanOptimized>)
    ->Name("NEON_Linear_Optimized")
    ->RANGE_SIZES
    ->Unit(benchmark::kNanosecond);

// Prefetch-assisted scalar
BENCHMARK(BM_Strategy<simdex::NEONPrefetchScalar>)
    ->Name("NEON_Prefetch_Scalar")
    ->RANGE_SIZES
    ->Unit(benchmark::kNanosecond);
#endif

} // namespace
