/// @file bench_crossover.cpp
/// @brief Find the crossover points where SIMD beats scalar
///
/// This benchmark measures performance at fine-grained range sizes
/// to determine exactly where each strategy wins.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "simdex/simdex.hpp"

namespace {

// Pre-generate data and queries to avoid setup overhead
struct TestData {
    std::vector<std::uint64_t> data;
    std::vector<std::uint64_t> queries;
    std::vector<std::size_t> positions;  // Random positions for range start

    TestData() : data(100000), queries(10000), positions(10000) {
        std::iota(data.begin(), data.end(), 0);

        std::mt19937_64 rng(42);
        for (auto& q : queries) {
            q = rng() % data.size();
        }
        for (auto& p : positions) {
            p = rng() % (data.size() - 1000);  // Leave room for range
        }
    }
};

static TestData& get_test_data() {
    static TestData data;
    return data;
}

// Benchmark a specific strategy at a specific range size
template<typename Strategy>
static void BM_Strategy_RangeSize(benchmark::State& state) {
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

// Range sizes to test: powers of 2 from 4 to 512
#define RANGE_SIZES Arg(4)->Arg(8)->Arg(16)->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Arg(512)

// Register benchmarks for each strategy
BENCHMARK(BM_Strategy_RangeSize<simdex::ScalarBaseline<std::uint64_t>>)
    ->Name("Scalar")
    ->RANGE_SIZES
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Strategy_RangeSize<simdex::SIMDLinearScan<std::uint64_t>>)
    ->Name("SIMD_Linear")
    ->RANGE_SIZES
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Strategy_RangeSize<simdex::SIMDKarySearch<std::uint64_t>>)
    ->Name("SIMD_Kary")
    ->RANGE_SIZES
    ->Unit(benchmark::kNanosecond);

// Summary benchmark showing which strategy wins at each range size
static void BM_BestStrategy(benchmark::State& state) {
    auto& td = get_test_data();
    const std::size_t range_size = static_cast<std::size_t>(state.range(0));

    simdex::ScalarBaseline<std::uint64_t> scalar;
    simdex::SIMDLinearScan<std::uint64_t> linear;
    simdex::SIMDKarySearch<std::uint64_t> kary;

    // Warm up and determine best
    std::size_t idx = 0;
    auto test_strategy = [&](auto& strategy) {
        volatile const std::uint64_t* sink;
        for (int i = 0; i < 1000; ++i) {
            std::size_t pos = td.positions[idx % td.positions.size()];
            std::uint64_t key = td.data[pos + (td.queries[idx % td.queries.size()] % range_size)];
            ++idx;
            sink = strategy.lower_bound(td.data.data(), pos, pos + range_size, key);
        }
        return sink;
    };

    test_strategy(scalar);
    test_strategy(linear);
    test_strategy(kary);

    // Actual benchmark with auto-selection
    idx = 0;
    for (auto _ : state) {
        std::size_t pos = td.positions[idx % td.positions.size()];
        std::uint64_t key = td.data[pos + (td.queries[idx % td.queries.size()] % range_size)];
        ++idx;

        const std::uint64_t* result;

        // Platform-aware selection
#if defined(SIMDEX_HAS_AVX2)
        if (range_size <= 32) {
            result = linear.lower_bound(td.data.data(), pos, pos + range_size, key);
        } else if (range_size <= 128) {
            result = kary.lower_bound(td.data.data(), pos, pos + range_size, key);
        } else {
            result = scalar.lower_bound(td.data.data(), pos, pos + range_size, key);
        }
#elif defined(SIMDEX_HAS_NEON)
        if (range_size <= 8) {
            result = linear.lower_bound(td.data.data(), pos, pos + range_size, key);
        } else {
            result = scalar.lower_bound(td.data.data(), pos, pos + range_size, key);
        }
#else
        result = scalar.lower_bound(td.data.data(), pos, pos + range_size, key);
#endif

        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

BENCHMARK(BM_BestStrategy)
    ->Name("Auto_Select")
    ->RANGE_SIZES
    ->Unit(benchmark::kNanosecond);

} // namespace
