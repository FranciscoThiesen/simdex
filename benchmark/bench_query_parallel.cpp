/// @file bench_query_parallel.cpp
/// @brief Benchmarks for query-parallel SIMD search
///
/// This is the hail-mary: using SIMD gather to process multiple queries
/// in parallel, rather than parallelizing within a single query.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "simdex/simdex.hpp"
#include "simdex/search/parallel_queries.hpp"

namespace {

// ============================================================================
// Data generation
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

std::vector<std::uint64_t> generate_queries(
    const std::vector<std::uint64_t>& data,
    std::size_t count,
    std::uint64_t seed = 123
) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t> dist(0, data.size() - 1);
    std::vector<std::uint64_t> queries(count);
    for (auto& q : queries) {
        q = data[dist(rng)];
    }
    return queries;
}

// Overload for pointer-based data (used with index.data())
std::vector<std::uint64_t> generate_queries(
    const std::uint64_t* begin,
    const std::uint64_t* end,
    std::size_t count
) {
    std::size_t size = static_cast<std::size_t>(end - begin);
    std::mt19937_64 rng(999);
    std::uniform_int_distribution<std::size_t> dist(0, size - 1);
    std::vector<std::uint64_t> queries(count);
    for (auto& q : queries) {
        q = begin[dist(rng)];
    }
    return queries;
}

// ============================================================================
// Baseline: Sequential single-query processing
// ============================================================================

static void BM_Sequential(benchmark::State& state) {
    const std::size_t data_size = 1'000'000;
    const std::size_t batch_size = static_cast<std::size_t>(state.range(0));

    auto data = generate_sorted_data(data_size);
    auto queries = generate_queries(data, batch_size);

    simdex::SIMDPGMIndex<std::uint64_t, 64> index(std::move(data));

    for (auto _ : state) {
        std::size_t sum = 0;
        for (const auto& key : queries) {
            auto it = index.lower_bound(key);
            sum += static_cast<std::size_t>(it - index.begin());
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * batch_size));
    state.SetLabel("sequential");
}

// ============================================================================
// Vanilla PGM with std::lower_bound
// ============================================================================

static void BM_VanillaPGM(benchmark::State& state) {
    const std::size_t data_size = 1'000'000;
    const std::size_t batch_size = static_cast<std::size_t>(state.range(0));

    auto data = generate_sorted_data(data_size);
    auto queries = generate_queries(data, batch_size);

    pgm::PGMIndex<std::uint64_t, 64> index(data.begin(), data.end());

    for (auto _ : state) {
        std::size_t sum = 0;
        for (const auto& key : queries) {
            auto approx = index.search(key);
            auto lo = data.begin() + static_cast<std::ptrdiff_t>(approx.lo);
            auto hi = data.begin() + static_cast<std::ptrdiff_t>(std::min(approx.hi, data.size()));
            auto it = std::lower_bound(lo, hi, key);
            sum += (it - data.begin());
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * batch_size));
    state.SetLabel("vanilla_pgm");
}

// ============================================================================
// Query-parallel SIMD processing
// ============================================================================

static void BM_QueryParallel(benchmark::State& state) {
    const std::size_t data_size = 1'000'000;
    const std::size_t batch_size = static_cast<std::size_t>(state.range(0));

    auto data = generate_sorted_data(data_size);
    auto queries = generate_queries(data, batch_size);

    simdex::SIMDPGMIndex<std::uint64_t, 64> index(std::move(data));
    auto processor = simdex::make_query_parallel_processor(index);

    for (auto _ : state) {
        auto results = processor.process(queries);
        benchmark::DoNotOptimize(results.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * batch_size));
    state.SetLabel("query_parallel");
}

// ============================================================================
// Direct comparison: 4 queries at a time
// ============================================================================

static void BM_Direct_Sequential_4(benchmark::State& state) {
    const std::size_t data_size = 1'000'000;

    auto data = generate_sorted_data(data_size);
    simdex::SIMDPGMIndex<std::uint64_t, 64> index(std::move(data));

    // Generate exactly 4 queries
    std::vector<std::uint64_t> queries = generate_queries(index.data(), index.data() + index.size(), 4);

    for (auto _ : state) {
        std::size_t sum = 0;
        for (const auto& key : queries) {
            auto it = index.lower_bound(key);
            sum += static_cast<std::size_t>(it - index.begin());
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * 4));
    state.SetLabel("seq_4");
}

static void BM_Direct_QueryParallel_4(benchmark::State& state) {
    const std::size_t data_size = 1'000'000;

    auto data = generate_sorted_data(data_size);
    simdex::SIMDPGMIndex<std::uint64_t, 64> index(std::move(data));

    std::vector<std::uint64_t> queries = generate_queries(index.data(), index.data() + index.size(), 4);

    // Pre-compute bounds
    std::vector<std::size_t> lo(4), hi(4), results(4);
    for (int i = 0; i < 4; i++) {
        auto approx = index.get_approx_pos(queries[i]);
        lo[i] = approx.lo;
        hi[i] = std::min(approx.hi, index.size());
    }

    for (auto _ : state) {
#if defined(SIMDEX_HAS_AVX2)
        simdex::parallel_lower_bound_4x64(
            index.data(), queries.data(), lo.data(), hi.data(), results.data()
        );
#else
        simdex::parallel_lower_bound_scalar(
            index.data(), queries.data(), lo.data(), hi.data(), results.data(), 4
        );
#endif
        benchmark::DoNotOptimize(results[0]);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * 4));
    state.SetLabel("parallel_4");
}

// ============================================================================
// Micro-benchmark: just the SIMD gather vs sequential loads
// ============================================================================

static void BM_Micro_SequentialBinarySearch(benchmark::State& state) {
    const std::size_t data_size = 1'000'000;
    auto data = generate_sorted_data(data_size);

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<std::size_t> dist(0, data_size - 200);

    for (auto _ : state) {
        // Random range of ~128 elements
        std::size_t lo = dist(rng);
        std::size_t hi = lo + 128;
        std::uint64_t key = data[lo + 64];  // Key is in the middle

        // Standard binary search
        std::size_t result = lo;
        while (lo < hi) {
            std::size_t mid = lo + (hi - lo) / 2;
            if (data[mid] < key) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        result = lo;
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations());
    state.SetLabel("micro_seq_binary");
}

#if defined(SIMDEX_HAS_AVX2)
static void BM_Micro_ParallelBinarySearch4(benchmark::State& state) {
    const std::size_t data_size = 1'000'000;
    auto data = generate_sorted_data(data_size);

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<std::size_t> dist(0, data_size - 200);

    std::vector<std::uint64_t> keys(4);
    std::vector<std::size_t> lo(4), hi(4), results(4);

    for (auto _ : state) {
        // Set up 4 searches
        for (int i = 0; i < 4; i++) {
            lo[i] = dist(rng);
            hi[i] = lo[i] + 128;
            keys[i] = data[lo[i] + 64];
        }

        simdex::parallel_lower_bound_4x64(
            data.data(), keys.data(), lo.data(), hi.data(), results.data()
        );
        benchmark::DoNotOptimize(results[0]);
    }

    // We processed 4 queries per iteration
    state.SetItemsProcessed(state.iterations() * 4);
    state.SetLabel("micro_par_binary_4");
}
#endif

// ============================================================================
// Throughput comparison at various batch sizes
// ============================================================================

static void BM_Throughput_Sequential(benchmark::State& state) {
    const std::size_t data_size = 1'000'000;
    const std::size_t batch_size = static_cast<std::size_t>(state.range(0));

    auto data = generate_sorted_data(data_size);
    auto queries = generate_queries(data, batch_size);
    simdex::SIMDPGMIndex<std::uint64_t, 64> index(std::move(data));

    for (auto _ : state) {
        std::size_t sum = 0;
        for (const auto& key : queries) {
            auto it = index.lower_bound(key);
            sum += static_cast<std::size_t>(it - index.begin());
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * batch_size));
}

static void BM_Throughput_QueryParallel(benchmark::State& state) {
    const std::size_t data_size = 1'000'000;
    const std::size_t batch_size = static_cast<std::size_t>(state.range(0));

    auto data = generate_sorted_data(data_size);
    auto queries = generate_queries(data, batch_size);
    simdex::SIMDPGMIndex<std::uint64_t, 64> index(std::move(data));
    auto processor = simdex::make_query_parallel_processor(index);

    for (auto _ : state) {
        auto results = processor.process(queries);
        benchmark::DoNotOptimize(results.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * batch_size));
}

// ============================================================================
// Register benchmarks
// ============================================================================

// Main comparison
BENCHMARK(BM_Sequential)->Arg(4)->Arg(16)->Arg(64)->Arg(256)->Arg(1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_VanillaPGM)->Arg(4)->Arg(16)->Arg(64)->Arg(256)->Arg(1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_QueryParallel)->Arg(4)->Arg(16)->Arg(64)->Arg(256)->Arg(1024)->Unit(benchmark::kMicrosecond);

// Direct 4-query comparison
BENCHMARK(BM_Direct_Sequential_4)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Direct_QueryParallel_4)->Unit(benchmark::kNanosecond);

// Micro benchmarks
BENCHMARK(BM_Micro_SequentialBinarySearch)->Unit(benchmark::kNanosecond);
#if defined(SIMDEX_HAS_AVX2)
BENCHMARK(BM_Micro_ParallelBinarySearch4)->Unit(benchmark::kNanosecond);
#endif

// Throughput scaling
BENCHMARK(BM_Throughput_Sequential)->RangeMultiplier(4)->Range(4, 4096)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Throughput_QueryParallel)->RangeMultiplier(4)->Range(4, 4096)->Unit(benchmark::kMicrosecond);

} // namespace
