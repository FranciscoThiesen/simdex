/// @file basic_usage.cpp
/// @brief Simple example demonstrating simdex usage

#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "simdex/simdex.hpp"

template<typename Strategy>
void benchmark_strategy(const Strategy& strategy,
                        const std::vector<std::uint64_t>& data,
                        const std::vector<std::uint64_t>& queries,
                        std::size_t range_size) {
    using Clock = std::chrono::high_resolution_clock;

    auto start = Clock::now();

    volatile const std::uint64_t* result_sink;
    for (auto key : queries) {
        std::size_t key_pos = static_cast<std::size_t>(key);
        std::size_t approx_pos = std::min(key_pos, data.size() - range_size);
        std::size_t lo = approx_pos;
        std::size_t hi = std::min(approx_pos + range_size, data.size());

        result_sink = strategy.lower_bound(data.data(), lo, hi, key);
    }
    (void)result_sink;

    auto end = Clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double ns_per_query = static_cast<double>(duration.count()) / queries.size();

    std::cout << "  " << Strategy::name() << ": "
              << ns_per_query << " ns/query\n";
}

int main() {
    std::cout << "SIMDEX v" << simdex::version() << "\n";
    std::cout << "Platform: " << simdex::platform_info() << "\n\n";

    // Create sorted test data
    constexpr std::size_t data_size = 1'000'000;
    std::vector<std::uint64_t> data(data_size);
    std::iota(data.begin(), data.end(), 0);

    // Create random queries
    constexpr std::size_t num_queries = 100'000;
    std::vector<std::uint64_t> queries(num_queries);
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<std::uint64_t> dist(0, data_size - 1);
    for (auto& q : queries) {
        q = dist(rng);
    }

    // Create search strategies
    simdex::ScalarBaseline<std::uint64_t> scalar;
    simdex::SIMDLinearScan<std::uint64_t> linear;
    simdex::SIMDKarySearch<std::uint64_t> kary;

    // Benchmark with different range sizes
    for (std::size_t range_size : {16, 32, 64, 128, 256}) {
        std::cout << "\nRange size: " << range_size << "\n";
        benchmark_strategy(scalar, data, queries, range_size);
        benchmark_strategy(linear, data, queries, range_size);
        benchmark_strategy(kary, data, queries, range_size);
    }

    std::cout << "\nDone!\n";
    return 0;
}
