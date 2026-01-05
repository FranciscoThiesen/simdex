/// @file pgm_example.cpp
/// @brief Example demonstrating SIMDPGMIndex usage

#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "simdex/simdex.hpp"

int main() {
    std::cout << "SIMDEX PGM-Index Demo\n";
    std::cout << "Platform: " << simdex::platform_info() << "\n\n";

    // Create sorted test data with gaps (more realistic)
    constexpr std::size_t data_size = 1'000'000;
    std::vector<std::uint64_t> data(data_size);

    // Generate data with random gaps (harder for learned model)
    std::mt19937_64 data_rng(12345);
    std::uint64_t val = 0;
    for (auto& d : data) {
        val += 1 + (data_rng() % 100);  // Random gaps 1-100
        d = val;
    }

    // RNG for sampling
    std::mt19937_64 sample_rng(999);

    // Test different epsilon values
    auto test_epsilon = [&](auto& idx, const char* name) {
        std::cout << "\n=== " << name << " ===\n";
        std::cout << "  - Epsilon: " << idx.epsilon() << "\n";
        std::cout << "  - Segments: " << idx.segment_count() << "\n";

        // Check average range size
        std::size_t total_range = 0;
        for (int i = 0; i < 1000; ++i) {
            auto approx = idx.get_approx_pos(data[sample_rng() % data.size()]);
            total_range += approx.hi - approx.lo;
        }
        std::cout << "  - Avg range size: " << total_range / 1000 << "\n";
    };

    // Create indices with different epsilon values
    std::cout << "Building indices for " << data_size << " elements...\n";
    simdex::SIMDPGMIndex<std::uint64_t, 64> index64(data);
    simdex::SIMDPGMIndex<std::uint64_t, 256> index256(data);

    test_epsilon(index64, "Epsilon=64");
    test_epsilon(index256, "Epsilon=256");

    // Generate random queries (mix of hits and misses)
    constexpr std::size_t num_queries = 100'000;
    std::vector<std::uint64_t> queries(num_queries);
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<std::size_t> idx_dist(0, data_size - 1);
    for (auto& q : queries) {
        // 80% existing keys, 20% random (may not exist)
        if (rng() % 5 != 0) {
            q = data[idx_dist(rng)];  // Existing key
        } else {
            q = rng() % data.back();  // Random key
        }
    }

    // Benchmark different strategies on an index
    auto run_benchmark = [&queries, num_queries](auto& idx, const char* idx_name) {
        auto benchmark = [&](simdex::StrategyPolicy policy, const char* name) {
            using Clock = std::chrono::high_resolution_clock;

            auto start = Clock::now();

            std::size_t found = 0;
            for (auto key : queries) {
                auto it = idx.lower_bound(key, policy);
                if (it != idx.end() && *it == key) {
                    ++found;
                }
            }

            auto end = Clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            double ns_per_query = static_cast<double>(duration.count()) / num_queries;

            std::cout << "  " << name << ": " << ns_per_query << " ns/query\n";
        };

        std::cout << "\n" << idx_name << " (" << num_queries << " queries):\n";
        benchmark(simdex::StrategyPolicy::AlwaysScalar, "Scalar (std::lower_bound)");
        benchmark(simdex::StrategyPolicy::AlwaysLinear, "SIMD Linear Scan      ");
        benchmark(simdex::StrategyPolicy::AlwaysKary,   "SIMD K-ary Search     ");
        benchmark(simdex::StrategyPolicy::Auto,         "Auto (adaptive)       ");
    };

    run_benchmark(index64, "Epsilon=64 (range ~130)");
    run_benchmark(index256, "Epsilon=256 (range ~514)");

    std::cout << "\nDone!\n";
    return 0;
}
