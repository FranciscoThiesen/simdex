/// @file test_parallel_queries.cpp
/// @brief Tests for query-parallel SIMD search

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "simdex/simdex.hpp"
#include "simdex/search/parallel_queries.hpp"

namespace {

class QueryParallelTest : public ::testing::Test {
protected:
    void SetUp() override {
        data_.resize(10000);
        std::iota(data_.begin(), data_.end(), 0);
        index_ = std::make_unique<simdex::SIMDPGMIndex<std::uint64_t, 64>>(data_);
    }

    std::vector<std::uint64_t> data_;
    std::unique_ptr<simdex::SIMDPGMIndex<std::uint64_t, 64>> index_;
};

TEST_F(QueryParallelTest, ScalarFallbackCorrectness) {
    std::vector<std::uint64_t> keys = {100, 500, 1000, 5000};
    std::vector<std::size_t> lo = {0, 400, 900, 4900};
    std::vector<std::size_t> hi = {200, 600, 1100, 5100};
    std::vector<std::size_t> results(4);

    simdex::parallel_lower_bound_scalar(
        data_.data(), keys.data(), lo.data(), hi.data(), results.data(), 4
    );

    EXPECT_EQ(results[0], 100u);
    EXPECT_EQ(results[1], 500u);
    EXPECT_EQ(results[2], 1000u);
    EXPECT_EQ(results[3], 5000u);
}

#if defined(SIMDEX_HAS_AVX2)
TEST_F(QueryParallelTest, AVX2_4x64_Correctness) {
    std::vector<std::uint64_t> keys = {100, 500, 1000, 5000};
    std::vector<std::size_t> lo = {0, 400, 900, 4900};
    std::vector<std::size_t> hi = {200, 600, 1100, 5100};
    std::vector<std::size_t> results(4);

    simdex::parallel_lower_bound_4x64(
        data_.data(), keys.data(), lo.data(), hi.data(), results.data()
    );

    EXPECT_EQ(results[0], 100u);
    EXPECT_EQ(results[1], 500u);
    EXPECT_EQ(results[2], 1000u);
    EXPECT_EQ(results[3], 5000u);
}

TEST_F(QueryParallelTest, AVX2_MatchesScalar) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<std::size_t> dist(0, data_.size() - 200);

    for (int trial = 0; trial < 100; ++trial) {
        std::vector<std::uint64_t> keys(4);
        std::vector<std::size_t> lo(4), hi(4);
        std::vector<std::size_t> results_simd(4), results_scalar(4);

        for (int i = 0; i < 4; ++i) {
            lo[i] = dist(rng);
            hi[i] = lo[i] + 128;
            std::uniform_int_distribution<std::size_t> key_dist(lo[i], hi[i] - 1);
            keys[i] = data_[key_dist(rng)];
        }

        simdex::parallel_lower_bound_4x64(
            data_.data(), keys.data(), lo.data(), hi.data(), results_simd.data()
        );
        simdex::parallel_lower_bound_scalar(
            data_.data(), keys.data(), lo.data(), hi.data(), results_scalar.data(), 4
        );

        for (int i = 0; i < 4; ++i) {
            EXPECT_EQ(results_simd[i], results_scalar[i])
                << "Mismatch at trial " << trial << ", query " << i
                << ": key=" << keys[i] << ", lo=" << lo[i] << ", hi=" << hi[i];
        }
    }
}
#endif

TEST_F(QueryParallelTest, ProcessorCorrectness) {
    std::vector<std::uint64_t> queries = {100, 500, 1000, 5000, 9999};
    auto processor = simdex::make_query_parallel_processor(*index_);

    auto results = processor.process(queries);

    ASSERT_EQ(results.size(), queries.size());
    for (std::size_t i = 0; i < queries.size(); ++i) {
        EXPECT_EQ(results[i], queries[i]) << "Query: " << queries[i];
        EXPECT_EQ(data_[results[i]], queries[i]) << "Query: " << queries[i];
    }
}

TEST_F(QueryParallelTest, ProcessorWithFlags) {
    std::vector<std::uint64_t> queries = {100, 99999, 500, 88888};  // Mix of hits and misses
    auto processor = simdex::make_query_parallel_processor(*index_);

    auto [positions, found] = processor.process_with_flags(queries);

    EXPECT_TRUE(found[0]);   // 100 exists
    EXPECT_FALSE(found[1]);  // 99999 doesn't
    EXPECT_TRUE(found[2]);   // 500 exists
    EXPECT_FALSE(found[3]);  // 88888 doesn't
}

TEST_F(QueryParallelTest, LargeBatchCorrectness) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<std::size_t> dist(0, data_.size() - 1);

    std::vector<std::uint64_t> queries(1000);
    for (auto& q : queries) {
        q = data_[dist(rng)];
    }

    auto processor = simdex::make_query_parallel_processor(*index_);
    auto results = processor.process(queries);

    ASSERT_EQ(results.size(), queries.size());
    for (std::size_t i = 0; i < queries.size(); ++i) {
        EXPECT_EQ(data_[results[i]], queries[i]) << "Query index: " << i;
    }
}

TEST_F(QueryParallelTest, EmptyBatch) {
    std::vector<std::uint64_t> queries = {};
    auto processor = simdex::make_query_parallel_processor(*index_);

    auto results = processor.process(queries);

    EXPECT_TRUE(results.empty());
}

TEST_F(QueryParallelTest, NonMultipleOf4Batch) {
    // Test batch sizes that aren't multiples of 4 (SIMD width)
    for (std::size_t size : {1, 2, 3, 5, 7, 13, 17}) {
        std::vector<std::uint64_t> queries(size);
        for (std::size_t i = 0; i < size; ++i) {
            queries[i] = i * 100;
        }

        auto processor = simdex::make_query_parallel_processor(*index_);
        auto results = processor.process(queries);

        ASSERT_EQ(results.size(), size);
        for (std::size_t i = 0; i < size; ++i) {
            EXPECT_EQ(results[i], queries[i]) << "Size=" << size << ", i=" << i;
        }
    }
}

} // namespace
