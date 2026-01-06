/// @file test_batch_query.cpp
/// @brief Tests for batched query processing

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "simdex/simdex.hpp"

namespace {

class BatchQueryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create sorted test data
        data_.resize(10000);
        std::iota(data_.begin(), data_.end(), 0);

        // Build index
        index_ = std::make_unique<simdex::SIMDPGMIndex<std::uint64_t, 64>>(data_);
    }

    std::vector<std::uint64_t> data_;
    std::unique_ptr<simdex::SIMDPGMIndex<std::uint64_t, 64>> index_;
};

TEST_F(BatchQueryTest, SequentialStrategyFindsAllKeys) {
    std::vector<std::uint64_t> queries = {100, 500, 1000, 5000, 9999};
    auto processor = simdex::make_batch_processor(*index_);

    auto result = processor.process(queries, simdex::BatchStrategy::Sequential);

    ASSERT_EQ(result.positions.size(), queries.size());
    ASSERT_EQ(result.found.size(), queries.size());

    for (std::size_t i = 0; i < queries.size(); ++i) {
        EXPECT_EQ(result.positions[i], queries[i]) << "Query: " << queries[i];
        EXPECT_TRUE(result.found[i]) << "Query: " << queries[i];
    }
}

TEST_F(BatchQueryTest, SortedStrategyFindsAllKeys) {
    std::vector<std::uint64_t> queries = {5000, 100, 9999, 500, 1000};  // Unsorted
    auto processor = simdex::make_batch_processor(*index_);

    auto result = processor.process(queries, simdex::BatchStrategy::SortedPrediction);

    ASSERT_EQ(result.positions.size(), queries.size());

    for (std::size_t i = 0; i < queries.size(); ++i) {
        EXPECT_EQ(result.positions[i], queries[i]) << "Query: " << queries[i];
        EXPECT_TRUE(result.found[i]) << "Query: " << queries[i];
    }
}

TEST_F(BatchQueryTest, CoalescedStrategyFindsAllKeys) {
    std::vector<std::uint64_t> queries = {100, 101, 102, 103, 104};  // Adjacent
    auto processor = simdex::make_batch_processor(*index_);

    auto result = processor.process(queries, simdex::BatchStrategy::SegmentCoalescing);

    ASSERT_EQ(result.positions.size(), queries.size());

    for (std::size_t i = 0; i < queries.size(); ++i) {
        EXPECT_EQ(result.positions[i], queries[i]) << "Query: " << queries[i];
        EXPECT_TRUE(result.found[i]) << "Query: " << queries[i];
    }

    // Adjacent queries should result in cache hits
    EXPECT_GT(result.cache_hits, 0u);
}

TEST_F(BatchQueryTest, FullyOptimizedStrategyFindsAllKeys) {
    std::vector<std::uint64_t> queries = {100, 500, 1000, 5000, 9999};
    auto processor = simdex::make_batch_processor(*index_);

    auto result = processor.process(queries, simdex::BatchStrategy::FullyOptimized);

    ASSERT_EQ(result.positions.size(), queries.size());

    for (std::size_t i = 0; i < queries.size(); ++i) {
        EXPECT_EQ(result.positions[i], queries[i]) << "Query: " << queries[i];
        EXPECT_TRUE(result.found[i]) << "Query: " << queries[i];
    }
}

TEST_F(BatchQueryTest, HandlesNonExistentKeys) {
    std::vector<std::uint64_t> queries = {50000, 100000};  // Beyond data range
    auto processor = simdex::make_batch_processor(*index_);

    auto result = processor.process(queries, simdex::BatchStrategy::FullyOptimized);

    ASSERT_EQ(result.positions.size(), queries.size());
    EXPECT_FALSE(result.found[0]);
    EXPECT_FALSE(result.found[1]);
}

TEST_F(BatchQueryTest, HandlesMixedHitsAndMisses) {
    std::vector<std::uint64_t> queries = {100, 99999, 500, 88888};  // Mix
    auto processor = simdex::make_batch_processor(*index_);

    auto result = processor.process(queries, simdex::BatchStrategy::FullyOptimized);

    EXPECT_TRUE(result.found[0]);   // 100 exists
    EXPECT_FALSE(result.found[1]);  // 99999 doesn't
    EXPECT_TRUE(result.found[2]);   // 500 exists
    EXPECT_FALSE(result.found[3]);  // 88888 doesn't
}

TEST_F(BatchQueryTest, HandlesEmptyBatch) {
    std::vector<std::uint64_t> queries = {};
    auto processor = simdex::make_batch_processor(*index_);

    auto result = processor.process(queries, simdex::BatchStrategy::FullyOptimized);

    EXPECT_TRUE(result.positions.empty());
    EXPECT_TRUE(result.found.empty());
}

TEST_F(BatchQueryTest, HandlesSingleQuery) {
    std::vector<std::uint64_t> queries = {500};
    auto processor = simdex::make_batch_processor(*index_);

    auto result = processor.process(queries, simdex::BatchStrategy::FullyOptimized);

    ASSERT_EQ(result.positions.size(), 1u);
    EXPECT_EQ(result.positions[0], 500u);
    EXPECT_TRUE(result.found[0]);
}

TEST_F(BatchQueryTest, LargeBatchCorrectness) {
    // Large batch with random queries
    std::mt19937 rng(42);
    std::uniform_int_distribution<std::uint64_t> dist(0, data_.size() - 1);

    std::vector<std::uint64_t> queries(1000);
    for (auto& q : queries) {
        q = data_[dist(rng)];  // All should be found
    }

    auto processor = simdex::make_batch_processor(*index_);
    auto result = processor.process(queries, simdex::BatchStrategy::FullyOptimized);

    ASSERT_EQ(result.positions.size(), queries.size());

    for (std::size_t i = 0; i < queries.size(); ++i) {
        EXPECT_EQ(result.positions[i], queries[i]) << "Query: " << queries[i];
        EXPECT_TRUE(result.found[i]) << "Query: " << queries[i];
    }
}

TEST_F(BatchQueryTest, AllStrategiesAreEquivalent) {
    std::vector<std::uint64_t> queries = {100, 500, 1000, 2500, 5000, 7500, 9000};
    auto processor = simdex::make_batch_processor(*index_);

    auto sequential = processor.process(queries, simdex::BatchStrategy::Sequential);
    auto sorted = processor.process(queries, simdex::BatchStrategy::SortedPrediction);
    auto coalesced = processor.process(queries, simdex::BatchStrategy::SegmentCoalescing);
    auto prefetched = processor.process(queries, simdex::BatchStrategy::Prefetched);
    auto optimized = processor.process(queries, simdex::BatchStrategy::FullyOptimized);

    // All strategies should return the same results
    EXPECT_EQ(sequential.positions, sorted.positions);
    EXPECT_EQ(sorted.positions, coalesced.positions);
    EXPECT_EQ(coalesced.positions, prefetched.positions);
    EXPECT_EQ(prefetched.positions, optimized.positions);

    EXPECT_EQ(sequential.found, sorted.found);
    EXPECT_EQ(sorted.found, coalesced.found);
    EXPECT_EQ(coalesced.found, prefetched.found);
    EXPECT_EQ(prefetched.found, optimized.found);
}

TEST(SegmentCoalescerTest, CoalescesOverlappingRanges) {
    using Query = simdex::PredictedQuery<std::uint64_t>;

    std::vector<Query> queries = {
        {100, 0, 100, 90, 110},   // Range [90, 110)
        {105, 1, 105, 95, 115},   // Range [95, 115) - overlaps
        {200, 2, 200, 190, 210},  // Range [190, 210) - separate
    };

    auto groups = simdex::SegmentCoalescer<std::uint64_t>::coalesce(queries, 0);

    ASSERT_EQ(groups.size(), 2u);

    // First group should merge queries 0 and 1
    EXPECT_EQ(groups[0].merged_lo, 90u);
    EXPECT_EQ(groups[0].merged_hi, 115u);
    EXPECT_EQ(groups[0].query_indices.size(), 2u);

    // Second group is query 2 alone
    EXPECT_EQ(groups[1].merged_lo, 190u);
    EXPECT_EQ(groups[1].merged_hi, 210u);
    EXPECT_EQ(groups[1].query_indices.size(), 1u);
}

TEST(SegmentCoalescerTest, CoalescesWithGap) {
    using Query = simdex::PredictedQuery<std::uint64_t>;

    std::vector<Query> queries = {
        {100, 0, 100, 90, 110},
        {120, 1, 120, 115, 125},  // Gap of 5 between ranges
    };

    // With max_gap = 0, should not merge
    auto groups_no_gap = simdex::SegmentCoalescer<std::uint64_t>::coalesce(queries, 0);
    EXPECT_EQ(groups_no_gap.size(), 2u);

    // With max_gap = 10, should merge
    auto groups_with_gap = simdex::SegmentCoalescer<std::uint64_t>::coalesce(queries, 10);
    EXPECT_EQ(groups_with_gap.size(), 1u);
}

} // namespace
