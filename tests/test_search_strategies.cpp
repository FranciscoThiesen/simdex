/// @file test_search_strategies.cpp
/// @brief Comprehensive tests for all search strategies
///
/// Tests verify that all SIMD search strategies produce identical results
/// to the scalar baseline (std::lower_bound).

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "simdex/simdex.hpp"

namespace {

// ============================================================================
// Test fixture with common test data
// ============================================================================

class SearchStrategyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Contiguous data (no gaps)
        contiguous_16_.resize(16);
        std::iota(contiguous_16_.begin(), contiguous_16_.end(), 0);

        contiguous_64_.resize(64);
        std::iota(contiguous_64_.begin(), contiguous_64_.end(), 0);

        contiguous_256_.resize(256);
        std::iota(contiguous_256_.begin(), contiguous_256_.end(), 0);

        contiguous_1000_.resize(1000);
        std::iota(contiguous_1000_.begin(), contiguous_1000_.end(), 0);

        // Data with gaps
        sparse_ = {1, 5, 10, 15, 20, 50, 100, 200, 500, 1000, 2000, 5000};

        // Random data (sorted)
        std::mt19937_64 rng(42);
        std::uniform_int_distribution<std::uint64_t> dist(0, 1'000'000);
        random_1000_.resize(1000);
        for (auto& v : random_1000_) {
            v = dist(rng);
        }
        std::sort(random_1000_.begin(), random_1000_.end());
        // Remove duplicates
        random_1000_.erase(
            std::unique(random_1000_.begin(), random_1000_.end()),
            random_1000_.end());
    }

    std::vector<std::uint64_t> contiguous_16_;
    std::vector<std::uint64_t> contiguous_64_;
    std::vector<std::uint64_t> contiguous_256_;
    std::vector<std::uint64_t> contiguous_1000_;
    std::vector<std::uint64_t> sparse_;
    std::vector<std::uint64_t> random_1000_;

    // Strategies to test
    simdex::ScalarBaseline<std::uint64_t> scalar_;
    simdex::SIMDLinearScan<std::uint64_t> linear_;
    simdex::SIMDKarySearch<std::uint64_t> kary_;
};

// ============================================================================
// Helper function to compare strategy against std::lower_bound
// ============================================================================

template<typename Strategy>
void verify_against_std(const Strategy& strategy,
                        const std::vector<std::uint64_t>& data,
                        const std::vector<std::uint64_t>& queries) {
    for (auto key : queries) {
        auto expected = std::lower_bound(data.begin(), data.end(), key);
        auto actual = strategy.lower_bound(data.data(), 0, data.size(), key);

        if (expected == data.end()) {
            EXPECT_EQ(actual, data.data() + data.size())
                << "Key: " << key << ", Strategy: " << Strategy::name();
        } else {
            EXPECT_EQ(actual, &*expected)
                << "Key: " << key << ", Strategy: " << Strategy::name()
                << ", Expected value: " << *expected
                << ", Actual value: " << *actual;
        }
    }
}

template<typename Strategy>
void verify_subrange(const Strategy& strategy,
                     const std::vector<std::uint64_t>& data,
                     std::size_t lo, std::size_t hi,
                     const std::vector<std::uint64_t>& queries) {
    for (auto key : queries) {
        auto expected = std::lower_bound(data.begin() + lo,
                                         data.begin() + hi, key);
        auto actual = strategy.lower_bound(data.data(), lo, hi, key);

        if (expected == data.begin() + hi) {
            EXPECT_EQ(actual, data.data() + hi)
                << "Key: " << key << ", Range: [" << lo << ", " << hi << ")";
        } else {
            EXPECT_EQ(actual, &*expected)
                << "Key: " << key << ", Range: [" << lo << ", " << hi << ")";
        }
    }
}

// ============================================================================
// SIMD Linear Scan Tests
// ============================================================================

TEST_F(SearchStrategyTest, LinearScan_SmallContiguous) {
    std::vector<std::uint64_t> queries = {0, 1, 7, 8, 15, 16, 100};
    verify_against_std(linear_, contiguous_16_, queries);
}

TEST_F(SearchStrategyTest, LinearScan_MediumContiguous) {
    std::vector<std::uint64_t> queries;
    for (std::uint64_t i = 0; i <= 70; i += 5) {
        queries.push_back(i);
    }
    verify_against_std(linear_, contiguous_64_, queries);
}

TEST_F(SearchStrategyTest, LinearScan_Sparse) {
    std::vector<std::uint64_t> queries = {0, 1, 2, 5, 7, 10, 12, 100, 150, 500, 1000, 3000, 10000};
    verify_against_std(linear_, sparse_, queries);
}

TEST_F(SearchStrategyTest, LinearScan_Random) {
    // Test with actual values from the array
    verify_against_std(linear_, random_1000_, random_1000_);

    // Test with random queries (some may not exist)
    std::mt19937_64 rng(123);
    std::uniform_int_distribution<std::uint64_t> dist(0, 1'500'000);
    std::vector<std::uint64_t> queries(500);
    for (auto& q : queries) {
        q = dist(rng);
    }
    verify_against_std(linear_, random_1000_, queries);
}

TEST_F(SearchStrategyTest, LinearScan_Subrange) {
    std::vector<std::uint64_t> queries = {10, 20, 30, 40, 45, 50, 60};
    verify_subrange(linear_, contiguous_256_, 10, 50, queries);
}

TEST_F(SearchStrategyTest, LinearScan_EmptyRange) {
    auto result = linear_.lower_bound(contiguous_16_.data(), 5, 5, 10);
    EXPECT_EQ(result, contiguous_16_.data() + 5);
}

TEST_F(SearchStrategyTest, LinearScan_SingleElement) {
    auto result = linear_.lower_bound(contiguous_16_.data(), 5, 6, 5);
    EXPECT_EQ(result, contiguous_16_.data() + 5);
    EXPECT_EQ(*result, 5);
}

// ============================================================================
// SIMD K-ary Search Tests
// ============================================================================

TEST_F(SearchStrategyTest, KarySearch_SmallContiguous) {
    std::vector<std::uint64_t> queries = {0, 1, 7, 8, 15, 16, 100};
    verify_against_std(kary_, contiguous_16_, queries);
}

TEST_F(SearchStrategyTest, KarySearch_MediumContiguous) {
    std::vector<std::uint64_t> queries;
    for (std::uint64_t i = 0; i <= 70; i += 5) {
        queries.push_back(i);
    }
    verify_against_std(kary_, contiguous_64_, queries);
}

TEST_F(SearchStrategyTest, KarySearch_LargeContiguous) {
    std::vector<std::uint64_t> queries;
    for (std::uint64_t i = 0; i <= 260; i += 10) {
        queries.push_back(i);
    }
    verify_against_std(kary_, contiguous_256_, queries);
}

TEST_F(SearchStrategyTest, KarySearch_Sparse) {
    std::vector<std::uint64_t> queries = {0, 1, 2, 5, 7, 10, 12, 100, 150, 500, 1000, 3000, 10000};
    verify_against_std(kary_, sparse_, queries);
}

TEST_F(SearchStrategyTest, KarySearch_Random) {
    verify_against_std(kary_, random_1000_, random_1000_);

    std::mt19937_64 rng(456);
    std::uniform_int_distribution<std::uint64_t> dist(0, 1'500'000);
    std::vector<std::uint64_t> queries(500);
    for (auto& q : queries) {
        q = dist(rng);
    }
    verify_against_std(kary_, random_1000_, queries);
}

TEST_F(SearchStrategyTest, KarySearch_Subrange) {
    std::vector<std::uint64_t> queries = {100, 150, 200, 220, 250, 300};
    verify_subrange(kary_, contiguous_1000_, 100, 250, queries);
}

// ============================================================================
// Eytzinger Layout Tests
// ============================================================================

TEST(EytzingerLayoutTest, TransformSmall) {
    std::vector<std::uint64_t> sorted = {1, 2, 3, 4, 5, 6, 7};
    auto eytz = simdex::EytzingerLayout<std::uint64_t>::transform(sorted);

    // Eytzinger layout for [1,2,3,4,5,6,7]:
    // Index: 0  1  2  3  4  5  6  7
    // Value: -  4  2  6  1  3  5  7  (root=4, left subtree root=2, right=6)
    EXPECT_EQ(eytz.size(), 8);
    EXPECT_EQ(eytz[1], 4);  // Root
    EXPECT_EQ(eytz[2], 2);  // Left child of root
    EXPECT_EQ(eytz[3], 6);  // Right child of root
    EXPECT_EQ(eytz[4], 1);
    EXPECT_EQ(eytz[5], 3);
    EXPECT_EQ(eytz[6], 5);
    EXPECT_EQ(eytz[7], 7);
}

TEST(EytzingerLayoutTest, TransformEmpty) {
    std::vector<std::uint64_t> sorted;
    auto eytz = simdex::EytzingerLayout<std::uint64_t>::transform(sorted);
    EXPECT_EQ(eytz.size(), 1);  // Just the dummy element
}

TEST(EytzingerLayoutTest, TransformSingle) {
    std::vector<std::uint64_t> sorted = {42};
    auto eytz = simdex::EytzingerLayout<std::uint64_t>::transform(sorted);
    EXPECT_EQ(eytz.size(), 2);
    EXPECT_EQ(eytz[1], 42);
}

TEST(EytzingerSearchTest, SearchSmall) {
    std::vector<std::uint64_t> sorted = {1, 2, 3, 4, 5, 6, 7};
    auto eytz = simdex::EytzingerLayout<std::uint64_t>::transform(sorted);

    simdex::EytzingerSearch<std::uint64_t> search;

    // Test finding each element
    for (std::uint64_t key = 1; key <= 7; ++key) {
        std::size_t idx = search.search(eytz.data(), sorted.size(), key);
        EXPECT_LT(idx, sorted.size()) << "Key: " << key;
        EXPECT_EQ(sorted[idx], key) << "Key: " << key;
    }

    // Test not found (larger than all)
    std::size_t idx = search.search(eytz.data(), sorted.size(), 10);
    EXPECT_EQ(idx, sorted.size());

    // Test insertion point (between values)
    // Searching for 3 should find position of 3
    // Searching for value between 2 and 3 (like 2.5, but using 3) should also work
}

TEST(EytzingerSearchTest, SearchLarge) {
    std::vector<std::uint64_t> sorted(1000);
    std::iota(sorted.begin(), sorted.end(), 0);
    auto eytz = simdex::EytzingerLayout<std::uint64_t>::transform(sorted);

    simdex::EytzingerSearch<std::uint64_t> search;

    // Test all values
    for (std::uint64_t key = 0; key < 1000; ++key) {
        std::size_t idx = search.search(eytz.data(), sorted.size(), key);
        EXPECT_EQ(sorted[idx], key) << "Key: " << key;
    }
}

// ============================================================================
// Cross-strategy consistency tests
// ============================================================================

TEST_F(SearchStrategyTest, AllStrategiesAgree_Contiguous) {
    std::vector<std::uint64_t> queries;
    for (std::uint64_t i = 0; i <= 260; i += 7) {
        queries.push_back(i);
    }

    for (auto key : queries) {
        auto scalar_result = scalar_.lower_bound(
            contiguous_256_.data(), 0, contiguous_256_.size(), key);
        auto linear_result = linear_.lower_bound(
            contiguous_256_.data(), 0, contiguous_256_.size(), key);
        auto kary_result = kary_.lower_bound(
            contiguous_256_.data(), 0, contiguous_256_.size(), key);

        EXPECT_EQ(scalar_result, linear_result)
            << "Key: " << key << " (scalar vs linear)";
        EXPECT_EQ(scalar_result, kary_result)
            << "Key: " << key << " (scalar vs kary)";
    }
}

TEST_F(SearchStrategyTest, AllStrategiesAgree_Random) {
    std::mt19937_64 rng(789);
    std::uniform_int_distribution<std::uint64_t> dist(0, 1'500'000);

    for (int i = 0; i < 1000; ++i) {
        std::uint64_t key = dist(rng);

        auto scalar_result = scalar_.lower_bound(
            random_1000_.data(), 0, random_1000_.size(), key);
        auto linear_result = linear_.lower_bound(
            random_1000_.data(), 0, random_1000_.size(), key);
        auto kary_result = kary_.lower_bound(
            random_1000_.data(), 0, random_1000_.size(), key);

        EXPECT_EQ(scalar_result, linear_result)
            << "Key: " << key << " (scalar vs linear)";
        EXPECT_EQ(scalar_result, kary_result)
            << "Key: " << key << " (scalar vs kary)";
    }
}

// ============================================================================
// Strategy metadata tests
// ============================================================================

TEST(StrategyMetadataTest, Names) {
    EXPECT_STREQ(simdex::ScalarBaseline<std::uint64_t>::name(), "scalar_baseline");

    // SIMD strategy names depend on platform
    const char* linear_name = simdex::SIMDLinearScan<std::uint64_t>::name();
    EXPECT_TRUE(
        std::string(linear_name).find("linear_scan") != std::string::npos);

    const char* kary_name = simdex::SIMDKarySearch<std::uint64_t>::name();
    EXPECT_TRUE(
        std::string(kary_name).find("kary_search") != std::string::npos);

    EXPECT_STREQ(simdex::EytzingerSearch<std::uint64_t>::name(), "eytzinger_search");
}

TEST(StrategyMetadataTest, OptimalRangeSizes) {
    EXPECT_EQ(simdex::ScalarBaseline<std::uint64_t>::optimal_range_size(), 0);
    EXPECT_EQ(simdex::SIMDLinearScan<std::uint64_t>::optimal_range_size(), 32);
    EXPECT_EQ(simdex::SIMDKarySearch<std::uint64_t>::optimal_range_size(), 128);
    EXPECT_EQ(simdex::EytzingerSearch<std::uint64_t>::optimal_range_size(), 512);
}

} // namespace
