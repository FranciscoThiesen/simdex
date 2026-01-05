/// @file test_pgm_integration.cpp
/// @brief Integration tests for SIMDPGMIndex

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "simdex/simdex.hpp"

namespace {

// ============================================================================
// Test fixture
// ============================================================================

class SIMDPGMIndexTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Contiguous data
        contiguous_.resize(10000);
        std::iota(contiguous_.begin(), contiguous_.end(), 0);

        // Sparse data with gaps
        sparse_ = {1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000};

        // Random data (unique, sorted)
        std::mt19937_64 rng(42);
        std::uniform_int_distribution<std::uint64_t> dist(0, 10'000'000);
        std::set<std::uint64_t> unique_values;
        while (unique_values.size() < 100000) {
            unique_values.insert(dist(rng));
        }
        random_.assign(unique_values.begin(), unique_values.end());
    }

    std::vector<std::uint64_t> contiguous_;
    std::vector<std::uint64_t> sparse_;
    std::vector<std::uint64_t> random_;
};

// ============================================================================
// Construction tests
// ============================================================================

TEST_F(SIMDPGMIndexTest, ConstructFromIterators) {
    simdex::SIMDPGMIndex<std::uint64_t> index(contiguous_.begin(), contiguous_.end());
    EXPECT_EQ(index.size(), contiguous_.size());
    EXPECT_FALSE(index.empty());
}

TEST_F(SIMDPGMIndexTest, ConstructFromVector) {
    simdex::SIMDPGMIndex<std::uint64_t> index(contiguous_);
    EXPECT_EQ(index.size(), contiguous_.size());
}

TEST_F(SIMDPGMIndexTest, ConstructFromMovedVector) {
    auto data_copy = contiguous_;
    simdex::SIMDPGMIndex<std::uint64_t> index(std::move(data_copy));
    EXPECT_EQ(index.size(), contiguous_.size());
    EXPECT_TRUE(data_copy.empty());  // Moved from
}

TEST_F(SIMDPGMIndexTest, EmptyIndex) {
    std::vector<std::uint64_t> empty;
    simdex::SIMDPGMIndex<std::uint64_t> index(empty);
    EXPECT_TRUE(index.empty());
    EXPECT_EQ(index.size(), 0);
    EXPECT_EQ(index.begin(), index.end());
}

// ============================================================================
// Lookup tests - Auto policy
// ============================================================================

TEST_F(SIMDPGMIndexTest, LowerBoundFindsExact) {
    simdex::SIMDPGMIndex<std::uint64_t> index(contiguous_);

    for (std::uint64_t key : {0ULL, 100ULL, 5000ULL, 9999ULL}) {
        auto it = index.lower_bound(key);
        ASSERT_NE(it, index.end()) << "Key: " << key;
        EXPECT_EQ(*it, key) << "Key: " << key;
    }
}

TEST_F(SIMDPGMIndexTest, LowerBoundFindsInsertionPoint) {
    simdex::SIMDPGMIndex<std::uint64_t> index(sparse_);

    // Key 7 doesn't exist, should find 10 (next larger)
    auto it = index.lower_bound(7);
    ASSERT_NE(it, index.end());
    EXPECT_EQ(*it, 10);

    // Key 2 doesn't exist, should find 5
    it = index.lower_bound(2);
    ASSERT_NE(it, index.end());
    EXPECT_EQ(*it, 5);
}

TEST_F(SIMDPGMIndexTest, LowerBoundReturnsEndForLargeKey) {
    simdex::SIMDPGMIndex<std::uint64_t> index(contiguous_);

    auto it = index.lower_bound(100000);  // Larger than any element
    EXPECT_EQ(it, index.end());
}

TEST_F(SIMDPGMIndexTest, FindExisting) {
    simdex::SIMDPGMIndex<std::uint64_t> index(contiguous_);

    auto it = index.find(5000);
    ASSERT_NE(it, index.end());
    EXPECT_EQ(*it, 5000);
}

TEST_F(SIMDPGMIndexTest, FindNonExisting) {
    simdex::SIMDPGMIndex<std::uint64_t> index(sparse_);

    auto it = index.find(7);  // Doesn't exist
    EXPECT_EQ(it, index.end());
}

TEST_F(SIMDPGMIndexTest, Contains) {
    simdex::SIMDPGMIndex<std::uint64_t> index(sparse_);

    EXPECT_TRUE(index.contains(100));
    EXPECT_TRUE(index.contains(1000));
    EXPECT_FALSE(index.contains(7));
    EXPECT_FALSE(index.contains(999));
}

TEST_F(SIMDPGMIndexTest, Count) {
    simdex::SIMDPGMIndex<std::uint64_t> index(sparse_);

    EXPECT_EQ(index.count(100), 1);
    EXPECT_EQ(index.count(7), 0);
}

// ============================================================================
// Lookup tests - All policies
// ============================================================================

TEST_F(SIMDPGMIndexTest, AllPoliciesProduceSameResults) {
    simdex::SIMDPGMIndex<std::uint64_t> index(random_);

    std::mt19937_64 rng(123);
    std::uniform_int_distribution<std::uint64_t> dist(0, 15'000'000);

    for (int i = 0; i < 1000; ++i) {
        std::uint64_t key = dist(rng);

        auto auto_result = index.lower_bound(key, simdex::StrategyPolicy::Auto);
        auto linear_result = index.lower_bound(key, simdex::StrategyPolicy::AlwaysLinear);
        auto kary_result = index.lower_bound(key, simdex::StrategyPolicy::AlwaysKary);
        auto scalar_result = index.lower_bound(key, simdex::StrategyPolicy::AlwaysScalar);

        EXPECT_EQ(auto_result, linear_result) << "Key: " << key << " (auto vs linear)";
        EXPECT_EQ(auto_result, kary_result) << "Key: " << key << " (auto vs kary)";
        EXPECT_EQ(auto_result, scalar_result) << "Key: " << key << " (auto vs scalar)";
    }
}

TEST_F(SIMDPGMIndexTest, MatchesStdLowerBound) {
    simdex::SIMDPGMIndex<std::uint64_t> index(random_);

    // Test with existing values
    for (const auto& key : random_) {
        auto simd_it = index.lower_bound(key);
        auto std_it = std::lower_bound(random_.begin(), random_.end(), key);

        ASSERT_NE(simd_it, index.end());
        EXPECT_EQ(*simd_it, *std_it) << "Key: " << key;
    }

    // Test with random queries
    std::mt19937_64 rng(456);
    std::uniform_int_distribution<std::uint64_t> dist(0, 15'000'000);

    for (int i = 0; i < 1000; ++i) {
        std::uint64_t key = dist(rng);

        auto simd_it = index.lower_bound(key);
        auto std_it = std::lower_bound(random_.begin(), random_.end(), key);

        if (std_it == random_.end()) {
            EXPECT_EQ(simd_it, index.end()) << "Key: " << key;
        } else {
            ASSERT_NE(simd_it, index.end()) << "Key: " << key;
            EXPECT_EQ(*simd_it, *std_it) << "Key: " << key;
        }
    }
}

// ============================================================================
// Container interface tests
// ============================================================================

TEST_F(SIMDPGMIndexTest, ContainerInterface) {
    simdex::SIMDPGMIndex<std::uint64_t> index(contiguous_);

    EXPECT_EQ(index.front(), 0);
    EXPECT_EQ(index.back(), 9999);
    EXPECT_EQ(index[0], 0);
    EXPECT_EQ(index[5000], 5000);
    EXPECT_EQ(index.at(100), 100);
    EXPECT_NE(index.data(), nullptr);
}

TEST_F(SIMDPGMIndexTest, Iteration) {
    simdex::SIMDPGMIndex<std::uint64_t> index(sparse_);

    std::vector<std::uint64_t> iterated;
    for (auto val : index) {
        iterated.push_back(val);
    }

    EXPECT_EQ(iterated, sparse_);
}

// ============================================================================
// Index statistics tests
// ============================================================================

TEST_F(SIMDPGMIndexTest, Statistics) {
    simdex::SIMDPGMIndex<std::uint64_t, 64> index(random_);

    EXPECT_EQ(index.epsilon(), 64);
    EXPECT_GT(index.segment_count(), 0);
    EXPECT_GE(index.height(), 1);
    EXPECT_GT(index.memory_usage(), 0);
}

TEST_F(SIMDPGMIndexTest, ApproxPos) {
    simdex::SIMDPGMIndex<std::uint64_t, 64> index(contiguous_);

    auto approx = index.get_approx_pos(5000);

    // Position should be approximately correct
    EXPECT_NEAR(static_cast<double>(approx.pos), 5000.0, 64.0 * 2);

    // Range should contain the key
    EXPECT_LE(approx.lo, 5000);
    EXPECT_GT(approx.hi, 5000);

    // Range size should be bounded by 2*epsilon + 2
    EXPECT_LE(approx.hi - approx.lo, 2 * 64 + 2);
}

// ============================================================================
// Different epsilon configurations
// ============================================================================

TEST_F(SIMDPGMIndexTest, TightEpsilon) {
    simdex::SIMDPGMIndex<std::uint64_t, 16> index(contiguous_);

    // Should still find all keys correctly
    for (std::uint64_t key : {0ULL, 1000ULL, 5000ULL, 9999ULL}) {
        auto it = index.lower_bound(key);
        ASSERT_NE(it, index.end()) << "Key: " << key;
        EXPECT_EQ(*it, key) << "Key: " << key;
    }

    // Epsilon should be 16
    EXPECT_EQ(index.epsilon(), 16);
}

TEST_F(SIMDPGMIndexTest, LooseEpsilon) {
    simdex::SIMDPGMIndex<std::uint64_t, 256> index(contiguous_);

    // Should still find all keys correctly
    for (std::uint64_t key : {0ULL, 1000ULL, 5000ULL, 9999ULL}) {
        auto it = index.lower_bound(key);
        ASSERT_NE(it, index.end()) << "Key: " << key;
        EXPECT_EQ(*it, key) << "Key: " << key;
    }

    // Epsilon should be 256
    EXPECT_EQ(index.epsilon(), 256);
}

// ============================================================================
// 32-bit key tests
// ============================================================================

TEST(SIMDPGMIndex32Test, BasicOperations) {
    std::vector<std::uint32_t> data(10000);
    std::iota(data.begin(), data.end(), 0);

    simdex::SIMDPGMIndex<std::uint32_t> index(data);

    EXPECT_EQ(index.size(), 10000);

    auto it = index.lower_bound(5000);
    ASSERT_NE(it, index.end());
    EXPECT_EQ(*it, 5000);

    EXPECT_TRUE(index.contains(1000));
    EXPECT_FALSE(index.contains(100000));
}

// ============================================================================
// Edge cases
// ============================================================================

TEST_F(SIMDPGMIndexTest, SingleElement) {
    std::vector<std::uint64_t> single = {42};
    simdex::SIMDPGMIndex<std::uint64_t> index(single);

    EXPECT_EQ(index.size(), 1);
    EXPECT_TRUE(index.contains(42));
    EXPECT_FALSE(index.contains(0));
    EXPECT_FALSE(index.contains(100));

    auto it = index.lower_bound(0);
    EXPECT_EQ(*it, 42);  // First element >= 0

    it = index.lower_bound(42);
    EXPECT_EQ(*it, 42);

    it = index.lower_bound(100);
    EXPECT_EQ(it, index.end());
}

TEST_F(SIMDPGMIndexTest, TwoElements) {
    std::vector<std::uint64_t> two = {10, 20};
    simdex::SIMDPGMIndex<std::uint64_t> index(two);

    EXPECT_EQ(*index.lower_bound(5), 10);
    EXPECT_EQ(*index.lower_bound(10), 10);
    EXPECT_EQ(*index.lower_bound(15), 20);
    EXPECT_EQ(*index.lower_bound(20), 20);
    EXPECT_EQ(index.lower_bound(25), index.end());
}

TEST_F(SIMDPGMIndexTest, FirstAndLastElements) {
    simdex::SIMDPGMIndex<std::uint64_t> index(contiguous_);

    // First element
    auto it = index.lower_bound(0);
    EXPECT_EQ(it, index.begin());
    EXPECT_EQ(*it, 0);

    // Last element
    it = index.lower_bound(9999);
    EXPECT_EQ(*it, 9999);
    EXPECT_EQ(std::next(it), index.end());
}

} // namespace
