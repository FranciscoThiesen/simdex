/// @file test_baseline.cpp
/// @brief Tests for scalar baseline search strategy

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "simdex/simdex.hpp"

namespace {

class ScalarBaselineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create sorted test data
        data_small_.resize(16);
        std::iota(data_small_.begin(), data_small_.end(), 0);

        data_medium_.resize(256);
        std::iota(data_medium_.begin(), data_medium_.end(), 0);

        data_large_.resize(10000);
        std::iota(data_large_.begin(), data_large_.end(), 0);

        // Create data with gaps
        data_gaps_ = {1, 5, 10, 15, 20, 50, 100, 200, 500, 1000};
    }

    std::vector<std::uint64_t> data_small_;
    std::vector<std::uint64_t> data_medium_;
    std::vector<std::uint64_t> data_large_;
    std::vector<std::uint64_t> data_gaps_;
    simdex::ScalarBaseline<std::uint64_t> strategy_;
};

TEST_F(ScalarBaselineTest, FindsExactMatch) {
    const auto* result = strategy_.lower_bound(
        data_small_.data(), 0, data_small_.size(), 8);

    ASSERT_NE(result, data_small_.data() + data_small_.size());
    EXPECT_EQ(*result, 8);
}

TEST_F(ScalarBaselineTest, FindsFirstElement) {
    const auto* result = strategy_.lower_bound(
        data_small_.data(), 0, data_small_.size(), 0);

    EXPECT_EQ(result, data_small_.data());
    EXPECT_EQ(*result, 0);
}

TEST_F(ScalarBaselineTest, FindsLastElement) {
    const auto* result = strategy_.lower_bound(
        data_small_.data(), 0, data_small_.size(), 15);

    EXPECT_EQ(result, data_small_.data() + 15);
    EXPECT_EQ(*result, 15);
}

TEST_F(ScalarBaselineTest, ReturnsEndForNotFound) {
    const auto* result = strategy_.lower_bound(
        data_small_.data(), 0, data_small_.size(), 100);

    EXPECT_EQ(result, data_small_.data() + data_small_.size());
}

TEST_F(ScalarBaselineTest, WorksWithSubrange) {
    // Search in range [4, 12)
    const auto* result = strategy_.lower_bound(
        data_small_.data(), 4, 12, 8);

    EXPECT_EQ(result, data_small_.data() + 8);
    EXPECT_EQ(*result, 8);
}

TEST_F(ScalarBaselineTest, HandlesEmptyRange) {
    const auto* result = strategy_.lower_bound(
        data_small_.data(), 5, 5, 10);

    EXPECT_EQ(result, data_small_.data() + 5);
}

TEST_F(ScalarBaselineTest, HandlesSingleElement) {
    const auto* result = strategy_.lower_bound(
        data_small_.data(), 5, 6, 5);

    EXPECT_EQ(result, data_small_.data() + 5);
    EXPECT_EQ(*result, 5);
}

TEST_F(ScalarBaselineTest, FindsInsertionPoint) {
    // Key 7 doesn't exist in gaps data, should find 10 (next larger)
    const auto* result = strategy_.lower_bound(
        data_gaps_.data(), 0, data_gaps_.size(), 7);

    ASSERT_NE(result, data_gaps_.data() + data_gaps_.size());
    EXPECT_EQ(*result, 10);
}

TEST_F(ScalarBaselineTest, WorksWithLargeData) {
    const auto* result = strategy_.lower_bound(
        data_large_.data(), 0, data_large_.size(), 5000);

    ASSERT_NE(result, data_large_.data() + data_large_.size());
    EXPECT_EQ(*result, 5000);
}

TEST_F(ScalarBaselineTest, MatchesStdLowerBound) {
    // Verify our implementation matches std::lower_bound for random queries
    std::mt19937 rng(42);
    std::uniform_int_distribution<std::uint64_t> dist(0, 15000);

    for (int i = 0; i < 1000; ++i) {
        std::uint64_t key = dist(rng);

        const auto* our_result = strategy_.lower_bound(
            data_large_.data(), 0, data_large_.size(), key);
        auto std_result = std::lower_bound(
            data_large_.begin(), data_large_.end(), key);

        if (std_result == data_large_.end()) {
            EXPECT_EQ(our_result, data_large_.data() + data_large_.size())
                << "Key: " << key;
        } else {
            EXPECT_EQ(our_result, &*std_result) << "Key: " << key;
        }
    }
}

TEST(ScalarBaselineMetaTest, StrategyName) {
    EXPECT_STREQ(simdex::ScalarBaseline<std::uint64_t>::name(), "scalar_baseline");
}

TEST(ScalarBaselineMetaTest, OptimalRangeSize) {
    EXPECT_EQ(simdex::ScalarBaseline<std::uint64_t>::optimal_range_size(), 0);
}

TEST(PlatformTest, ReportsPlatform) {
    const char* info = simdex::platform_info();
    EXPECT_NE(info, nullptr);
    EXPECT_GT(std::strlen(info), 0);

#if defined(SIMDEX_HAS_AVX2)
    EXPECT_STREQ(info, "x86_64 with AVX2");
#elif defined(SIMDEX_HAS_NEON)
    EXPECT_STREQ(info, "ARM64 with NEON");
#else
    EXPECT_STREQ(info, "scalar (no SIMD)");
#endif
}

TEST(VersionTest, ReportsVersion) {
    EXPECT_STREQ(simdex::version(), "0.1.0");
}

} // namespace
