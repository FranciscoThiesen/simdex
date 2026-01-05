#pragma once

/// @file simd_pgm_index.hpp
/// @brief SIMD-accelerated PGM-Index wrapper
///
/// Wraps PGM-Index with SIMD-optimized last-mile search strategies,
/// automatically selecting the best strategy based on the predicted range size.

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <vector>

#include <pgm/pgm_index.hpp>

#include "simdex/core/concepts.hpp"
#include "simdex/core/types.hpp"
#include "simdex/search/scalar_baseline.hpp"
#include "simdex/search/linear_scan.hpp"
#include "simdex/search/kary_search.hpp"

namespace simdex {

/// @brief Strategy selection policy for automatic strategy dispatch
enum class StrategyPolicy {
    Auto,           ///< Automatically select based on range size
    AlwaysLinear,   ///< Always use SIMD linear scan
    AlwaysKary,     ///< Always use SIMD k-ary search
    AlwaysScalar    ///< Always use scalar baseline (for comparison)
};

/// @brief SIMD-accelerated PGM-Index
///
/// Wraps the PGM-Index with SIMD-optimized last-mile search. After the
/// PGM model predicts an approximate position, this class uses SIMD
/// instructions to quickly search the small range [lo, hi).
///
/// @tparam K Key type (must be arithmetic)
/// @tparam Epsilon Error bound for PGM-Index (default: 64)
/// @tparam EpsilonRecursive Error bound for recursive levels (default: 4)
template<typename K,
         std::size_t Epsilon = 64,
         std::size_t EpsilonRecursive = 4>
class SIMDPGMIndex {
public:
    using key_type = K;
    using size_type = std::size_t;
    using value_type = K;
    using iterator = typename std::vector<K>::const_iterator;

    /// @brief Construct from a sorted range
    /// @param first Iterator to first element
    /// @param last Iterator past last element
    template<typename RandomIt>
    SIMDPGMIndex(RandomIt first, RandomIt last)
        : data_(first, last)
        , pgm_(data_.begin(), data_.end()) {
        static_assert(std::is_arithmetic_v<K>, "Key type must be arithmetic");
    }

    /// @brief Construct from a sorted vector
    /// @param sorted_data A sorted vector of keys (will be copied or moved)
    explicit SIMDPGMIndex(std::vector<K> sorted_data)
        : data_(std::move(sorted_data))
        , pgm_(data_.begin(), data_.end()) {}

    // ========================================================================
    // Core lookup operations
    // ========================================================================

    /// @brief Find the first element >= key
    /// @param key The key to search for
    /// @param policy Strategy selection policy
    /// @return Iterator to first element >= key, or end() if not found
    [[nodiscard]]
    iterator lower_bound(K key, StrategyPolicy policy = StrategyPolicy::Auto) const {
        if (data_.empty()) {
            return data_.end();
        }

        // Get approximate position from PGM model
        auto approx = pgm_.search(key);

        // Clamp bounds to valid range
        size_type lo = approx.lo;
        size_type hi = std::min(approx.hi, data_.size());

        // Dispatch to appropriate SIMD strategy
        const K* result = dispatch_search(lo, hi, key, policy);

        // Convert pointer to iterator
        return data_.begin() + (result - data_.data());
    }

    /// @brief Find element equal to key
    /// @param key The key to search for
    /// @return Iterator to element, or end() if not found
    [[nodiscard]]
    iterator find(K key, StrategyPolicy policy = StrategyPolicy::Auto) const {
        auto it = lower_bound(key, policy);
        if (it != data_.end() && *it == key) {
            return it;
        }
        return data_.end();
    }

    /// @brief Check if key exists in the index
    [[nodiscard]]
    bool contains(K key, StrategyPolicy policy = StrategyPolicy::Auto) const {
        return find(key, policy) != data_.end();
    }

    /// @brief Count occurrences of key (0 or 1 for unique keys)
    [[nodiscard]]
    size_type count(K key, StrategyPolicy policy = StrategyPolicy::Auto) const {
        return contains(key, policy) ? 1 : 0;
    }

    // ========================================================================
    // Container-like interface
    // ========================================================================

    [[nodiscard]] iterator begin() const noexcept { return data_.begin(); }
    [[nodiscard]] iterator end() const noexcept { return data_.end(); }
    [[nodiscard]] size_type size() const noexcept { return data_.size(); }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

    [[nodiscard]] const K& operator[](size_type idx) const { return data_[idx]; }
    [[nodiscard]] const K& at(size_type idx) const { return data_.at(idx); }

    [[nodiscard]] const K& front() const { return data_.front(); }
    [[nodiscard]] const K& back() const { return data_.back(); }

    [[nodiscard]] const K* data() const noexcept { return data_.data(); }

    // ========================================================================
    // Index statistics
    // ========================================================================

    /// @brief Get the PGM-Index epsilon (max error bound)
    [[nodiscard]]
    static constexpr size_type epsilon() noexcept { return Epsilon; }

    /// @brief Get the approximate position for a key (for debugging/analysis)
    [[nodiscard]]
    pgm::ApproxPos get_approx_pos(K key) const {
        return pgm_.search(key);
    }

    /// @brief Get the number of segments in the PGM model
    [[nodiscard]]
    size_type segment_count() const {
        return pgm_.segments_count();
    }

    /// @brief Get the height of the PGM model (number of levels)
    [[nodiscard]]
    size_type height() const {
        return pgm_.height();
    }

    /// @brief Estimate memory usage in bytes
    [[nodiscard]]
    size_type memory_usage() const {
        return data_.capacity() * sizeof(K) + pgm_.size_in_bytes();
    }

private:
    std::vector<K> data_;
    pgm::PGMIndex<K, Epsilon, EpsilonRecursive> pgm_;

    // Search strategies (stateless, so we can keep them as members)
    ScalarBaseline<K> scalar_strategy_;
    SIMDLinearScan<K> linear_strategy_;
    SIMDKarySearch<K> kary_strategy_;

    // Platform-aware thresholds
    // SIMD linear scan does N/lanes iterations vs log2(N) for binary search
    // Only wins when N/lanes < ~2*log2(N), accounting for SIMD overhead
    //
    // NEON (2 lanes for 64-bit): linear wins for N <= ~8
    // AVX2 (4 lanes for 64-bit): linear wins for N <= ~32
    //
    // K-ary search with K lanes does log_{K+1}(N) iterations
    // But each iteration has K memory accesses vs 1 for binary search
    // Overhead means it rarely wins over well-predicted binary search

#if defined(SIMDEX_HAS_AVX2)
    static constexpr size_type kLinearThreshold = 32;   // 4 lanes, log2(32)=5, 32/4=8 iterations
    static constexpr size_type kKaryThreshold = 128;    // k-ary viable for medium ranges
#elif defined(SIMDEX_HAS_NEON)
    static constexpr size_type kLinearThreshold = 8;    // 2 lanes, log2(8)=3, 8/2=4 iterations
    static constexpr size_type kKaryThreshold = 16;     // k-ary has high overhead with K=2
#else
    static constexpr size_type kLinearThreshold = 0;    // No SIMD, always use scalar
    static constexpr size_type kKaryThreshold = 0;
#endif

    /// @brief Dispatch to the appropriate search strategy
    [[nodiscard]]
    const K* dispatch_search(size_type lo, size_type hi, K key,
                             StrategyPolicy policy) const noexcept {
        switch (policy) {
            case StrategyPolicy::AlwaysLinear:
                return linear_strategy_.lower_bound(data_.data(), lo, hi, key);

            case StrategyPolicy::AlwaysKary:
                return kary_strategy_.lower_bound(data_.data(), lo, hi, key);

            case StrategyPolicy::AlwaysScalar:
                return scalar_strategy_.lower_bound(data_.data(), lo, hi, key);

            case StrategyPolicy::Auto:
            default:
                return auto_select_search(lo, hi, key);
        }
    }

    /// @brief Automatically select the best strategy based on range size
    [[nodiscard]]
    const K* auto_select_search(size_type lo, size_type hi, K key) const noexcept {
        size_type range_size = hi - lo;

        if (range_size <= kLinearThreshold) {
            // Small range: SIMD linear scan wins
            return linear_strategy_.lower_bound(data_.data(), lo, hi, key);
        } else if (range_size <= kKaryThreshold) {
            // Medium range: SIMD k-ary search wins
            return kary_strategy_.lower_bound(data_.data(), lo, hi, key);
        } else {
            // Large range: fall back to scalar (or could use Eytzinger)
            // Note: Epsilon is typically 64-256, so this case is rare
            return scalar_strategy_.lower_bound(data_.data(), lo, hi, key);
        }
    }
};

// Convenience type aliases for common configurations
using SIMDPGMIndex64 = SIMDPGMIndex<std::uint64_t, 64>;
using SIMDPGMIndex32 = SIMDPGMIndex<std::uint32_t, 64>;

// Variants with different epsilon values
template<typename K>
using SIMDPGMIndexTight = SIMDPGMIndex<K, 16>;   // Tight bounds, smaller ranges

template<typename K>
using SIMDPGMIndexLoose = SIMDPGMIndex<K, 256>;  // Loose bounds, larger ranges

} // namespace simdex
