#pragma once

/// @file kary_search.hpp
/// @brief SIMD-accelerated k-ary search strategy
///
/// K-ary search divides the search range into K+1 segments and uses SIMD
/// to compare the search key against K pivot points simultaneously.
/// This reduces the number of iterations from log2(n) to log_{K+1}(n).
///
/// Based on: Schlegel et al. "k-ary Search on Modern Processors"

#include <algorithm>
#include <cstddef>

#include "simdex/core/concepts.hpp"
#include "simdex/simd/platform.hpp"
#include "simdex/simd/vec_traits.hpp"

namespace simdex {

/// @brief SIMD-accelerated k-ary search
///
/// Compares against K pivots simultaneously, narrowing the search space
/// by a factor of K+1 per iteration instead of 2.
///
/// @tparam T Key type (uint32_t or uint64_t)
/// @tparam Platform SIMD platform tag
template<typename T, typename Platform = simd::auto_platform_tag>
class SIMDKarySearch {
public:
    using platform_type = simd::resolve_platform_t<Platform>;
    using traits = simd::vec_traits<T, platform_type>;

    // K is determined by the number of SIMD lanes
    static constexpr std::size_t K = traits::lanes;

    static constexpr const char* name() noexcept {
        if constexpr (std::is_same_v<platform_type, simd::avx2_tag>) {
            return "simd_kary_search_avx2";
        } else if constexpr (std::is_same_v<platform_type, simd::neon_tag>) {
            return "simd_kary_search_neon";
        } else {
            return "simd_kary_search_scalar";
        }
    }

    /// @brief Optimal for medium ranges where binary search overhead is noticeable
    static constexpr std::size_t optimal_range_size() noexcept {
        return 128;
    }

    /// @brief Find first element >= key in [lo, hi)
    ///
    /// Algorithm:
    /// 1. Divide range into K+1 equal segments
    /// 2. Load K pivot points (at segment boundaries)
    /// 3. Compare key against all K pivots using SIMD
    /// 4. Use popcount to determine which segment contains the key
    /// 5. Recurse into that segment
    /// 6. Fall back to linear scan for small ranges
    ///
    /// @param data Pointer to sorted array
    /// @param lo Start index (inclusive)
    /// @param hi End index (exclusive)
    /// @param key Key to search for
    /// @return Pointer to first element >= key, or data + hi if not found
    [[nodiscard]]
    const T* lower_bound(const T* data, std::size_t lo,
                         std::size_t hi, T key) const noexcept {
        // For scalar platform or tiny ranges, use simple binary search
        if constexpr (K == 1) {
            return std::lower_bound(data + lo, data + hi, key);
        }

        // Minimum range size for k-ary to be worthwhile
        constexpr std::size_t min_kary_size = K * 2;

        while (hi - lo > min_kary_size) {
            // Calculate K pivot positions (evenly spaced)
            const std::size_t range_size = hi - lo;
            const std::size_t step = range_size / (K + 1);

            if (step == 0) break;  // Range too small

            // Load K pivots into SIMD register
            // Pivots are at positions: lo + step, lo + 2*step, ..., lo + K*step
            alignas(traits::alignment) T pivots[K];
            for (std::size_t i = 0; i < K; ++i) {
                pivots[i] = data[lo + (i + 1) * step];
            }

            auto vec_pivots = traits::load_aligned(pivots);
            auto needle = traits::broadcast(key);

            // Compare: key > pivots[i] for each pivot
            auto cmp = traits::cmpgt(needle, vec_pivots);
            int mask = traits::movemask(cmp);

            // Count how many pivots are < key
            // This tells us which segment to search
            int segment = simd::popcount(mask);

            // Update range to the identified segment
            // segment 0: [lo, lo + step)
            // segment 1: [lo + step, lo + 2*step)
            // ...
            // segment K: [lo + K*step, hi)
            if (segment == 0) {
                hi = lo + step;
            } else if (segment == static_cast<int>(K)) {
                lo = lo + K * step;
            } else {
                lo = lo + static_cast<std::size_t>(segment) * step;
                hi = lo + step;
            }
        }

        // Linear scan for small remaining range
        for (std::size_t i = lo; i < hi; ++i) {
            if (data[i] >= key) {
                return data + i;
            }
        }

        return data + hi;
    }
};

// Verify concept satisfaction
static_assert(SearchStrategy<SIMDKarySearch<std::uint64_t>, std::uint64_t>);
static_assert(SearchStrategy<SIMDKarySearch<std::uint32_t>, std::uint32_t>);

// Convenience type aliases
using SIMDKarySearch64 = SIMDKarySearch<std::uint64_t>;
using SIMDKarySearch32 = SIMDKarySearch<std::uint32_t>;

} // namespace simdex
