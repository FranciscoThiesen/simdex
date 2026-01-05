#pragma once

/// @file linear_scan.hpp
/// @brief SIMD-accelerated linear scan search strategy
///
/// This strategy excels for small ranges (< 32 elements) where the overhead
/// of binary search's random access pattern exceeds the cost of scanning.
/// Uses SIMD to compare multiple elements per iteration.

#include <algorithm>
#include <cstddef>

#include "simdex/core/concepts.hpp"
#include "simdex/simd/platform.hpp"
#include "simdex/simd/vec_traits.hpp"

namespace simdex {

/// @brief SIMD-accelerated linear scan for lower_bound
///
/// Scans through the range comparing `lanes` elements per iteration using
/// SIMD operations. Falls back to scalar for the tail elements.
///
/// @tparam T Key type (uint32_t or uint64_t)
/// @tparam Platform SIMD platform tag (avx2_tag, neon_tag, or auto_platform_tag)
template<typename T, typename Platform = simd::auto_platform_tag>
class SIMDLinearScan {
public:
    using platform_type = simd::resolve_platform_t<Platform>;
    using traits = simd::vec_traits<T, platform_type>;

    static constexpr const char* name() noexcept {
        if constexpr (std::is_same_v<platform_type, simd::avx2_tag>) {
            return "simd_linear_scan_avx2";
        } else if constexpr (std::is_same_v<platform_type, simd::neon_tag>) {
            return "simd_linear_scan_neon";
        } else {
            return "simd_linear_scan_scalar";
        }
    }

    /// @brief Optimal for small ranges where cache locality beats log(n)
    static constexpr std::size_t optimal_range_size() noexcept {
        return 32;
    }

    /// @brief Find first element >= key in [lo, hi)
    ///
    /// Algorithm:
    /// 1. Broadcast search key to all SIMD lanes
    /// 2. Load `lanes` elements at a time
    /// 3. Compare: find first element >= key
    /// 4. Use movemask + ctz to find exact position
    /// 5. Scalar tail for remaining elements
    ///
    /// @param data Pointer to sorted array
    /// @param lo Start index (inclusive)
    /// @param hi End index (exclusive)
    /// @param key Key to search for
    /// @return Pointer to first element >= key, or data + hi if not found
    [[nodiscard]]
    const T* lower_bound(const T* data, std::size_t lo,
                         std::size_t hi, T key) const noexcept {
        constexpr std::size_t lanes = traits::lanes;

        // For very small ranges or scalar platform, use simple scan
        if constexpr (lanes == 1) {
            return scalar_lower_bound(data, lo, hi, key);
        }

        // Broadcast search key to all lanes
        const auto needle = traits::broadcast(key);

        std::size_t i = lo;

        // SIMD loop: process `lanes` elements per iteration
        // We're looking for the first element >= key
        // Strategy: find first element where element >= key
        //          which is equivalent to NOT(element < key)
        //          or equivalently: first where key <= element
        for (; i + lanes <= hi; i += lanes) {
            // Load next chunk
            auto chunk = traits::load(data + i);

            // Compare: chunk >= key is equivalent to NOT(chunk < key)
            // But we have cmpgt, so: chunk > key OR chunk == key
            // Easier: find where chunk >= key by checking NOT(key > chunk)
            // Actually, for lower_bound we want first element >= key
            // So we check: key > chunk[i] for each lane
            // If key > chunk[i], we haven't found it yet (continue)
            // If key <= chunk[i], we found a candidate

            // cmpgt returns mask where needle > chunk (key > element)
            auto cmp = traits::cmpgt(needle, chunk);
            int mask = traits::movemask(cmp);

            // mask has bit set where key > element
            // we want first position where key <= element
            // that's the first 0 bit in the mask

            // Invert: now bits are set where key <= element
            constexpr int all_lanes_mask = (1 << lanes) - 1;
            int found_mask = (~mask) & all_lanes_mask;

            if (found_mask != 0) {
                // Found at least one element >= key
                int first_pos = simd::ctz(found_mask);
                return data + i + first_pos;
            }
        }

        // Scalar tail for remaining elements
        for (; i < hi; ++i) {
            if (data[i] >= key) {
                return data + i;
            }
        }

        return data + hi;  // Not found
    }

private:
    /// @brief Simple scalar lower_bound implementation
    [[nodiscard]]
    static const T* scalar_lower_bound(const T* data, std::size_t lo,
                                        std::size_t hi, T key) noexcept {
        for (std::size_t i = lo; i < hi; ++i) {
            if (data[i] >= key) {
                return data + i;
            }
        }
        return data + hi;
    }
};

// Verify concept satisfaction
static_assert(SearchStrategy<SIMDLinearScan<std::uint64_t>, std::uint64_t>);
static_assert(SearchStrategy<SIMDLinearScan<std::uint32_t>, std::uint32_t>);

// Convenience type aliases
using SIMDLinearScan64 = SIMDLinearScan<std::uint64_t>;
using SIMDLinearScan32 = SIMDLinearScan<std::uint32_t>;

} // namespace simdex
