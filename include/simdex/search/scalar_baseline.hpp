#pragma once

/// @file scalar_baseline.hpp
/// @brief Scalar baseline search using std::lower_bound
///
/// This serves as the baseline for comparison. It wraps std::lower_bound
/// with the SearchStrategy interface.

#include <algorithm>
#include <cstddef>

#include "simdex/core/concepts.hpp"

namespace simdex {

/// @brief Scalar baseline search strategy using std::lower_bound
///
/// This is the standard approach used by PGM-Index and other learned indexes.
/// All SIMD strategies should produce identical results to this baseline.
template<typename T>
class ScalarBaseline {
public:
    static constexpr const char* name() noexcept { return "scalar_baseline"; }

    /// @brief Optimal range: baseline works for any size, but SIMD wins for small
    static constexpr std::size_t optimal_range_size() noexcept {
        return 0; // No particular sweet spot
    }

    /// @brief Find first element >= key in [lo, hi)
    /// @param data Pointer to sorted array
    /// @param lo Start index (inclusive)
    /// @param hi End index (exclusive)
    /// @param key Key to search for
    /// @return Pointer to first element >= key, or data + hi if not found
    [[nodiscard]]
    const T* lower_bound(const T* data, std::size_t lo,
                         std::size_t hi, T key) const noexcept {
        return std::lower_bound(data + lo, data + hi, key);
    }
};

// Verify concept satisfaction
static_assert(SearchStrategy<ScalarBaseline<std::uint64_t>, std::uint64_t>);
static_assert(SearchStrategy<ScalarBaseline<std::uint32_t>, std::uint32_t>);

} // namespace simdex
