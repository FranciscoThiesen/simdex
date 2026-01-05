#pragma once

/// @file concepts.hpp
/// @brief C++20 concepts for search strategies

#include <concepts>
#include <cstddef>

namespace simdex {

/// @brief Concept for a search strategy that can find keys in sorted ranges
///
/// A SearchStrategy must provide:
/// - lower_bound(data, lo, hi, key) -> pointer to first element >= key
/// - name() -> const char* for logging/benchmarking
/// - optimal_range_size() -> hint for when this strategy is best
template<typename S, typename T>
concept SearchStrategy = requires(const S strategy, const T* data,
                                   std::size_t lo, std::size_t hi, T key) {
    // Core search operation: find first element >= key in [lo, hi)
    { strategy.lower_bound(data, lo, hi, key) } -> std::convertible_to<const T*>;

    // Strategy name for logging/benchmarking
    { S::name() } -> std::convertible_to<const char*>;

    // Hint for optimal range size (strategy selection heuristic)
    { S::optimal_range_size() } -> std::convertible_to<std::size_t>;
};

/// @brief Concept for key types that can be compared and loaded
template<typename T>
concept SearchableKey = std::totally_ordered<T> &&
                        std::is_trivially_copyable_v<T> &&
                        (sizeof(T) == 4 || sizeof(T) == 8);

} // namespace simdex
