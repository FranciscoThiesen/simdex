#pragma once

/// @file eytzinger.hpp
/// @brief Eytzinger (BFS) layout for cache-optimized binary search
///
/// The Eytzinger layout stores a sorted array in BFS order of an implicit
/// binary search tree. This enables:
/// 1. Branchless binary search (always predictable memory access pattern)
/// 2. Effective prefetching (can prefetch deeper levels ahead)
/// 3. Better cache utilization for cold data
///
/// Based on: Algorithmica - Eytzinger Binary Search
/// https://algorithmica.org/en/eytzinger

#include <algorithm>
#include <cstddef>
#include <vector>

#include "simdex/core/concepts.hpp"

namespace simdex {

/// @brief Utilities for Eytzinger layout transformation and search
template<typename T>
class EytzingerLayout {
public:
    /// @brief Transform sorted array to Eytzinger (BFS) layout
    ///
    /// The resulting array is 1-indexed (element 0 is unused) to simplify
    /// the tree navigation: left child of node k is 2k, right child is 2k+1.
    ///
    /// @param sorted Input sorted array
    /// @return Eytzinger-layout array (size = sorted.size() + 1)
    [[nodiscard]]
    static std::vector<T> transform(const std::vector<T>& sorted) {
        if (sorted.empty()) {
            return {T{}};  // Just the dummy element at index 0
        }

        std::vector<T> eytz(sorted.size() + 1);
        eytz[0] = T{};  // Unused element at index 0

        build_recursive(sorted.data(), eytz.data(),
                       sorted.size(), 0, 1);
        return eytz;
    }

    /// @brief Transform in-place using a temporary buffer
    /// @param data Sorted array to transform
    /// @return Eytzinger-layout array
    [[nodiscard]]
    static std::vector<T> transform(std::vector<T>&& sorted) {
        return transform(static_cast<const std::vector<T>&>(sorted));
    }

private:
    /// @brief Recursively build Eytzinger layout
    /// @param src Source sorted array
    /// @param dst Destination Eytzinger array (1-indexed)
    /// @param n Size of source array
    /// @param i Current position in source array
    /// @param k Current position in Eytzinger tree (1-indexed)
    /// @return Next position in source array
    static std::size_t build_recursive(const T* src, T* dst,
                                        std::size_t n,
                                        std::size_t i, std::size_t k) {
        if (k <= n) {
            i = build_recursive(src, dst, n, i, 2 * k);      // Left subtree
            dst[k] = src[i++];                                // Current node
            i = build_recursive(src, dst, n, i, 2 * k + 1);  // Right subtree
        }
        return i;
    }
};

/// @brief Branchless binary search on Eytzinger layout with prefetching
///
/// This search strategy works on pre-transformed Eytzinger arrays.
/// It provides excellent performance for cold cache scenarios due to
/// its predictable memory access pattern and prefetching capability.
///
/// @tparam T Key type
/// @tparam PrefetchDepth How many tree levels ahead to prefetch (default: 3)
template<typename T, std::size_t PrefetchDepth = 3>
class EytzingerSearch {
public:
    static constexpr const char* name() noexcept {
        return "eytzinger_search";
    }

    /// @brief Optimal for large ranges with cold cache
    static constexpr std::size_t optimal_range_size() noexcept {
        return 512;
    }

    /// @brief Find first element >= key in Eytzinger array
    ///
    /// @note The array must be in Eytzinger layout (1-indexed, size n+1)
    ///       where n is the number of elements.
    ///
    /// @param eytz Eytzinger-layout array (1-indexed)
    /// @param n Number of elements (array size is n+1)
    /// @param key Key to search for
    /// @return Index in original sorted order (0 to n), or n if not found
    [[nodiscard]]
    std::size_t search(const T* eytz, std::size_t n, T key) const noexcept {
        std::size_t k = 1;

        while (k <= n) {
            // Prefetch ahead in the tree
            prefetch_ahead(eytz, k, n);

            // Branchless navigation: k = 2k + (eytz[k] < key)
            // If eytz[k] < key, go right (2k+1), else go left (2k)
            k = 2 * k + (eytz[k] < key ? 1 : 0);
        }

        // k is now at a "virtual" leaf position
        // Convert back to original sorted index
        // The position is encoded in the bit pattern of k

        // Find the first ancestor that went left (has a 0 bit after leading 1s)
        k >>= static_cast<std::size_t>(__builtin_ffs(static_cast<int>(~k)));

        // Handle edge case: key is larger than all elements
        return k == 0 ? n : k - 1;
    }

    /// @brief Lower bound on Eytzinger array, returns pointer for API compatibility
    ///
    /// @note For compatibility with SearchStrategy concept, but requires
    ///       the data to already be in Eytzinger layout.
    [[nodiscard]]
    const T* lower_bound(const T* eytz, std::size_t lo,
                         std::size_t hi, T key) const noexcept {
        // Note: lo is expected to be 0 for Eytzinger layout
        // hi is the number of elements (eytz array size is hi + 1)
        (void)lo;  // Unused for Eytzinger

        std::size_t idx = search(eytz, hi, key);

        // Convert Eytzinger index to pointer
        // Since Eytzinger is 1-indexed, we need to handle this carefully
        if (idx >= hi) {
            return eytz + hi + 1;  // Past end
        }
        return eytz + idx + 1;  // +1 because Eytzinger is 1-indexed
    }

private:
    /// @brief Prefetch nodes ahead in the tree
    static void prefetch_ahead(const T* eytz, std::size_t k,
                               [[maybe_unused]] std::size_t n) noexcept {
        if constexpr (PrefetchDepth > 0) {
            // Prefetch both possible paths at depth PrefetchDepth
            std::size_t prefetch_idx = k << PrefetchDepth;

            // Use __builtin_prefetch for GCC/Clang
#if defined(__GNUC__) || defined(__clang__)
            // Prefetch for read, low temporal locality
            __builtin_prefetch(eytz + prefetch_idx, 0, 0);
            __builtin_prefetch(eytz + prefetch_idx + (1 << (PrefetchDepth - 1)), 0, 0);
#endif
        }
    }
};

// Convenience type aliases
using EytzingerSearch64 = EytzingerSearch<std::uint64_t>;
using EytzingerSearch32 = EytzingerSearch<std::uint32_t>;

// Verify concept satisfaction
static_assert(SearchStrategy<EytzingerSearch<std::uint64_t>, std::uint64_t>);
static_assert(SearchStrategy<EytzingerSearch<std::uint32_t>, std::uint32_t>);

} // namespace simdex
