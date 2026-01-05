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
        if (n == 0) return 0;

        std::size_t k = 1;
        std::size_t answer_k = 0;  // Best Eytzinger index found (>= key)

        while (k <= n) {
            // Prefetch ahead in the tree
            prefetch_ahead(eytz, k, n);

            if (eytz[k] >= key) {
                answer_k = k;       // This could be our answer
                k = 2 * k;          // Look left for smaller or equal values
            } else {
                k = 2 * k + 1;      // Need larger values, go right
            }
        }

        if (answer_k == 0) {
            return n;  // Not found, all values < key
        }

        // Convert Eytzinger index to sorted index (in-order rank)
        return eytz_to_sorted(answer_k, n);
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

        if (idx >= hi) {
            return eytz + hi + 1;  // Past end
        }
        // Find the Eytzinger index for this sorted position
        // For lower_bound, we return pointer to the value we found
        return eytz + sorted_to_eytz(idx, hi);
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

    /// @brief Compute subtree size for node k in tree of size n
    static std::size_t subtree_size(std::size_t k, std::size_t n) noexcept {
        if (k > n) return 0;

        // Find height of tree: floor(log2(n))
        std::size_t height = 0;
        for (std::size_t x = n; x > 0; x >>= 1) height++;
        height--;

        // Level of node k: floor(log2(k))
        std::size_t level_k = 0;
        for (std::size_t x = k; x > 0; x >>= 1) level_k++;
        level_k--;

        // Levels below k
        std::size_t levels_below = height - level_k;

        // Leftmost and rightmost potential leaves in subtree
        std::size_t leftmost = k << levels_below;
        std::size_t rightmost = ((k + 1) << levels_below) - 1;

        // Nodes above the last level in subtree: 2^levels_below - 1
        std::size_t full_nodes = (1ULL << levels_below) - 1;

        // Count nodes at the last level that exist
        std::size_t last_level_start = 1ULL << height;
        std::size_t last_level_end = n;

        // Intersection with [leftmost, rightmost]
        if (leftmost > last_level_end || rightmost < last_level_start) {
            // No nodes at last level in this subtree
            return full_nodes;
        }

        std::size_t left = (leftmost > last_level_start) ? leftmost : last_level_start;
        std::size_t right = (rightmost < last_level_end) ? rightmost : last_level_end;
        std::size_t last_level_count = right - left + 1;

        return full_nodes + last_level_count;
    }

    /// @brief Convert Eytzinger index to sorted index (in-order rank)
    static std::size_t eytz_to_sorted(std::size_t k, std::size_t n) noexcept {
        if (k > n || k == 0) return n;

        // Rank = left subtree size + contributions from ancestors
        std::size_t rank = subtree_size(2 * k, n);  // Left subtree comes before k

        // Walk up to root, adding contributions from right-child relationships
        std::size_t node = k;
        while (node > 1) {
            std::size_t parent = node / 2;
            if (node == 2 * parent + 1) {
                // node is right child of parent
                // Parent and parent's left subtree come before node
                rank += 1 + subtree_size(2 * parent, n);
            }
            node = parent;
        }

        return rank;
    }

    /// @brief Convert sorted index to Eytzinger index (for lower_bound pointer)
    static std::size_t sorted_to_eytz(std::size_t sorted_idx, std::size_t n) noexcept {
        // Do in-order traversal to find the k-th element
        std::size_t count = 0;
        std::size_t k = 1;

        while (k <= n) {
            std::size_t left_size = subtree_size(2 * k, n);
            if (count + left_size == sorted_idx) {
                return k;  // Found it
            } else if (count + left_size < sorted_idx) {
                // Target is in right subtree
                count += left_size + 1;
                k = 2 * k + 1;
            } else {
                // Target is in left subtree
                k = 2 * k;
            }
        }

        return 1;  // Fallback (shouldn't reach here for valid input)
    }
};

// Convenience type aliases
using EytzingerSearch64 = EytzingerSearch<std::uint64_t>;
using EytzingerSearch32 = EytzingerSearch<std::uint32_t>;

// Verify concept satisfaction
static_assert(SearchStrategy<EytzingerSearch<std::uint64_t>, std::uint64_t>);
static_assert(SearchStrategy<EytzingerSearch<std::uint32_t>, std::uint32_t>);

} // namespace simdex
