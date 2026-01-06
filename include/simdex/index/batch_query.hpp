#pragma once

/// @file batch_query.hpp
/// @brief Batched query processing for learned indexes
///
/// This is the core novel contribution: instead of processing queries one at a time,
/// we batch them together to enable:
/// 1. Query reordering by predicted position (improves cache locality)
/// 2. Segment coalescing (queries in same region share memory loads)
/// 3. Prefetching based on predictions (hide memory latency)
/// 4. Vectorized result gathering
///
/// Key insight: Learned indexes give us predictions BEFORE we access data.
/// We can use these predictions to optimize the access pattern across a batch.

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>
#include <utility>

#include "simdex/simd/platform.hpp"

namespace simdex {

/// @brief Strategy for batched query processing
enum class BatchStrategy {
    /// Process queries sequentially (baseline for comparison)
    Sequential,

    /// Sort queries by predicted position before searching
    /// Improves cache locality for queries with spatial correlation
    SortedPrediction,

    /// Group queries that fall in overlapping search regions
    /// Enables sharing of memory loads across multiple queries
    SegmentCoalescing,

    /// Prefetch next region while searching current
    /// Hides memory latency for cache-cold scenarios
    Prefetched,

    /// Full optimization: sort + coalesce + prefetch
    FullyOptimized
};

/// @brief Result of a batched query operation
/// @tparam K Key type
template<typename K>
struct BatchResult {
    std::vector<std::size_t> positions;  // Result positions in original data
    std::vector<bool> found;              // Whether each query found exact match

    // Statistics for analysis
    std::size_t cache_hits = 0;           // Queries that shared a loaded region
    std::size_t segments_loaded = 0;      // Number of distinct segments accessed
    std::size_t total_elements_scanned = 0;
};

/// @brief Query with its prediction metadata
template<typename K>
struct PredictedQuery {
    K key;                      // The search key
    std::size_t original_idx;   // Position in original query array
    std::size_t predicted_pos;  // Model's predicted position
    std::size_t lo;             // Search range lower bound
    std::size_t hi;             // Search range upper bound

    // For sorting by predicted position
    bool operator<(const PredictedQuery& other) const noexcept {
        return predicted_pos < other.predicted_pos;
    }
};

/// @brief Segment coalescing helper
/// Groups queries whose search ranges overlap
template<typename K>
class SegmentCoalescer {
public:
    struct CoalescedGroup {
        std::size_t merged_lo;   // Merged search range start
        std::size_t merged_hi;   // Merged search range end
        std::vector<std::size_t> query_indices;  // Queries in this group
    };

    /// @brief Coalesce queries with overlapping search ranges
    /// @param queries Sorted array of predicted queries
    /// @param max_gap Maximum gap between ranges to merge (0 = must overlap)
    /// @return Vector of coalesced groups
    static std::vector<CoalescedGroup> coalesce(
        const std::vector<PredictedQuery<K>>& queries,
        std::size_t max_gap = 0
    ) {
        if (queries.empty()) return {};

        std::vector<CoalescedGroup> groups;
        CoalescedGroup current;
        current.merged_lo = queries[0].lo;
        current.merged_hi = queries[0].hi;
        current.query_indices.push_back(0);

        for (std::size_t i = 1; i < queries.size(); ++i) {
            const auto& q = queries[i];

            // Check if this query's range overlaps or is close to current group
            bool should_merge = (q.lo <= current.merged_hi + max_gap);

            if (should_merge) {
                // Extend the merged range
                current.merged_hi = std::max(current.merged_hi, q.hi);
                current.query_indices.push_back(i);
            } else {
                // Start new group
                groups.push_back(std::move(current));
                current = CoalescedGroup{};
                current.merged_lo = q.lo;
                current.merged_hi = q.hi;
                current.query_indices.push_back(i);
            }
        }

        groups.push_back(std::move(current));
        return groups;
    }
};

/// @brief Multi-key SIMD search within a loaded region
/// Searches for multiple keys in a contiguous memory region
template<typename T, typename Platform = simd::auto_platform_tag>
class MultiKeySearch {
public:
    using platform_type = simd::resolve_platform_t<Platform>;
    using traits = simd::vec_traits<T, platform_type>;
    static constexpr std::size_t lanes = traits::lanes;

    /// @brief Search for multiple keys in a region
    /// @param data Pointer to sorted data
    /// @param region_lo Start of region
    /// @param region_hi End of region
    /// @param keys Keys to search for (must be sorted)
    /// @param results Output: position of each key (or hi if not found)
    static void multi_search(
        const T* data,
        std::size_t region_lo,
        std::size_t region_hi,
        const std::vector<T>& keys,
        std::vector<std::size_t>& results
    ) noexcept {
        results.resize(keys.size());

        if (keys.empty() || region_lo >= region_hi) {
            std::fill(results.begin(), results.end(), region_hi);
            return;
        }

        // For small regions or few keys, use simple approach
        if (region_hi - region_lo <= lanes * 2 || keys.size() <= 2) {
            for (std::size_t k = 0; k < keys.size(); ++k) {
                results[k] = scalar_lower_bound(data, region_lo, region_hi, keys[k]);
            }
            return;
        }

        // Optimized: scan region once, find all keys
        // Since keys are sorted, we can track progress through keys as we scan data
        std::size_t key_idx = 0;
        std::size_t data_idx = region_lo;

        // SIMD scan with multiple key tracking
        while (data_idx + lanes <= region_hi && key_idx < keys.size()) {
            auto chunk = traits::load(data + data_idx);

            // Check each pending key against this chunk
            while (key_idx < keys.size()) {
                T key = keys[key_idx];

                // If key is less than first element of chunk, it's before this chunk
                if (key <= data[data_idx]) {
                    results[key_idx] = data_idx;
                    key_idx++;
                    continue;
                }

                // If key is greater than last element of chunk, move to next chunk
                if (key > data[data_idx + lanes - 1]) {
                    break;  // Process next chunk
                }

                // Key is within this chunk - find exact position
                auto needle = traits::broadcast(key);
                auto cmp = traits::cmpgt(needle, chunk);
                int mask = traits::movemask(cmp);

                constexpr int all_lanes_mask = (1 << lanes) - 1;
                int found_mask = (~mask) & all_lanes_mask;

                if (found_mask != 0) {
                    int pos = simd::ctz(found_mask);
                    results[key_idx] = data_idx + pos;
                } else {
                    results[key_idx] = data_idx + lanes;
                }
                key_idx++;
            }

            data_idx += lanes;
        }

        // Handle remaining keys with scalar search
        for (; key_idx < keys.size(); ++key_idx) {
            results[key_idx] = scalar_lower_bound(data, data_idx, region_hi, keys[key_idx]);
        }
    }

private:
    static std::size_t scalar_lower_bound(const T* data, std::size_t lo,
                                           std::size_t hi, T key) noexcept {
        while (lo < hi) {
            std::size_t mid = lo + (hi - lo) / 2;
            if (data[mid] < key) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }
};

/// @brief Batched query processor for learned indexes
/// @tparam Index The learned index type (must have search() method returning ApproxPos)
template<typename Index>
class BatchProcessor {
public:
    using key_type = typename Index::key_type;
    using size_type = std::size_t;

    explicit BatchProcessor(const Index& index) : index_(index) {}

    /// @brief Process a batch of queries
    /// @param queries Vector of keys to search for
    /// @param strategy Optimization strategy to use
    /// @return BatchResult with positions and statistics
    BatchResult<key_type> process(
        const std::vector<key_type>& queries,
        BatchStrategy strategy = BatchStrategy::FullyOptimized
    ) const {
        switch (strategy) {
            case BatchStrategy::Sequential:
                return process_sequential(queries);
            case BatchStrategy::SortedPrediction:
                return process_sorted(queries);
            case BatchStrategy::SegmentCoalescing:
                return process_coalesced(queries);
            case BatchStrategy::Prefetched:
                return process_prefetched(queries);
            case BatchStrategy::FullyOptimized:
            default:
                return process_fully_optimized(queries);
        }
    }

private:
    const Index& index_;

    /// @brief Sequential processing (baseline)
    BatchResult<key_type> process_sequential(const std::vector<key_type>& queries) const {
        BatchResult<key_type> result;
        result.positions.reserve(queries.size());
        result.found.reserve(queries.size());
        result.segments_loaded = queries.size();  // One segment per query

        for (const auto& key : queries) {
            auto it = index_.lower_bound(key);
            std::size_t pos = it - index_.begin();
            result.positions.push_back(pos);
            result.found.push_back(it != index_.end() && *it == key);
            result.total_elements_scanned += 2 * Index::epsilon();
        }

        return result;
    }

    /// @brief Sorted by prediction processing
    BatchResult<key_type> process_sorted(const std::vector<key_type>& queries) const {
        // Step 1: Get predictions for all queries
        std::vector<PredictedQuery<key_type>> predicted;
        predicted.reserve(queries.size());

        for (std::size_t i = 0; i < queries.size(); ++i) {
            auto approx = index_.get_approx_pos(queries[i]);
            predicted.push_back({
                queries[i],
                i,
                (approx.lo + approx.hi) / 2,
                approx.lo,
                std::min(approx.hi, index_.size())
            });
        }

        // Step 2: Sort by predicted position
        std::sort(predicted.begin(), predicted.end());

        // Step 3: Process in sorted order
        BatchResult<key_type> result;
        result.positions.resize(queries.size());
        result.found.resize(queries.size());
        result.segments_loaded = queries.size();

        for (const auto& pq : predicted) {
            auto it = index_.lower_bound(pq.key);
            std::size_t pos = it - index_.begin();
            result.positions[pq.original_idx] = pos;
            result.found[pq.original_idx] = (it != index_.end() && *it == pq.key);
            result.total_elements_scanned += pq.hi - pq.lo;
        }

        return result;
    }

    /// @brief Coalesced segment processing
    BatchResult<key_type> process_coalesced(const std::vector<key_type>& queries) const {
        // Step 1: Get predictions
        std::vector<PredictedQuery<key_type>> predicted;
        predicted.reserve(queries.size());

        for (std::size_t i = 0; i < queries.size(); ++i) {
            auto approx = index_.get_approx_pos(queries[i]);
            predicted.push_back({
                queries[i],
                i,
                (approx.lo + approx.hi) / 2,
                approx.lo,
                std::min(approx.hi, index_.size())
            });
        }

        // Step 2: Sort by predicted position
        std::sort(predicted.begin(), predicted.end());

        // Step 3: Coalesce overlapping regions
        auto groups = SegmentCoalescer<key_type>::coalesce(predicted, 0);

        // Step 4: Process each coalesced group
        BatchResult<key_type> result;
        result.positions.resize(queries.size());
        result.found.resize(queries.size());
        result.segments_loaded = groups.size();

        MultiKeySearch<key_type> searcher;

        for (const auto& group : groups) {
            // Collect keys for this group (in sorted order)
            std::vector<key_type> group_keys;
            std::vector<std::size_t> group_original_indices;
            group_keys.reserve(group.query_indices.size());
            group_original_indices.reserve(group.query_indices.size());

            for (std::size_t idx : group.query_indices) {
                group_keys.push_back(predicted[idx].key);
                group_original_indices.push_back(predicted[idx].original_idx);
            }

            // Sort keys within group for efficient multi-search
            std::vector<std::size_t> sort_perm(group_keys.size());
            std::iota(sort_perm.begin(), sort_perm.end(), 0);
            std::sort(sort_perm.begin(), sort_perm.end(),
                [&](std::size_t a, std::size_t b) {
                    return group_keys[a] < group_keys[b];
                });

            std::vector<key_type> sorted_keys(group_keys.size());
            for (std::size_t i = 0; i < sort_perm.size(); ++i) {
                sorted_keys[i] = group_keys[sort_perm[i]];
            }

            // Multi-search within merged region
            std::vector<std::size_t> positions;
            MultiKeySearch<key_type>::multi_search(
                index_.data(),
                group.merged_lo,
                group.merged_hi,
                sorted_keys,
                positions
            );

            // Store results back in original order
            for (std::size_t i = 0; i < sort_perm.size(); ++i) {
                std::size_t orig_idx = group_original_indices[sort_perm[i]];
                std::size_t pos = positions[i];
                result.positions[orig_idx] = pos;
                result.found[orig_idx] = (pos < index_.size() &&
                                          index_.data()[pos] == sorted_keys[i]);
            }

            // Track cache efficiency
            if (group.query_indices.size() > 1) {
                result.cache_hits += group.query_indices.size() - 1;
            }
            result.total_elements_scanned += group.merged_hi - group.merged_lo;
        }

        return result;
    }

    /// @brief Prefetched processing
    BatchResult<key_type> process_prefetched(const std::vector<key_type>& queries) const {
        // Step 1: Get all predictions upfront
        std::vector<PredictedQuery<key_type>> predicted;
        predicted.reserve(queries.size());

        for (std::size_t i = 0; i < queries.size(); ++i) {
            auto approx = index_.get_approx_pos(queries[i]);
            predicted.push_back({
                queries[i],
                i,
                (approx.lo + approx.hi) / 2,
                approx.lo,
                std::min(approx.hi, index_.size())
            });
        }

        // Sort for better prefetching
        std::sort(predicted.begin(), predicted.end());

        // Step 2: Process with prefetching
        BatchResult<key_type> result;
        result.positions.resize(queries.size());
        result.found.resize(queries.size());
        result.segments_loaded = queries.size();

        const std::size_t prefetch_distance = 4;  // Look ahead 4 queries

        for (std::size_t i = 0; i < predicted.size(); ++i) {
            // Prefetch future queries' data regions
            if (i + prefetch_distance < predicted.size()) {
                const auto& future = predicted[i + prefetch_distance];
#if defined(__GNUC__) || defined(__clang__)
                __builtin_prefetch(index_.data() + future.predicted_pos, 0, 0);
#endif
            }

            // Process current query
            const auto& pq = predicted[i];
            auto it = index_.lower_bound(pq.key);
            std::size_t pos = it - index_.begin();
            result.positions[pq.original_idx] = pos;
            result.found[pq.original_idx] = (it != index_.end() && *it == pq.key);
            result.total_elements_scanned += pq.hi - pq.lo;
        }

        return result;
    }

    /// @brief Fully optimized: sort + coalesce + prefetch
    BatchResult<key_type> process_fully_optimized(const std::vector<key_type>& queries) const {
        if (queries.size() < 16) {
            // Small batch: coalescing overhead not worth it
            return process_sorted(queries);
        }

        // For larger batches, use full optimization
        // Step 1: Get predictions
        std::vector<PredictedQuery<key_type>> predicted;
        predicted.reserve(queries.size());

        for (std::size_t i = 0; i < queries.size(); ++i) {
            auto approx = index_.get_approx_pos(queries[i]);
            predicted.push_back({
                queries[i],
                i,
                (approx.lo + approx.hi) / 2,
                approx.lo,
                std::min(approx.hi, index_.size())
            });
        }

        // Step 2: Sort by predicted position
        std::sort(predicted.begin(), predicted.end());

        // Step 3: Coalesce with small gap allowance (cache line size / element size)
        constexpr std::size_t cache_line_elements = 64 / sizeof(key_type);
        auto groups = SegmentCoalescer<key_type>::coalesce(predicted, cache_line_elements);

        // Step 4: Process groups with prefetching
        BatchResult<key_type> result;
        result.positions.resize(queries.size());
        result.found.resize(queries.size());
        result.segments_loaded = groups.size();

        const std::size_t prefetch_distance = 2;  // Prefetch 2 groups ahead

        for (std::size_t g = 0; g < groups.size(); ++g) {
            // Prefetch future groups
            if (g + prefetch_distance < groups.size()) {
                const auto& future_group = groups[g + prefetch_distance];
#if defined(__GNUC__) || defined(__clang__)
                __builtin_prefetch(index_.data() + future_group.merged_lo, 0, 0);
#endif
            }

            const auto& group = groups[g];

            // Collect and sort keys within group
            std::vector<key_type> group_keys;
            std::vector<std::size_t> group_original_indices;
            group_keys.reserve(group.query_indices.size());
            group_original_indices.reserve(group.query_indices.size());

            for (std::size_t idx : group.query_indices) {
                group_keys.push_back(predicted[idx].key);
                group_original_indices.push_back(predicted[idx].original_idx);
            }

            std::vector<std::size_t> sort_perm(group_keys.size());
            std::iota(sort_perm.begin(), sort_perm.end(), 0);
            std::sort(sort_perm.begin(), sort_perm.end(),
                [&](std::size_t a, std::size_t b) {
                    return group_keys[a] < group_keys[b];
                });

            std::vector<key_type> sorted_keys(group_keys.size());
            for (std::size_t i = 0; i < sort_perm.size(); ++i) {
                sorted_keys[i] = group_keys[sort_perm[i]];
            }

            // Multi-search within merged region
            std::vector<std::size_t> positions;
            MultiKeySearch<key_type>::multi_search(
                index_.data(),
                group.merged_lo,
                group.merged_hi,
                sorted_keys,
                positions
            );

            // Store results
            for (std::size_t i = 0; i < sort_perm.size(); ++i) {
                std::size_t orig_idx = group_original_indices[sort_perm[i]];
                std::size_t pos = positions[i];
                result.positions[orig_idx] = pos;
                result.found[orig_idx] = (pos < index_.size() &&
                                          index_.data()[pos] == sorted_keys[i]);
            }

            if (group.query_indices.size() > 1) {
                result.cache_hits += group.query_indices.size() - 1;
            }
            result.total_elements_scanned += group.merged_hi - group.merged_lo;
        }

        return result;
    }
};

/// @brief Create a batch processor for an index
template<typename Index>
BatchProcessor<Index> make_batch_processor(const Index& index) {
    return BatchProcessor<Index>(index);
}

} // namespace simdex
