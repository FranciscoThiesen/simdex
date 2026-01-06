#pragma once

/// @file parallel_queries.hpp
/// @brief Query-parallel SIMD search - process multiple queries simultaneously
///
/// KEY INSIGHT: Instead of using SIMD lanes to compare multiple ELEMENTS for ONE query,
/// use SIMD lanes to process multiple QUERIES simultaneously.
///
/// Traditional (element-parallel):
///   Query key=100, compare against [data[0], data[1], data[2], data[3]] in parallel
///
/// New (query-parallel):
///   4 queries [k0, k1, k2, k3], each searches its own range
///   All do the same binary search step in lockstep
///   Use GATHER to load data[mid0], data[mid1], data[mid2], data[mid3]
///
/// This avoids the batch processing overhead (sorting, coalescing) while still
/// enabling parallelism across queries.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "simdex/simd/platform.hpp"

namespace simdex {

// ============================================================================
// AVX2 Implementation (4 queries in parallel for uint64_t)
// ============================================================================

#if defined(SIMDEX_HAS_AVX2)

/// @brief Process 4 binary searches in parallel using AVX2 gather
/// @param data Pointer to sorted data array
/// @param keys Array of 4 search keys
/// @param lo Array of 4 lower bounds
/// @param hi Array of 4 upper bounds (exclusive)
/// @param results Output array of 4 result positions
///
/// Uses fixed-iteration binary search to avoid divergence handling.
/// All 4 queries execute exactly the same number of iterations.
inline void parallel_lower_bound_4x64(
    const std::uint64_t* data,
    const std::uint64_t* keys,
    const std::size_t* lo,
    const std::size_t* hi,
    std::size_t* results
) noexcept {
    // Load keys and bounds into vectors
    __m256i v_keys = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(keys));
    __m256i v_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lo));
    __m256i v_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(hi));
    __m256i v_one = _mm256_set1_epi64x(1);

    // Fixed iterations: log2(max_range) iterations
    // For epsilon=128, max range = 256, so 8 iterations suffice
    // For epsilon=64, max range = 128, so 7 iterations suffice
    // We use 9 to be safe for epsilon up to 256
    #pragma unroll
    for (int iter = 0; iter < 9; ++iter) {
        // Compute mid = lo + (hi - lo) / 2 for each query
        // Using (lo + hi) / 2 can overflow, so we use lo + (hi - lo) / 2
        __m256i v_range = _mm256_sub_epi64(v_hi, v_lo);
        __m256i v_half = _mm256_srli_epi64(v_range, 1);
        __m256i v_mid = _mm256_add_epi64(v_lo, v_half);

        // GATHER: load data[mid[0]], data[mid[1]], data[mid[2]], data[mid[3]]
        // This is the key operation - fetches 4 potentially non-contiguous elements
        // Scale = 8 bytes per uint64_t
        __m256i v_data = _mm256_i64gather_epi64(
            reinterpret_cast<const long long*>(data),
            v_mid,
            8  // scale: sizeof(uint64_t)
        );

        // Compare: is key > data[mid]?
        // If yes, search right half (lo = mid + 1)
        // If no, search left half (hi = mid)
        __m256i v_cmp = _mm256_cmpgt_epi64(v_keys, v_data);

        // Compute mid + 1 for the "go right" case
        __m256i v_mid_plus_1 = _mm256_add_epi64(v_mid, v_one);

        // Update lo: lo = cmp ? mid + 1 : lo
        // Use blendv_pd because it operates on 64-bit granularity (via sign bit)
        v_lo = _mm256_castpd_si256(_mm256_blendv_pd(
            _mm256_castsi256_pd(v_lo),
            _mm256_castsi256_pd(v_mid_plus_1),
            _mm256_castsi256_pd(v_cmp)
        ));

        // Update hi: hi = cmp ? hi : mid
        v_hi = _mm256_castpd_si256(_mm256_blendv_pd(
            _mm256_castsi256_pd(v_mid),
            _mm256_castsi256_pd(v_hi),
            _mm256_castsi256_pd(v_cmp)
        ));
    }

    // lo now contains the lower_bound result for each query
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(results), v_lo);
}

/// @brief Process 8 binary searches in parallel for uint32_t
inline void parallel_lower_bound_8x32(
    const std::uint32_t* data,
    const std::uint32_t* keys,
    const std::size_t* lo,
    const std::size_t* hi,
    std::size_t* results
) noexcept {
    // For 32-bit, we can do 8 queries in parallel
    // But gather indices must be 32-bit, and we need 64-bit indices for large arrays
    // This is more complex - for now, fall back to 2x calls of 4-way

    // Convert size_t indices to int32 (assuming array < 2^31 elements)
    __m128i lo_low = _mm_set_epi32(
        static_cast<int>(lo[3]), static_cast<int>(lo[2]),
        static_cast<int>(lo[1]), static_cast<int>(lo[0])
    );
    __m128i lo_high = _mm_set_epi32(
        static_cast<int>(lo[7]), static_cast<int>(lo[6]),
        static_cast<int>(lo[5]), static_cast<int>(lo[4])
    );
    __m256i v_lo = _mm256_set_m128i(lo_high, lo_low);

    __m128i hi_low = _mm_set_epi32(
        static_cast<int>(hi[3]), static_cast<int>(hi[2]),
        static_cast<int>(hi[1]), static_cast<int>(hi[0])
    );
    __m128i hi_high = _mm_set_epi32(
        static_cast<int>(hi[7]), static_cast<int>(hi[6]),
        static_cast<int>(hi[5]), static_cast<int>(hi[4])
    );
    __m256i v_hi = _mm256_set_m128i(hi_high, hi_low);

    __m256i v_keys = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(keys));
    __m256i v_one = _mm256_set1_epi32(1);

    #pragma unroll
    for (int iter = 0; iter < 9; ++iter) {
        __m256i v_range = _mm256_sub_epi32(v_hi, v_lo);
        __m256i v_half = _mm256_srli_epi32(v_range, 1);
        __m256i v_mid = _mm256_add_epi32(v_lo, v_half);

        // Gather 8 x 32-bit values
        __m256i v_data = _mm256_i32gather_epi32(
            reinterpret_cast<const int*>(data),
            v_mid,
            4  // scale: sizeof(uint32_t)
        );

        __m256i v_cmp = _mm256_cmpgt_epi32(v_keys, v_data);
        __m256i v_mid_plus_1 = _mm256_add_epi32(v_mid, v_one);

        // blendv for 32-bit uses ps (single precision float)
        v_lo = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(v_lo),
            _mm256_castsi256_ps(v_mid_plus_1),
            _mm256_castsi256_ps(v_cmp)
        ));

        v_hi = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(v_mid),
            _mm256_castsi256_ps(v_hi),
            _mm256_castsi256_ps(v_cmp)
        ));
    }

    // Extract results back to size_t array
    alignas(32) int temp[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(temp), v_lo);
    for (int i = 0; i < 8; ++i) {
        results[i] = static_cast<std::size_t>(temp[i]);
    }
}

#endif // SIMDEX_HAS_AVX2

// ============================================================================
// AVX-512 Implementation (8 queries in parallel for uint64_t)
// ============================================================================

#if defined(SIMDEX_HAS_AVX512)

/// @brief Process 8 binary searches in parallel using AVX-512 gather
inline void parallel_lower_bound_8x64(
    const std::uint64_t* data,
    const std::uint64_t* keys,
    const std::size_t* lo,
    const std::size_t* hi,
    std::size_t* results
) noexcept {
    __m512i v_keys = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(keys));
    __m512i v_lo = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(lo));
    __m512i v_hi = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(hi));
    __m512i v_one = _mm512_set1_epi64(1);

    #pragma unroll
    for (int iter = 0; iter < 9; ++iter) {
        __m512i v_range = _mm512_sub_epi64(v_hi, v_lo);
        __m512i v_half = _mm512_srli_epi64(v_range, 1);
        __m512i v_mid = _mm512_add_epi64(v_lo, v_half);

        // AVX-512 gather with 64-bit indices
        __m512i v_data = _mm512_i64gather_epi64(v_mid, data, 8);

        // AVX-512 has mask-based operations, much cleaner
        __mmask8 cmp_mask = _mm512_cmpgt_epi64_mask(v_keys, v_data);

        __m512i v_mid_plus_1 = _mm512_add_epi64(v_mid, v_one);

        // Masked blend: lo = cmp ? mid+1 : lo
        v_lo = _mm512_mask_mov_epi64(v_lo, cmp_mask, v_mid_plus_1);

        // hi = cmp ? hi : mid (inverted mask)
        v_hi = _mm512_mask_mov_epi64(v_mid, cmp_mask, v_hi);
    }

    _mm512_storeu_si512(reinterpret_cast<__m512i*>(results), v_lo);
}

#endif // SIMDEX_HAS_AVX512

// ============================================================================
// Scalar fallback
// ============================================================================

/// @brief Scalar implementation for platforms without SIMD gather
template<typename T>
inline void parallel_lower_bound_scalar(
    const T* data,
    const T* keys,
    const std::size_t* lo,
    const std::size_t* hi,
    std::size_t* results,
    std::size_t num_queries
) noexcept {
    for (std::size_t q = 0; q < num_queries; ++q) {
        std::size_t l = lo[q];
        std::size_t h = hi[q];
        T key = keys[q];

        while (l < h) {
            std::size_t mid = l + (h - l) / 2;
            if (data[mid] < key) {
                l = mid + 1;
            } else {
                h = mid;
            }
        }
        results[q] = l;
    }
}

// ============================================================================
// High-level batch processor using query-parallel SIMD
// ============================================================================

/// @brief Query-parallel batch processor
/// Processes queries in groups of 4 (AVX2) or 8 (AVX-512) using SIMD gather
template<typename Index>
class QueryParallelProcessor {
public:
    using key_type = typename Index::key_type;
    using size_type = std::size_t;

    explicit QueryParallelProcessor(const Index& index) : index_(index) {}

    /// @brief Process batch of queries using query-parallel SIMD
    /// @param keys Vector of search keys
    /// @return Vector of result positions (lower_bound for each key)
    std::vector<size_type> process(const std::vector<key_type>& keys) const {
        if (keys.empty()) return {};

        const size_type n = keys.size();
        std::vector<size_type> results(n);

        // Get predictions for all queries first
        std::vector<size_type> lo_bounds(n);
        std::vector<size_type> hi_bounds(n);

        for (size_type i = 0; i < n; ++i) {
            auto approx = index_.get_approx_pos(keys[i]);
            lo_bounds[i] = approx.lo;
            hi_bounds[i] = std::min(approx.hi, index_.size());
        }

        // Process in SIMD-width chunks
        process_chunks(keys.data(), lo_bounds.data(), hi_bounds.data(),
                      results.data(), n);

        return results;
    }

    /// @brief Process batch and return found flags
    std::pair<std::vector<size_type>, std::vector<bool>> process_with_flags(
        const std::vector<key_type>& keys
    ) const {
        auto positions = process(keys);
        std::vector<bool> found(keys.size());

        for (size_type i = 0; i < keys.size(); ++i) {
            found[i] = (positions[i] < index_.size() &&
                       index_.data()[positions[i]] == keys[i]);
        }

        return {std::move(positions), std::move(found)};
    }

private:
    const Index& index_;

    void process_chunks(
        const key_type* keys,
        const size_type* lo,
        const size_type* hi,
        size_type* results,
        size_type n
    ) const noexcept {
        size_type i = 0;

#if defined(SIMDEX_HAS_AVX512)
        // Process 8 queries at a time with AVX-512
        if constexpr (sizeof(key_type) == 8) {
            for (; i + 8 <= n; i += 8) {
                parallel_lower_bound_8x64(
                    index_.data(),
                    keys + i,
                    lo + i,
                    hi + i,
                    results + i
                );
            }
        }
#endif

#if defined(SIMDEX_HAS_AVX2)
        // Process 4 queries at a time with AVX2
        if constexpr (sizeof(key_type) == 8) {
            for (; i + 4 <= n; i += 4) {
                parallel_lower_bound_4x64(
                    index_.data(),
                    keys + i,
                    lo + i,
                    hi + i,
                    results + i
                );
            }
        } else if constexpr (sizeof(key_type) == 4) {
            for (; i + 8 <= n; i += 8) {
                parallel_lower_bound_8x32(
                    index_.data(),
                    keys + i,
                    lo + i,
                    hi + i,
                    results + i
                );
            }
        }
#endif

        // Scalar fallback for remaining queries
        if (i < n) {
            parallel_lower_bound_scalar(
                index_.data(),
                keys + i,
                lo + i,
                hi + i,
                results + i,
                n - i
            );
        }
    }
};

/// @brief Create a query-parallel processor for an index
template<typename Index>
QueryParallelProcessor<Index> make_query_parallel_processor(const Index& index) {
    return QueryParallelProcessor<Index>(index);
}

} // namespace simdex
