#pragma once

/// @file vec_traits.hpp
/// @brief SIMD vector traits abstraction layer
///
/// This file provides a unified interface for SIMD operations across
/// different platforms (AVX2, NEON). Each platform specializes the
/// vec_traits template with platform-specific intrinsics.

#include <cstddef>
#include <cstdint>

#include "simdex/simd/platform.hpp"

namespace simdex::simd {

/// @brief Primary template for SIMD vector traits
///
/// Specializations must provide:
/// - vector_type: The native SIMD vector type
/// - lanes: Number of elements per vector
/// - alignment: Required memory alignment in bytes
/// - load(ptr): Unaligned load from memory
/// - load_aligned(ptr): Aligned load from memory
/// - broadcast(val): Broadcast scalar to all lanes
/// - cmpgt(a, b): Compare greater than (returns mask vector)
/// - movemask(v): Extract comparison results as bitmask
template<typename T, typename Platform>
struct vec_traits;

// ============================================================================
// Scalar fallback implementation
// ============================================================================

template<typename T>
struct vec_traits<T, scalar_tag> {
    // "Vector" is just a single scalar
    using vector_type = T;

    static constexpr std::size_t lanes = 1;
    static constexpr std::size_t alignment = alignof(T);

    static vector_type load(const T* ptr) noexcept {
        return *ptr;
    }

    static vector_type load_aligned(const T* ptr) noexcept {
        return *ptr;
    }

    static vector_type broadcast(T val) noexcept {
        return val;
    }

    static vector_type cmpgt(vector_type a, vector_type b) noexcept {
        // Return all 1s if a > b, else 0
        return a > b ? static_cast<T>(~T{0}) : T{0};
    }

    static int movemask(vector_type v) noexcept {
        // Non-zero value means the comparison was true
        return v != 0 ? 1 : 0;
    }
};

// ============================================================================
// AVX2 implementation (256-bit vectors)
// ============================================================================

#if defined(SIMDEX_HAS_AVX2)

/// @brief AVX2 traits for 64-bit unsigned integers
template<>
struct vec_traits<std::uint64_t, avx2_tag> {
    using vector_type = __m256i;

    static constexpr std::size_t lanes = 4;  // 256 bits / 64 bits
    static constexpr std::size_t alignment = 32;

    /// @brief Unaligned load of 4 x uint64_t
    static vector_type load(const std::uint64_t* ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    /// @brief Aligned load (ptr must be 32-byte aligned)
    static vector_type load_aligned(const std::uint64_t* ptr) noexcept {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    /// @brief Broadcast single value to all 4 lanes
    static vector_type broadcast(std::uint64_t val) noexcept {
        return _mm256_set1_epi64x(static_cast<std::int64_t>(val));
    }

    /// @brief Compare a > b (signed comparison)
    /// @note For unsigned comparison of values < 2^63, this works correctly
    static vector_type cmpgt(vector_type a, vector_type b) noexcept {
        return _mm256_cmpgt_epi64(a, b);
    }

    /// @brief Extract sign bits as 4-bit mask
    /// @return Bitmask where bit i is set if lane i comparison was true
    static int movemask(vector_type v) noexcept {
        // Cast to double and use movemask_pd to get sign bits
        return _mm256_movemask_pd(_mm256_castsi256_pd(v));
    }
};

/// @brief AVX2 traits for 32-bit unsigned integers
template<>
struct vec_traits<std::uint32_t, avx2_tag> {
    using vector_type = __m256i;

    static constexpr std::size_t lanes = 8;  // 256 bits / 32 bits
    static constexpr std::size_t alignment = 32;

    static vector_type load(const std::uint32_t* ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    static vector_type load_aligned(const std::uint32_t* ptr) noexcept {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    static vector_type broadcast(std::uint32_t val) noexcept {
        return _mm256_set1_epi32(static_cast<std::int32_t>(val));
    }

    static vector_type cmpgt(vector_type a, vector_type b) noexcept {
        return _mm256_cmpgt_epi32(a, b);
    }

    static int movemask(vector_type v) noexcept {
        // Use movemask_ps for 32-bit lanes (gets all 8 sign bits)
        return _mm256_movemask_ps(_mm256_castsi256_ps(v));
    }
};

#endif // SIMDEX_HAS_AVX2

// ============================================================================
// AVX-512 implementation (512-bit vectors)
// ============================================================================

#if defined(SIMDEX_HAS_AVX512)

/// @brief AVX-512 traits for 64-bit unsigned integers
/// 8 lanes for uint64 - double the parallelism of AVX2
template<>
struct vec_traits<std::uint64_t, avx512_tag> {
    using vector_type = __m512i;

    static constexpr std::size_t lanes = 8;  // 512 bits / 64 bits
    static constexpr std::size_t alignment = 64;

    /// @brief Unaligned load of 8 x uint64_t
    static vector_type load(const std::uint64_t* ptr) noexcept {
        return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
    }

    /// @brief Aligned load (ptr must be 64-byte aligned)
    static vector_type load_aligned(const std::uint64_t* ptr) noexcept {
        return _mm512_load_si512(reinterpret_cast<const __m512i*>(ptr));
    }

    /// @brief Broadcast single value to all 8 lanes
    static vector_type broadcast(std::uint64_t val) noexcept {
        return _mm512_set1_epi64(static_cast<std::int64_t>(val));
    }

    /// @brief Compare a > b using AVX-512 mask comparison
    /// Returns a vector with all 1s in lanes where a > b
    static vector_type cmpgt(vector_type a, vector_type b) noexcept {
        // AVX-512 has native mask comparison, but we need a vector result
        // for compatibility with our movemask interface
        __mmask8 mask = _mm512_cmpgt_epi64_mask(a, b);
        return _mm512_maskz_set1_epi64(mask, ~0ULL);
    }

    /// @brief Extract comparison results as 8-bit mask
    /// More efficient than AVX2 because AVX-512 has native mask support
    static int movemask(vector_type v) noexcept {
        // Convert to mask by comparing with zero
        __mmask8 mask = _mm512_test_epi64_mask(v, v);
        return static_cast<int>(mask);
    }

    /// @brief Direct mask comparison (AVX-512 native operation)
    /// More efficient than cmpgt + movemask for search operations
    static int cmpgt_mask(vector_type a, vector_type b) noexcept {
        return static_cast<int>(_mm512_cmpgt_epi64_mask(a, b));
    }

    /// @brief Compare less-than-or-equal (for lower_bound)
    static int cmple_mask(vector_type a, vector_type b) noexcept {
        return static_cast<int>(_mm512_cmple_epi64_mask(a, b));
    }
};

/// @brief AVX-512 traits for 32-bit unsigned integers
/// 16 lanes for uint32 - 4x more parallelism than AVX2 for 32-bit
template<>
struct vec_traits<std::uint32_t, avx512_tag> {
    using vector_type = __m512i;

    static constexpr std::size_t lanes = 16;  // 512 bits / 32 bits
    static constexpr std::size_t alignment = 64;

    static vector_type load(const std::uint32_t* ptr) noexcept {
        return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
    }

    static vector_type load_aligned(const std::uint32_t* ptr) noexcept {
        return _mm512_load_si512(reinterpret_cast<const __m512i*>(ptr));
    }

    static vector_type broadcast(std::uint32_t val) noexcept {
        return _mm512_set1_epi32(static_cast<std::int32_t>(val));
    }

    static vector_type cmpgt(vector_type a, vector_type b) noexcept {
        __mmask16 mask = _mm512_cmpgt_epi32_mask(a, b);
        return _mm512_maskz_set1_epi32(mask, ~0U);
    }

    static int movemask(vector_type v) noexcept {
        __mmask16 mask = _mm512_test_epi32_mask(v, v);
        return static_cast<int>(mask);
    }

    static int cmpgt_mask(vector_type a, vector_type b) noexcept {
        return static_cast<int>(_mm512_cmpgt_epi32_mask(a, b));
    }

    static int cmple_mask(vector_type a, vector_type b) noexcept {
        return static_cast<int>(_mm512_cmple_epi32_mask(a, b));
    }
};

#endif // SIMDEX_HAS_AVX512

// ============================================================================
// ARM NEON implementation (128-bit vectors)
// ============================================================================

#if defined(SIMDEX_HAS_NEON)

/// @brief NEON traits for 64-bit unsigned integers
template<>
struct vec_traits<std::uint64_t, neon_tag> {
    using vector_type = uint64x2_t;

    static constexpr std::size_t lanes = 2;  // 128 bits / 64 bits
    static constexpr std::size_t alignment = 16;

    static vector_type load(const std::uint64_t* ptr) noexcept {
        return vld1q_u64(ptr);
    }

    static vector_type load_aligned(const std::uint64_t* ptr) noexcept {
        return vld1q_u64(ptr);
    }

    static vector_type broadcast(std::uint64_t val) noexcept {
        return vdupq_n_u64(val);
    }

    /// @brief Compare a > b (unsigned comparison)
    static vector_type cmpgt(vector_type a, vector_type b) noexcept {
        return vcgtq_u64(a, b);
    }

    /// @brief Optimized movemask for NEON using addv
    /// @return 2-bit mask where bit i is set if lane i is non-zero
    static int movemask(vector_type v) noexcept {
        // Shift to get sign bits in low position, narrow, then extract
        // For 2 lanes, this is simpler - just extract and test
        // Using vget_lane is unavoidable but we minimize operations
        uint64_t lane0 = vgetq_lane_u64(v, 0);
        uint64_t lane1 = vgetq_lane_u64(v, 1);
        // Use arithmetic right shift behavior: all 1s -> non-zero, all 0s -> zero
        return (lane0 != 0 ? 1 : 0) | (lane1 != 0 ? 2 : 0);
    }

    /// @brief Find first set lane (optimized for small lane count)
    /// @return Index of first non-zero lane, or lanes if none
    static int find_first_set(vector_type v) noexcept {
        if (vgetq_lane_u64(v, 0) != 0) return 0;
        if (vgetq_lane_u64(v, 1) != 0) return 1;
        return 2;
    }
};

/// @brief NEON traits for 32-bit unsigned integers
template<>
struct vec_traits<std::uint32_t, neon_tag> {
    using vector_type = uint32x4_t;

    static constexpr std::size_t lanes = 4;  // 128 bits / 32 bits
    static constexpr std::size_t alignment = 16;

    static vector_type load(const std::uint32_t* ptr) noexcept {
        return vld1q_u32(ptr);
    }

    static vector_type load_aligned(const std::uint32_t* ptr) noexcept {
        return vld1q_u32(ptr);
    }

    static vector_type broadcast(std::uint32_t val) noexcept {
        return vdupq_n_u32(val);
    }

    static vector_type cmpgt(vector_type a, vector_type b) noexcept {
        return vcgtq_u32(a, b);
    }

    /// @brief Optimized movemask for 32-bit NEON
    /// @return 4-bit mask where bit i is set if lane i is non-zero
    static int movemask(vector_type v) noexcept {
        // Direct lane extraction - clearer and often faster than narrowing tricks
        int mask = 0;
        if (vgetq_lane_u32(v, 0) != 0) mask |= 1;
        if (vgetq_lane_u32(v, 1) != 0) mask |= 2;
        if (vgetq_lane_u32(v, 2) != 0) mask |= 4;
        if (vgetq_lane_u32(v, 3) != 0) mask |= 8;
        return mask;
    }

    /// @brief Find first set lane (optimized for small lane count)
    static int find_first_set(vector_type v) noexcept {
        if (vgetq_lane_u32(v, 0) != 0) return 0;
        if (vgetq_lane_u32(v, 1) != 0) return 1;
        if (vgetq_lane_u32(v, 2) != 0) return 2;
        if (vgetq_lane_u32(v, 3) != 0) return 3;
        return 4;
    }
};

#endif // SIMDEX_HAS_NEON

// ============================================================================
// Helper utilities
// ============================================================================

/// @brief Count trailing zeros (position of first set bit)
inline int ctz(int mask) noexcept {
    if (mask == 0) return 32;  // Undefined for 0, return safe value
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctz(static_cast<unsigned>(mask));
#elif defined(_MSC_VER)
    unsigned long index;
    _BitScanForward(&index, static_cast<unsigned long>(mask));
    return static_cast<int>(index);
#else
    // Portable fallback
    int count = 0;
    while ((mask & 1) == 0) {
        mask >>= 1;
        ++count;
    }
    return count;
#endif
}

/// @brief Population count (number of set bits)
inline int popcount(int mask) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(static_cast<unsigned>(mask));
#elif defined(_MSC_VER)
    return static_cast<int>(__popcnt(static_cast<unsigned>(mask)));
#else
    // Portable fallback
    int count = 0;
    while (mask) {
        count += mask & 1;
        mask >>= 1;
    }
    return count;
#endif
}

/// @brief Get traits for auto-detected platform
template<typename T>
using default_vec_traits = vec_traits<T, resolve_platform_t<auto_platform_tag>>;

} // namespace simdex::simd
