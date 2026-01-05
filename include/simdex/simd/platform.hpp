#pragma once

/// @file platform.hpp
/// @brief Platform detection and SIMD capability macros

namespace simdex::simd {

// Platform detection tags
struct scalar_tag {};
struct avx2_tag {};
struct neon_tag {};

// Compile-time platform detection
// These are set by CMake based on detected capabilities

#if defined(SIMDEX_HAS_AVX2)
    #include <immintrin.h>
    inline constexpr bool has_avx2 = true;
#else
    inline constexpr bool has_avx2 = false;
#endif

#if defined(SIMDEX_HAS_NEON)
    #include <arm_neon.h>
    inline constexpr bool has_neon = true;
#else
    inline constexpr bool has_neon = false;
#endif

// Select the best available platform at compile time
#if defined(SIMDEX_HAS_AVX2)
    using default_platform = avx2_tag;
#elif defined(SIMDEX_HAS_NEON)
    using default_platform = neon_tag;
#else
    using default_platform = scalar_tag;
#endif

// Auto-dispatch tag (uses default_platform)
struct auto_platform_tag {};

// Helper to resolve auto_platform_tag to concrete platform
template<typename Platform>
struct resolve_platform {
    using type = Platform;
};

template<>
struct resolve_platform<auto_platform_tag> {
    using type = default_platform;
};

template<typename Platform>
using resolve_platform_t = typename resolve_platform<Platform>::type;

// SIMD vector width in bytes
inline constexpr std::size_t avx2_vector_bytes = 32;
inline constexpr std::size_t neon_vector_bytes = 16;

#if defined(SIMDEX_HAS_AVX2)
    inline constexpr std::size_t default_vector_bytes = avx2_vector_bytes;
#elif defined(SIMDEX_HAS_NEON)
    inline constexpr std::size_t default_vector_bytes = neon_vector_bytes;
#else
    inline constexpr std::size_t default_vector_bytes = 0;
#endif

} // namespace simdex::simd
