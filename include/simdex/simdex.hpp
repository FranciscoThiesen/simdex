#pragma once

/// @file simdex.hpp
/// @brief Main header for simdex library
///
/// SIMDEX: SIMD-optimized last-mile search for learned indexes
///
/// This library provides SIMD-accelerated search strategies to replace
/// the naive std::lower_bound used in learned indexes like PGM-Index.

#include "simdex/core/types.hpp"
#include "simdex/core/concepts.hpp"
#include "simdex/simd/platform.hpp"
#include "simdex/simd/vec_traits.hpp"
#include "simdex/search/scalar_baseline.hpp"
#include "simdex/search/linear_scan.hpp"
#include "simdex/search/kary_search.hpp"
#include "simdex/search/eytzinger.hpp"
#include "simdex/index/simd_pgm_index.hpp"
#include "simdex/index/batch_query.hpp"

// Version information
#define SIMDEX_VERSION_MAJOR 0
#define SIMDEX_VERSION_MINOR 1
#define SIMDEX_VERSION_PATCH 0

namespace simdex {

/// @brief Get version string
inline constexpr const char* version() noexcept {
    return "0.1.0";
}

/// @brief Get platform information string
inline const char* platform_info() noexcept {
#if defined(SIMDEX_HAS_AVX512)
    return "x86_64 with AVX-512";
#elif defined(SIMDEX_HAS_AVX2)
    return "x86_64 with AVX2";
#elif defined(SIMDEX_HAS_NEON)
    return "ARM64 with NEON";
#else
    return "scalar (no SIMD)";
#endif
}

} // namespace simdex
