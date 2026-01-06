#pragma once

/// @file linear_scan_neon_opt.hpp
/// @brief Optimized NEON linear scan that minimizes cross-domain transfers
///
/// Key optimizations:
/// 1. Process 8 elements (4 NEON vectors) per iteration
/// 2. Use vmaxvq to check for ANY match (stays in SIMD domain)
/// 3. Only extract lanes when we KNOW there's a match
/// 4. Binary narrowing to find exact position with minimal extractions

#include <cstddef>
#include <cstdint>

#include "simdex/simd/platform.hpp"

#if defined(SIMDEX_HAS_NEON)
// Include arm_neon.h BEFORE any namespace declarations to avoid
// collision with Apple's simd:: namespace on newer Xcode versions
#include <arm_neon.h>

namespace simdex {

// Type alias to avoid collision with Apple's simd:: namespace
// arm_neon.h must be included before this to ensure types are in global namespace
using neon_u64x2 = uint64x2_t;

/// @brief Optimized NEON linear scan for 64-bit keys
///
/// This implementation addresses the core NEON performance issues:
/// - vgetq_lane is expensive (cross-domain transfer)
/// - We need to minimize lane extractions in the hot path
/// - vmaxvq_u64 can check "any match" without lane extraction
class NEONLinearScanOptimized {
public:
    static constexpr const char* name() noexcept {
        return "simd_linear_scan_neon_opt";
    }

    static constexpr std::size_t optimal_range_size() noexcept {
        return 32;  // Still best for small ranges
    }

    /// @brief Optimized lower_bound using multi-vector processing
    ///
    /// Algorithm:
    /// 1. Process 8 elements (4 vectors) per iteration
    /// 2. OR all comparison results together
    /// 3. Use vmaxvq_u64 to check if ANY matched (no lane extraction!)
    /// 4. If no match, skip all 8 elements (fast path)
    /// 5. If match found, use binary narrowing to find exact position
    [[nodiscard]]
    const std::uint64_t* lower_bound(const std::uint64_t* data, std::size_t lo,
                                      std::size_t hi, std::uint64_t key) const noexcept {
        const std::uint64_t* ptr = data + lo;
        const std::uint64_t* end = data + hi;

        // Broadcast key to all lanes
        neon_u64x2 vkey = vdupq_n_u64(key);

        // === Main loop: 8 elements per iteration ===
        // This amortizes the "any match" check over 8 elements
        while (ptr + 8 <= end) {
            // Load 4 vectors (8 elements total)
            neon_u64x2 v0 = vld1q_u64(ptr);
            neon_u64x2 v1 = vld1q_u64(ptr + 2);
            neon_u64x2 v2 = vld1q_u64(ptr + 4);
            neon_u64x2 v3 = vld1q_u64(ptr + 6);

            // Compare: find where element >= key (i.e., NOT(key > element))
            // vcgtq_u64 returns all 1s where a > b
            neon_u64x2 gt0 = vcgtq_u64(vkey, v0);  // key > v0[i] ?
            neon_u64x2 gt1 = vcgtq_u64(vkey, v1);
            neon_u64x2 gt2 = vcgtq_u64(vkey, v2);
            neon_u64x2 gt3 = vcgtq_u64(vkey, v3);

            // We want first position where key <= element
            // That's where gt[i] is 0 (NOT greater)
            // Invert: ge[i] = 1 where key <= element
            neon_u64x2 ge0 = vmvnq_u64(gt0);  // NOT(key > v) = (key <= v)
            neon_u64x2 ge1 = vmvnq_u64(gt1);
            neon_u64x2 ge2 = vmvnq_u64(gt2);
            neon_u64x2 ge3 = vmvnq_u64(gt3);

            // OR together: any_match[i] = 1 if ANY of the 8 positions matched
            neon_u64x2 any01 = vorrq_u64(ge0, ge1);
            neon_u64x2 any23 = vorrq_u64(ge2, ge3);
            neon_u64x2 any = vorrq_u64(any01, any23);

            // Check if ANY lane in 'any' is non-zero
            // vmaxvq_u64 returns max of all lanes as scalar
            // If any lane is 0xFFFF..., result is non-zero
            if (vmaxvq_u64(any) != 0) {
                // Found a match somewhere in these 8 elements
                // Binary narrowing to find exact position

                // Is it in first 4 or last 4?
                if (vmaxvq_u64(any01) != 0) {
                    // First 4 elements (positions 0-3)
                    if (vmaxvq_u64(ge0) != 0) {
                        // Positions 0-1: extract only needed lanes
                        if (vgetq_lane_u64(ge0, 0)) return ptr;
                        return ptr + 1;
                    } else {
                        // Positions 2-3
                        if (vgetq_lane_u64(ge1, 0)) return ptr + 2;
                        return ptr + 3;
                    }
                } else {
                    // Last 4 elements (positions 4-7)
                    if (vmaxvq_u64(ge2) != 0) {
                        // Positions 4-5
                        if (vgetq_lane_u64(ge2, 0)) return ptr + 4;
                        return ptr + 5;
                    } else {
                        // Positions 6-7
                        if (vgetq_lane_u64(ge3, 0)) return ptr + 6;
                        return ptr + 7;
                    }
                }
            }

            ptr += 8;
        }

        // === Secondary loop: 4 elements per iteration ===
        while (ptr + 4 <= end) {
            neon_u64x2 v0 = vld1q_u64(ptr);
            neon_u64x2 v1 = vld1q_u64(ptr + 2);

            neon_u64x2 ge0 = vmvnq_u64(vcgtq_u64(vkey, v0));
            neon_u64x2 ge1 = vmvnq_u64(vcgtq_u64(vkey, v1));

            neon_u64x2 any = vorrq_u64(ge0, ge1);

            if (vmaxvq_u64(any) != 0) {
                if (vmaxvq_u64(ge0) != 0) {
                    if (vgetq_lane_u64(ge0, 0)) return ptr;
                    return ptr + 1;
                } else {
                    if (vgetq_lane_u64(ge1, 0)) return ptr + 2;
                    return ptr + 3;
                }
            }

            ptr += 4;
        }

        // === Tertiary loop: 2 elements per iteration ===
        while (ptr + 2 <= end) {
            neon_u64x2 v = vld1q_u64(ptr);
            neon_u64x2 ge = vmvnq_u64(vcgtq_u64(vkey, v));

            if (vmaxvq_u64(ge) != 0) {
                if (vgetq_lane_u64(ge, 0)) return ptr;
                return ptr + 1;
            }

            ptr += 2;
        }

        // === Scalar tail ===
        while (ptr < end) {
            if (*ptr >= key) return ptr;
            ++ptr;
        }

        return end;
    }
};

/// @brief Alternative: Prefetch-assisted scalar search
///
/// Hypothesis: M1's branch predictor is so good that scalar wins.
/// But we can still use NEON for prefetching ahead while doing scalar work.
class NEONPrefetchScalar {
public:
    static constexpr const char* name() noexcept {
        return "neon_prefetch_scalar";
    }

    [[nodiscard]]
    const std::uint64_t* lower_bound(const std::uint64_t* data, std::size_t lo,
                                      std::size_t hi, std::uint64_t key) const noexcept {
        const std::uint64_t* ptr = data + lo;
        const std::uint64_t* end = data + hi;

        // Prefetch ahead while doing scalar comparisons
        while (ptr + 16 < end) {
            // Prefetch 16 elements ahead (2 cache lines on M1)
            __builtin_prefetch(ptr + 16, 0, 0);
            __builtin_prefetch(ptr + 24, 0, 0);

            // Unrolled scalar loop (branch predictor friendly)
            if (ptr[0] >= key) return ptr;
            if (ptr[1] >= key) return ptr + 1;
            if (ptr[2] >= key) return ptr + 2;
            if (ptr[3] >= key) return ptr + 3;
            if (ptr[4] >= key) return ptr + 4;
            if (ptr[5] >= key) return ptr + 5;
            if (ptr[6] >= key) return ptr + 6;
            if (ptr[7] >= key) return ptr + 7;

            ptr += 8;
        }

        // Finish remaining elements
        while (ptr < end) {
            if (*ptr >= key) return ptr;
            ++ptr;
        }

        return end;
    }
};

} // namespace simdex

#endif // SIMDEX_HAS_NEON
