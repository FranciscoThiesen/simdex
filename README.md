# SIMDEX

**SIMD-optimized last-mile search for learned indexes**

SIMDEX accelerates the "last-mile" search in learned indexes like PGM-Index by replacing naive `std::lower_bound` with SIMD-optimized alternatives.

## The Problem

Learned indexes predict a position, then search a small range (typically 16-256 elements) to find the exact key. This "last-mile" search uses `std::lower_bound`, which doesn't leverage SIMD instructions.

## The Solution

SIMDEX provides three SIMD-accelerated search strategies:

| Strategy | Best For | Description |
|----------|----------|-------------|
| **SIMD Linear Scan** | < 32 elements | Vectorized scan comparing 2-8 elements per iteration |
| **SIMD K-ary Search** | 32-256 elements | Compare K pivots simultaneously, narrowing by K+1 per step |
| **Eytzinger Search** | > 256 elements | Cache-optimized BFS layout with prefetching |

## Supported Platforms

- **x86-64**: AVX2 (256-bit vectors, 4x uint64 or 8x uint32)
- **ARM64**: NEON (128-bit vectors, 2x uint64 or 4x uint32)
- **Fallback**: Scalar implementation for unsupported platforms

## Quick Start

```cpp
#include "simdex/simdex.hpp"

// Create sorted data
std::vector<std::uint64_t> data = {1, 5, 10, 15, 20, 50, 100};

// Use SIMD linear scan (auto-selects AVX2 or NEON)
simdex::SIMDLinearScan<std::uint64_t> search;

// Search in range [0, 7) for key 15
const auto* result = search.lower_bound(data.data(), 0, data.size(), 15);
// result points to element 15
```

## Building

```bash
# Clone with submodules
git clone --recursive https://github.com/FranciscoThiesen/simdex.git
cd simdex

# Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run tests (requires GTest)
ctest --test-dir build

# Run benchmarks (requires Google Benchmark)
./build/benchmark/bench_strategies
```

### Dependencies

- **C++20** compiler (GCC 10+, Clang 12+, MSVC 2019+)
- **PGM-Index** (included as submodule)
- **Google Test** (optional, for tests)
- **Google Benchmark** (optional, for benchmarks)

Install dependencies via vcpkg:
```bash
vcpkg install gtest benchmark
cmake -B build -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
```

## API

All strategies satisfy the `SearchStrategy` concept:

```cpp
template<typename S, typename T>
concept SearchStrategy = requires(const S strategy, const T* data,
                                   std::size_t lo, std::size_t hi, T key) {
    { strategy.lower_bound(data, lo, hi, key) } -> std::convertible_to<const T*>;
    { S::name() } -> std::convertible_to<const char*>;
    { S::optimal_range_size() } -> std::convertible_to<std::size_t>;
};
```

### Available Strategies

```cpp
simdex::ScalarBaseline<T>      // std::lower_bound wrapper
simdex::SIMDLinearScan<T>      // SIMD vectorized scan
simdex::SIMDKarySearch<T>      // SIMD k-ary pivot search
simdex::EytzingerSearch<T>     // Cache-optimized layout
```

### Strategy Selection Heuristic

```cpp
const T* search(const T* data, size_t lo, size_t hi, T key) {
    size_t range = hi - lo;

    if (range <= 32) {
        return SIMDLinearScan<T>{}.lower_bound(data, lo, hi, key);
    } else if (range <= 256) {
        return SIMDKarySearch<T>{}.lower_bound(data, lo, hi, key);
    } else {
        // For Eytzinger, data must be pre-transformed
        return EytzingerSearch<T>{}.lower_bound(eytz_data, lo, hi, key);
    }
}
```

## Project Structure

```
simdex/
├── include/simdex/
│   ├── simdex.hpp              # Main header
│   ├── core/
│   │   ├── types.hpp           # Type definitions
│   │   └── concepts.hpp        # SearchStrategy concept
│   ├── simd/
│   │   ├── platform.hpp        # Platform detection
│   │   └── vec_traits.hpp      # SIMD abstraction layer
│   └── search/
│       ├── scalar_baseline.hpp # std::lower_bound wrapper
│       ├── linear_scan.hpp     # SIMD linear scan
│       ├── kary_search.hpp     # SIMD k-ary search
│       └── eytzinger.hpp       # Eytzinger layout
├── tests/                      # Unit tests
├── benchmark/                  # Performance benchmarks
├── examples/                   # Usage examples
└── third_party/pgm-index/      # PGM-Index submodule
```

## License

MIT License

## References

- [PGM-Index](https://github.com/gvinciguerra/PGM-index)
- [Algorithmica: Eytzinger Binary Search](https://algorithmica.org/en/eytzinger)
- [simdjson](https://github.com/simdjson/simdjson)
- Schlegel et al. "k-ary Search on Modern Processors"
