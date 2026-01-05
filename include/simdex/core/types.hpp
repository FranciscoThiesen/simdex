#pragma once

/// @file types.hpp
/// @brief Core type definitions for simdex

#include <cstddef>
#include <cstdint>

namespace simdex {

// Primary key types for benchmarking
using key32_t = std::uint32_t;
using key64_t = std::uint64_t;

// Size type
using size_type = std::size_t;

// Search result: pointer to found element (or end if not found)
template<typename T>
using search_result = const T*;

} // namespace simdex
