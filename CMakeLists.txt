cmake_minimum_required(VERSION 3.20)
project(simdex VERSION 0.1.0 LANGUAGES CXX)

# C++20 required for concepts
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(SIMDEX_BUILD_TESTS "Build tests" ON)
option(SIMDEX_BUILD_BENCHMARKS "Build benchmarks" ON)
option(SIMDEX_ENABLE_AVX2 "Enable AVX2 support" ON)
option(SIMDEX_ENABLE_NEON "Enable ARM NEON support" ON)

# Detect platform and set SIMD flags
include(CheckCXXCompilerFlag)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64)")
    set(SIMDEX_IS_X86_64 TRUE)
    if(SIMDEX_ENABLE_AVX2)
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        if(COMPILER_SUPPORTS_AVX2)
            add_compile_options(-mavx2)
            add_compile_definitions(SIMDEX_HAS_AVX2=1)
            message(STATUS "AVX2 support enabled")
        endif()
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|arm64|ARM64)")
    set(SIMDEX_IS_ARM64 TRUE)
    if(SIMDEX_ENABLE_NEON)
        # NEON is always available on ARM64
        add_compile_definitions(SIMDEX_HAS_NEON=1)
        message(STATUS "ARM NEON support enabled")
    endif()
endif()

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
elseif(MSVC)
    add_compile_options(/W4)
endif()

# Header-only library target
add_library(simdex INTERFACE)
target_include_directories(simdex INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# PGM-Index (header-only)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pgm-index/include")
    target_include_directories(simdex INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/pgm-index/include>
    )
    message(STATUS "PGM-Index found")
else()
    message(WARNING "PGM-Index not found. Run: git submodule update --init --recursive")
endif()

# Tests
if(SIMDEX_BUILD_TESTS)
    enable_testing()
    find_package(GTest CONFIG QUIET)
    if(GTest_FOUND)
        add_subdirectory(tests)
        message(STATUS "Tests enabled")
    else()
        message(STATUS "GTest not found, tests disabled. Install via: vcpkg install gtest")
    endif()
endif()

# Benchmarks
if(SIMDEX_BUILD_BENCHMARKS)
    find_package(benchmark CONFIG QUIET)
    if(benchmark_FOUND)
        add_subdirectory(benchmark)
        message(STATUS "Benchmarks enabled")
    else()
        message(STATUS "Google Benchmark not found, benchmarks disabled. Install via: vcpkg install benchmark")
    endif()
endif()

# Install
include(GNUInstallDirs)
install(TARGETS simdex EXPORT simdexTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT simdexTargets
    FILE simdexTargets.cmake
    NAMESPACE simdex::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/simdex
)
