# Benchmark CMakeLists.txt

find_package(benchmark CONFIG REQUIRED)

# Strategy micro-benchmarks
add_executable(bench_strategies bench_strategies.cpp)
target_link_libraries(bench_strategies PRIVATE simdex benchmark::benchmark benchmark::benchmark_main)

# End-to-end PGM-Index benchmarks
add_executable(bench_pgm_e2e bench_pgm_e2e.cpp)
target_link_libraries(bench_pgm_e2e PRIVATE simdex benchmark::benchmark benchmark::benchmark_main)

# Crossover point analysis
add_executable(bench_crossover bench_crossover.cpp)
target_link_libraries(bench_crossover PRIVATE simdex benchmark::benchmark benchmark::benchmark_main)

# Batched query processing benchmarks (novel contribution)
add_executable(bench_batch bench_batch.cpp)
target_link_libraries(bench_batch PRIVATE simdex benchmark::benchmark benchmark::benchmark_main)
