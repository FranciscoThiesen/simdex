# Tests CMakeLists.txt

find_package(GTest CONFIG REQUIRED)

# Test for scalar baseline
add_executable(test_baseline test_baseline.cpp)
target_link_libraries(test_baseline PRIVATE simdex GTest::gtest GTest::gtest_main)
add_test(NAME test_baseline COMMAND test_baseline)

# Comprehensive tests for all search strategies
add_executable(test_search_strategies test_search_strategies.cpp)
target_link_libraries(test_search_strategies PRIVATE simdex GTest::gtest GTest::gtest_main)
add_test(NAME test_search_strategies COMMAND test_search_strategies)

# PGM-Index integration tests
add_executable(test_pgm_integration test_pgm_integration.cpp)
target_link_libraries(test_pgm_integration PRIVATE simdex GTest::gtest GTest::gtest_main)
add_test(NAME test_pgm_integration COMMAND test_pgm_integration)

# Batched query processing tests
add_executable(test_batch_query test_batch_query.cpp)
target_link_libraries(test_batch_query PRIVATE simdex GTest::gtest GTest::gtest_main)
add_test(NAME test_batch_query COMMAND test_batch_query)
