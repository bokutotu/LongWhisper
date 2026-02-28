cmake_minimum_required(VERSION 3.20)
project(LongWhisper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LONGWHISPER_THIRD_PARTY_DIR "third_party")
set(LONGWHISPER_RAPIDJSON_INCLUDE
  "${CMAKE_CURRENT_SOURCE_DIR}/${LONGWHISPER_THIRD_PARTY_DIR}/rapidjson/include"
)

add_library(longwhisper_assets STATIC
  src/dtype.cpp
  src/assets_config.cpp
)
target_include_directories(longwhisper_assets
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${LONGWHISPER_RAPIDJSON_INCLUDE}
)

option(LONGWHISPER_BUILD_TESTS "Build LongWhisper tests" ON)
if(LONGWHISPER_BUILD_TESTS)
  include(CTest)
endif()

if(BUILD_TESTING)
  add_subdirectory(${LONGWHISPER_THIRD_PARTY_DIR}/googletest EXCLUDE_FROM_ALL)

  add_executable(assets_config_test
    tests/assets_config_test.cpp
  )
  target_link_libraries(assets_config_test
    PRIVATE
      longwhisper_assets
      GTest::gtest_main
  )
  target_compile_definitions(assets_config_test
    PRIVATE
      LONGWHISPER_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  )

  include(GoogleTest)
  gtest_discover_tests(assets_config_test)
endif()
