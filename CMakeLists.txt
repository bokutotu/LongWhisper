cmake_minimum_required(VERSION 3.20)
project(LongWhisper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LONGWHISPER_THIRD_PARTY_DIR "third_party")
set(LONGWHISPER_RAPIDJSON_INCLUDE
  "${CMAKE_CURRENT_SOURCE_DIR}/${LONGWHISPER_THIRD_PARTY_DIR}/rapidjson/include"
)
set(LONGWHISPER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
add_library(longwhisper STATIC)
target_include_directories(longwhisper
  PUBLIC
    ${LONGWHISPER_INCLUDE_DIR}
  PRIVATE
    ${LONGWHISPER_RAPIDJSON_INCLUDE}
)
add_subdirectory(src)

option(LONGWHISPER_BUILD_TESTS "Build LongWhisper tests" ON)
if(LONGWHISPER_BUILD_TESTS)
  include(CTest)
endif()

if(BUILD_TESTING)
  add_subdirectory(${LONGWHISPER_THIRD_PARTY_DIR}/googletest EXCLUDE_FROM_ALL)
  include(GoogleTest)
  add_subdirectory(tests)
endif()
