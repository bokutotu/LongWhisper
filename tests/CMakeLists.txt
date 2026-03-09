add_executable(longwhisper_tests
  assets_config_test.cpp
  vad_test.cpp
  wav_test.cpp
  preprocess400_test.cu
  whisper_preprocess_reference.cpp
  whisper_preprocess_reference.h
)

target_link_libraries(longwhisper_tests
  PRIVATE
    longwhisper
    GTest::gtest_main
)

target_compile_definitions(longwhisper_tests
  PRIVATE
    LONGWHISPER_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

target_include_directories(longwhisper_tests
  PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

set_target_properties(longwhisper_tests PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  CUDA_SEPARABLE_COMPILATION ON
)

gtest_discover_tests(longwhisper_tests)
