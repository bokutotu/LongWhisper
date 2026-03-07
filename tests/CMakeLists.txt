function(longwhisper_add_test name)
  add_executable(${name}
    ${ARGN}
  )
  target_link_libraries(${name}
    PRIVATE
      longwhisper
      GTest::gtest_main
  )
  target_compile_definitions(${name}
    PRIVATE
      LONGWHISPER_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  )
  gtest_discover_tests(${name})
endfunction()

longwhisper_add_test(assets_config_test
  assets_config_test.cpp
)

longwhisper_add_test(vad_test
  vad_test.cpp
)

longwhisper_add_test(wav_test
  wav_test.cpp
)
