cmake_minimum_required(VERSION 3.20)
project(pymixef_core VERSION 0.1.1 LANGUAGES CXX)

option(PYMIXEF_BUILD_TESTS "Build the native conformance smoke test" ON)

add_library(pymixef_core SHARED src/core.cpp)
target_compile_features(pymixef_core PUBLIC cxx_std_20)
target_include_directories(
  pymixef_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(
  pymixef_core
  PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

if(PYMIXEF_BUILD_TESTS)
  enable_testing()
  add_executable(pymixef_core_test tests/test_core.cpp)
  target_compile_features(pymixef_core_test PRIVATE cxx_std_20)
  target_link_libraries(pymixef_core_test PRIVATE pymixef_core)
  add_test(NAME pymixef_core_conformance COMMAND pymixef_core_test)
endif()

install(TARGETS pymixef_core)
install(FILES include/pymixef_core.h DESTINATION include)
