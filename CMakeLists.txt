cmake_minimum_required(VERSION 3.10)

project(training_data_loader LANGUAGES CXX)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ATOMIC_STOCKFISH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/Atomic-Stockfish")
set(ATOMIC_STOCKFISH_PIN "420c9f35266fbdc2167dc5b9d8d20d90281c60c9")
set(ATOMIC_BIN_V2_DATA_SCHEMA_SHA256
  "0352b036f2a140c609e3eb9c9d635dc553e8d77253d8faa92437390f5cf93cb6")
set(ATOMIC_BIN_V2_MANIFEST_SCHEMA_SHA256
  "83d63922df3ac4a0c81a21ec9d9fd9e180efe50f26efee62fe01710e09da5b42")
set(ATOMIC_DATA_TOOLS_DECODE_SCHEMA_SHA256
  "5e3f8d7c6db6ee955b71747ee063859e15609adb557a3754228a606f3df2caad")
if(NOT EXISTS "${ATOMIC_STOCKFISH_ROOT}/src/data/atomic_bin_v2_reader.cpp")
  message(FATAL_ERROR
    "Atomic-Stockfish submodule is missing; run git submodule update --init --recursive")
endif()

find_package(Git REQUIRED)
execute_process(
  COMMAND ${GIT_EXECUTABLE} config --file ${CMAKE_CURRENT_SOURCE_DIR}/.gitmodules
    --get submodule.external/Atomic-Stockfish.url
  OUTPUT_VARIABLE ATOMIC_STOCKFISH_URL
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE ATOMIC_STOCKFISH_URL_RESULT)
if(NOT ATOMIC_STOCKFISH_URL_RESULT EQUAL 0
   OR NOT ATOMIC_STOCKFISH_URL STREQUAL "https://github.com/Belzedar94/Atomic-Stockfish.git")
  message(FATAL_ERROR "Atomic-Stockfish submodule URL differs from the audited upstream")
endif()
execute_process(
  COMMAND ${GIT_EXECUTABLE} -C ${ATOMIC_STOCKFISH_ROOT} remote get-url origin
  OUTPUT_VARIABLE ATOMIC_STOCKFISH_ORIGIN_URL
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE ATOMIC_STOCKFISH_ORIGIN_URL_RESULT)
if(NOT ATOMIC_STOCKFISH_ORIGIN_URL_RESULT EQUAL 0
   OR NOT ATOMIC_STOCKFISH_ORIGIN_URL STREQUAL "https://github.com/Belzedar94/Atomic-Stockfish.git")
  message(FATAL_ERROR "Atomic-Stockfish submodule origin URL differs from the audited upstream")
endif()
execute_process(
  COMMAND ${GIT_EXECUTABLE} -C ${CMAKE_CURRENT_SOURCE_DIR}
    rev-parse :external/Atomic-Stockfish
  OUTPUT_VARIABLE ATOMIC_STOCKFISH_GITLINK
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE ATOMIC_STOCKFISH_GITLINK_RESULT)
if(NOT ATOMIC_STOCKFISH_GITLINK_RESULT EQUAL 0
   OR NOT ATOMIC_STOCKFISH_GITLINK STREQUAL ATOMIC_STOCKFISH_PIN)
  message(FATAL_ERROR "Atomic-Stockfish gitlink differs from the audited pin")
endif()
execute_process(
  COMMAND ${GIT_EXECUTABLE} -C ${ATOMIC_STOCKFISH_ROOT} rev-parse HEAD
  OUTPUT_VARIABLE ATOMIC_STOCKFISH_HEAD
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE ATOMIC_STOCKFISH_HEAD_RESULT)
if(NOT ATOMIC_STOCKFISH_HEAD_RESULT EQUAL 0
   OR NOT ATOMIC_STOCKFISH_HEAD STREQUAL ATOMIC_STOCKFISH_PIN)
  message(FATAL_ERROR
    "Atomic-Stockfish must be pinned exactly to ${ATOMIC_STOCKFISH_PIN}; "
    "run git submodule update --init --recursive")
endif()
execute_process(
  COMMAND ${GIT_EXECUTABLE} -C ${ATOMIC_STOCKFISH_ROOT} status --porcelain
  OUTPUT_VARIABLE ATOMIC_STOCKFISH_STATUS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE ATOMIC_STOCKFISH_STATUS_RESULT)
if(NOT ATOMIC_STOCKFISH_STATUS_RESULT EQUAL 0 OR NOT ATOMIC_STOCKFISH_STATUS STREQUAL "")
  message(FATAL_ERROR "Atomic-Stockfish submodule must be clean")
endif()
execute_process(
  COMMAND ${GIT_EXECUTABLE} -C ${ATOMIC_STOCKFISH_ROOT}
    merge-base --is-ancestor ${ATOMIC_STOCKFISH_PIN} origin/main
  RESULT_VARIABLE ATOMIC_STOCKFISH_MAIN_RESULT)
if(NOT ATOMIC_STOCKFISH_MAIN_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Pinned Atomic-Stockfish commit must be an ancestor of origin/main; fetch the submodule remote")
endif()

file(SHA256
  "${ATOMIC_STOCKFISH_ROOT}/schemas/atomic-bin-v2.json"
  ATOMIC_BIN_V2_DATA_SCHEMA_ACTUAL)
file(SHA256
  "${ATOMIC_STOCKFISH_ROOT}/schemas/atomic-bin-v2-manifest.json"
  ATOMIC_BIN_V2_MANIFEST_SCHEMA_ACTUAL)
file(SHA256
  "${ATOMIC_STOCKFISH_ROOT}/schemas/atomic-data-tools-decode-v1.json"
  ATOMIC_DATA_TOOLS_DECODE_SCHEMA_ACTUAL)
if(NOT ATOMIC_BIN_V2_DATA_SCHEMA_ACTUAL STREQUAL ATOMIC_BIN_V2_DATA_SCHEMA_SHA256)
  message(FATAL_ERROR "Pinned Atomic BIN V2 data schema hash differs")
endif()
if(NOT ATOMIC_BIN_V2_MANIFEST_SCHEMA_ACTUAL STREQUAL ATOMIC_BIN_V2_MANIFEST_SCHEMA_SHA256)
  message(FATAL_ERROR "Pinned Atomic BIN V2 manifest schema hash differs")
endif()
if(NOT ATOMIC_DATA_TOOLS_DECODE_SCHEMA_ACTUAL STREQUAL ATOMIC_DATA_TOOLS_DECODE_SCHEMA_SHA256)
  message(FATAL_ERROR "Pinned Atomic data-tools decode schema hash differs")
endif()

add_library(atomic_bin_v2_reader_core STATIC
  ${ATOMIC_STOCKFISH_ROOT}/src/atomic_init.cpp
  ${ATOMIC_STOCKFISH_ROOT}/src/attacks.cpp
  ${ATOMIC_STOCKFISH_ROOT}/src/bitboard.cpp
  ${ATOMIC_STOCKFISH_ROOT}/src/misc.cpp
  ${ATOMIC_STOCKFISH_ROOT}/src/movegen.cpp
  ${ATOMIC_STOCKFISH_ROOT}/src/position.cpp
  ${ATOMIC_STOCKFISH_ROOT}/src/syzygy/tbprobe.cpp
  ${ATOMIC_STOCKFISH_ROOT}/src/uci_move.cpp
  ${ATOMIC_STOCKFISH_ROOT}/src/ucioption.cpp
  ${ATOMIC_STOCKFISH_ROOT}/src/data/atomic_bin_v2_wire.cpp
  ${ATOMIC_STOCKFISH_ROOT}/src/data/atomic_bin_v2.cpp
  ${ATOMIC_STOCKFISH_ROOT}/src/data/sha256.cpp
  ${ATOMIC_STOCKFISH_ROOT}/src/data/atomic_bin_v2_manifest.cpp
  ${ATOMIC_STOCKFISH_ROOT}/src/data/atomic_bin_v2_reader.cpp
  lib/atomic_stockfish_tt_stub.cpp)
set_target_properties(atomic_bin_v2_reader_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(atomic_bin_v2_reader_core PUBLIC
  ${ATOMIC_STOCKFISH_ROOT}/src)
target_compile_definitions(atomic_bin_v2_reader_core PRIVATE NNUE_EMBEDDING_OFF)

add_library(dataset_file_identity STATIC lib/dataset_file_identity.cpp)
set_target_properties(dataset_file_identity PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(training_data_loader SHARED training_data_loader.cpp)

if(MSVC)
  target_compile_options(training_data_loader PRIVATE
    $<$<CONFIG:Release>:/O2>
    $<$<CONFIG:RelWithDebInfo>:/O2>)
else()
  target_compile_options(training_data_loader PRIVATE
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:RelWithDebInfo>:-g>
    $<$<CONFIG:RelWithDebInfo>:-O3>)
endif()

find_package(Threads REQUIRED)

target_link_libraries(training_data_loader PRIVATE
  atomic_bin_v2_reader_core
  dataset_file_identity
  Threads::Threads)
if(WIN32)
  target_link_libraries(training_data_loader PRIVATE shell32)
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(training_data_loader_tests tests/native_loader_tests.cpp)
  target_link_libraries(training_data_loader_tests PRIVATE
    atomic_bin_v2_reader_core
    dataset_file_identity
    Threads::Threads)
  if(WIN32)
    target_link_libraries(training_data_loader_tests PRIVATE shell32)
  endif()
  if(MSVC)
    target_compile_options(training_data_loader_tests PRIVATE $<$<CONFIG:Release>:/O2>)
  else()
    target_compile_options(training_data_loader_tests PRIVATE $<$<CONFIG:Release>:-O3>)
  endif()
  add_test(NAME training_data_loader_tests COMMAND training_data_loader_tests)
endif()

install(
  TARGETS training_data_loader
  RUNTIME DESTINATION .
  LIBRARY DESTINATION .)
install(FILES Copying.txt DESTINATION .)
