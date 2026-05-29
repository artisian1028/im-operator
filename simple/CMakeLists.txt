cmake_minimum_required(VERSION 3.18)
project(imop_examples VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to the im_operator library (built in ../build/)
set(IMOP_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Find the library
find_library(IM_OPERATOR_LIB im_operator
    PATHS ${IMOP_ROOT}/build/Release ${IMOP_ROOT}/build/Debug
    NO_DEFAULT_PATH
)

if(NOT IM_OPERATOR_LIB)
    message(FATAL_ERROR "im_operator.lib not found. Build the main project first: cd .. && cmake -B build && cmake --build build")
endif()

# Include directories
include_directories(${IMOP_ROOT}/include)
include_directories(${IMOP_ROOT}/src/calibration)  # for common.hpp (detail functions)

# Find CUDA runtime (needed by im_operator.lib which has CUDA kernels)
set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6"
    CACHE PATH "CUDA Toolkit root directory")
find_library(CUDART_LIBRARY cudart
    PATHS "${CUDAToolkit_ROOT}/lib/x64"
    NO_DEFAULT_PATH
)
if(NOT CUDART_LIBRARY)
    message(WARNING "cudart.lib not found - examples may have linker errors")
endif()

# --- Test data generator ---
add_executable(generate_test_data test_data/generate_test_data.cpp)

# --- Operator examples ---
set(EXAMPLES
    demosaic_example
    denoise_example
    white_balance_example
    ccm_example
    lut_example
    sharpen_example
    tone_example
    color_temp_example
    saturation_example
    hdr_example
    black_level_example
    defect_correct_example
    highlight_reconstruct_example
    local_contrast_example
    lens_shading_example
    jpeg_codec_example
    pixel_utils_example
    analyzer_example
    calibration_example
)

foreach(ex ${EXAMPLES})
    add_executable(${ex} ${ex}.cpp)
    target_link_libraries(${ex} PRIVATE ${IM_OPERATOR_LIB})
    if(CUDART_LIBRARY)
        target_link_libraries(${ex} PRIVATE ${CUDART_LIBRARY})
    endif()
endforeach()

# Build all examples at once
add_custom_target(all_examples DEPENDS ${EXAMPLES})

message(STATUS "Examples configured. Run 'cmake --build .' to build all.")
message(STATUS "Run 'generate_test_data' first to create test data files.")
