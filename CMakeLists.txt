cmake_minimum_required(VERSION 3.18)

set(VCPKG_APPLOCAL_DEPS OFF CACHE BOOL "" FORCE)

include(FetchContent)

option(IM_OPERATOR_ENABLE_CUDA "Enable CUDA GPU acceleration" ON)

# --- CUDA toolkit detection (best practice: pin version before project()) ---
if(IM_OPERATOR_ENABLE_CUDA)
    # Pin to CUDA 12.6 to avoid NVCC 13.0 MSBuild compatibility issues.
    # Override with -DCUDAToolkit_ROOT=... on the cmake command line if needed.
    if(NOT DEFINED CUDAToolkit_ROOT)
        set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6"
            CACHE PATH "CUDA Toolkit root directory (pin to a specific version)")
    endif()
    set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc.exe" CACHE FILEPATH "NVCC compiler path")
    set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA target architectures (semicolon-separated)")
    set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared" CACHE STRING "CUDA runtime linking: Shared=MD, Static=MT")
    mark_as_advanced(CUDAToolkit_ROOT CMAKE_CUDA_COMPILER)

    cmake_policy(SET CMP0104 NEW)
    project(im_operator VERSION 2.0.0 LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
    project(im_operator VERSION 2.0.0 LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(IM_OPERATOR_BUILD_TESTS "Build test programs" ON)
option(IM_OPERATOR_BUILD_UNIT_TESTS "Build unit tests (GoogleTest)" ON)
option(IM_OPERATOR_BUILD_BENCHMARKS "Build benchmark programs" ON)
option(IM_OPERATOR_ENABLE_SIMD "Enable SIMD optimizations" ON)
option(IM_OPERATOR_ENABLE_OPENMP "Enable OpenMP multithreading" ON)

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4> $<$<COMPILE_LANGUAGE:CXX>:/WX-> $<$<COMPILE_LANGUAGE:CXX>:/MP>)
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2> $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/Ob2>)
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra> $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
    add_compile_options($<$<CONFIG:Release>:-O3> $<$<CONFIG:Release>:-march=native> $<$<CONFIG:Release>:-flto>)
    add_link_options($<$<CONFIG:Release>:-flto>)
endif()

add_library(im_operator STATIC
    src/analyzer.cpp
    src/algorithms/dispatch.cpp
    src/algorithms/avx2.cpp
    src/algorithms/hqli.cpp
    src/algorithms/mg.cpp
    src/algorithms/l7.cpp
    src/algorithms/dfpd.cpp
    src/algorithms/ahd.cpp
    src/algorithms/amaze.cpp
    src/algorithms/rcd.cpp
    src/algorithms/prism.cpp
    src/algorithms/super_fast.cpp
    src/algorithms/cuda_dispatch.cpp
    src/denoise/dispatch.cpp
    src/denoise/gaussian.cpp
    src/denoise/median.cpp
    src/denoise/bilateral.cpp
    src/denoise/nlm.cpp
    src/denoise/wavelet.cpp
    src/denoise/bayer_denoise.cpp
    src/white_balance/dispatch.cpp
    src/white_balance/gray_world.cpp
    src/white_balance/white_patch.cpp
    src/white_balance/shade_of_gray.cpp
    src/white_balance/manual.cpp
    src/ccm/dispatch.cpp
    src/ccm/linear_3x3.cpp
    src/ccm/linear_4x3.cpp
    src/ccm/polynomial_3x9.cpp
    src/ccm/manual.cpp
    src/lut/dispatch.cpp
    src/lut/core.cpp
    src/lut/cube_file.cpp
    src/lut/custom_3d.cpp
    src/lut/style_presets.cpp
    src/lut/preset_styles.cpp
    src/sharpen/dispatch.cpp
    src/sharpen/unsharp_mask.cpp
    src/sharpen/laplacian.cpp
    src/sharpen/high_pass.cpp
    src/sharpen/adaptive.cpp
    src/tone/dispatch.cpp
    src/tone/gamma.cpp
    src/tone/s_curve.cpp
    src/tone/levels.cpp
    src/tone/curves_3point.cpp
    src/tone/shadows_highlights.cpp
    src/color_temp/dispatch.cpp
    src/color_temp/temperature.cpp
    src/color_temp/kelvin.cpp
    src/color_temp/preset.cpp
    src/color_temp/manual_temp.cpp
    src/color_temp/white_balance_auto.cpp
    src/saturation/dispatch.cpp
    src/saturation/hsl.cpp
    src/saturation/vibrance.cpp
    src/saturation/channel_mixer.cpp
    src/saturation/selective.cpp
    src/jpeg_codec/dispatch.cpp
    src/jpeg_codec/encode_baseline.cpp
    src/jpeg_codec/decode_baseline.cpp
    src/jpeg_codec/cuda_dispatch.cpp
    src/calibration/dispatch.cpp
    src/calibration/three_point.cpp
    src/calibration/right_triangle.cpp
    src/hdr/dispatch.cpp
    src/hdr/reinhard.cpp
    src/hdr/reinhard_ext.cpp
    src/hdr/filmic_aces.cpp
    src/hdr/hable.cpp
    src/hdr/drago.cpp
    src/hdr/adaptive_local.cpp
    src/hdr/exponential.cpp
    src/hdr/logarithmic.cpp
    src/hdr/transfer.cpp
    src/hdr/cuda_dispatch.cpp
    src/black_level/dispatch.cpp
    src/black_level/per_channel.cpp
    src/black_level/global.cpp
    src/defect_correct/dispatch.cpp
    src/defect_correct/adaptive.cpp
    src/defect_correct/map_based.cpp
    src/highlight_reconstruct/dispatch.cpp
    src/highlight_reconstruct/channel_guided.cpp
    src/highlight_reconstruct/gradient_based.cpp
    src/local_contrast/dispatch.cpp
    src/local_contrast/unsharp.cpp
    src/local_contrast/bilateral.cpp
    src/color_calibration/common.cpp
    src/color_calibration/detect_chart.cpp
    src/color_calibration/extract_patches.cpp
    src/color_calibration/solve_ccm.cpp
    src/color_calibration/generate_linearization.cpp
)

if(IM_OPERATOR_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    target_sources(im_operator PRIVATE
        src/algorithms/cuda_kernels.cu
        src/jpeg_codec/jpeg_codec_kernels.cu
        src/hdr/hdr_kernels.cu
    )
    target_compile_definitions(im_operator PRIVATE IM_OPERATOR_HAS_CUDA)

    # CUDA include path needed by both .cpp and .cu files
    get_filename_component(_cuda_inc_dir "${CMAKE_CUDA_COMPILER}/../../include" ABSOLUTE)
    target_include_directories(im_operator PRIVATE "${_cuda_inc_dir}")

    # Link cudart: resolve from NVCC path
    get_filename_component(_cuda_root "${CMAKE_CUDA_COMPILER}/../.." ABSOLUTE)
    find_library(CUDART_LIBRARY cudart
        PATHS "${_cuda_root}/lib/x64"
        NO_DEFAULT_PATH
    )
    if(CUDART_LIBRARY)
        target_link_libraries(im_operator PRIVATE "${CUDART_LIBRARY}")
    else()
        message(WARNING "cudart.lib not found at ${_cuda_root}/lib/x64")
    endif()

    set_source_files_properties(
        src/algorithms/cuda_kernels.cu
        src/jpeg_codec/jpeg_codec_kernels.cu
        src/hdr/hdr_kernels.cu
        PROPERTIES COMPILE_OPTIONS "--expt-relaxed-constexpr"
    )
endif()

target_include_directories(im_operator
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/denoise>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/white_balance>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/ccm>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/lut>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/sharpen>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/tone>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/color_temp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/saturation>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/jpeg_codec>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/calibration>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/hdr>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/black_level>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/defect_correct>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/highlight_reconstruct>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/local_contrast>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/color_calibration>
)

target_compile_features(im_operator PUBLIC cxx_std_17)

if(IM_OPERATOR_ENABLE_SIMD)
    if(MSVC)
        target_compile_options(im_operator PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
        target_compile_definitions(im_operator PRIVATE __AVX2__)
    else()
        target_compile_options(im_operator PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2> $<$<COMPILE_LANGUAGE:CXX>:-mfma>)
    endif()
endif()

if(IM_OPERATOR_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        if(MSVC)
            target_compile_options(im_operator PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/openmp>)
        endif()
        target_link_libraries(im_operator PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

if(IM_OPERATOR_BUILD_TESTS)
    add_executable(imop_tool_folder_debayer tools/folder_data_debayer.cpp)
    target_link_libraries(imop_tool_folder_debayer PRIVATE im_operator)

    add_executable(synthetic_test tools/synthetic_test.cpp)
    target_link_libraries(synthetic_test PRIVATE im_operator)

    add_executable(cuda_verify tools/cuda_verify.cpp)
    target_link_libraries(cuda_verify PRIVATE im_operator)

    add_executable(denoise_synthetic_test tools/denoise_synthetic_test.cpp)
    target_link_libraries(denoise_synthetic_test PRIVATE im_operator)

    add_executable(jpeg_diag tools/jpeg_diag.cpp)
    target_link_libraries(jpeg_diag PRIVATE im_operator)

    add_executable(dct_test tools/dct_test.cpp)
    target_link_libraries(dct_test PRIVATE im_operator)
    target_include_directories(dct_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(jpeg_bench tools/jpeg_bench.cpp)
    target_link_libraries(jpeg_bench PRIVATE im_operator)

    add_executable(real_jpeg_bench tools/real_jpeg_bench.cpp)
    target_link_libraries(real_jpeg_bench PRIVATE im_operator)

    add_executable(jpeg_compare tools/jpeg_compare.cpp)
    target_link_libraries(jpeg_compare PRIVATE im_operator)
endif()

if(IM_OPERATOR_BUILD_UNIT_TESTS)
    # Make GoogleTest available without requiring a system install
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.15.2
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(im_operator_unit_tests
        tests/unit/main.cpp
        tests/unit/test_types.cpp
        tests/unit/test_pixel_utils.cpp
        tests/unit/test_dispatch.cpp
        tests/unit/test_algorithms.cpp
        tests/unit/test_correctness.cpp
        tests/unit/test_denoise.cpp
        tests/unit/test_white_balance.cpp
        tests/unit/test_ccm.cpp
        tests/unit/test_lut.cpp
        tests/unit/test_sharpen.cpp
        tests/unit/test_tone.cpp
        tests/unit/test_color_temp.cpp
        tests/unit/test_saturation.cpp
        tests/unit/test_jpeg_codec.cpp
        tests/unit/test_calibration.cpp
        tests/unit/test_hdr.cpp
        tests/unit/test_black_level.cpp
        tests/unit/test_defect_correct.cpp
        tests/unit/test_highlight_reconstruct.cpp
        tests/unit/test_local_contrast.cpp
        tests/unit/test_color_calibration.cpp
    )
    target_link_libraries(im_operator_unit_tests PRIVATE im_operator gtest_main)

    # Enable CTest integration
    include(CTest)
    enable_testing()
    add_test(NAME ImOperatorUnitTests COMMAND im_operator_unit_tests)
endif()

if(IM_OPERATOR_BUILD_BENCHMARKS)
    add_executable(im_operator_benchmark benchmarks/benchmark.cpp)
    target_link_libraries(im_operator_benchmark PRIVATE im_operator)
endif()

install(TARGETS im_operator
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES include/im_operator.h DESTINATION include)
install(FILES include/imop/types.hpp DESTINATION include/imop)
install(FILES include/imop/pixel_utils.hpp DESTINATION include/imop)
install(FILES include/imop/analyzer.hpp DESTINATION include/imop)
install(FILES include/imop/algorithms.hpp DESTINATION include/imop)
install(FILES include/denoise.h DESTINATION include)
install(FILES include/denoise/types.hpp DESTINATION include/denoise)
install(FILES include/denoise/algorithms.hpp DESTINATION include/denoise)
install(FILES include/white_balance.h DESTINATION include)
install(FILES include/white_balance/types.hpp      DESTINATION include/white_balance)
install(FILES include/white_balance/algorithms.hpp  DESTINATION include/white_balance)
install(FILES include/ccm.h DESTINATION include)
install(FILES include/ccm/types.hpp      DESTINATION include/ccm)
install(FILES include/ccm/algorithms.hpp  DESTINATION include/ccm)
install(FILES include/lut.h DESTINATION include)
install(FILES include/lut/types.hpp      DESTINATION include/lut)
install(FILES include/lut/algorithms.hpp  DESTINATION include/lut)
install(FILES include/sharpen.h DESTINATION include)
install(FILES include/sharpen/types.hpp      DESTINATION include/sharpen)
install(FILES include/sharpen/algorithms.hpp  DESTINATION include/sharpen)
install(FILES include/tone.h DESTINATION include)
install(FILES include/tone/types.hpp      DESTINATION include/tone)
install(FILES include/tone/algorithms.hpp  DESTINATION include/tone)
install(FILES include/color_temp.h DESTINATION include)
install(FILES include/color_temp/types.hpp      DESTINATION include/color_temp)
install(FILES include/color_temp/algorithms.hpp  DESTINATION include/color_temp)
install(FILES include/saturation.h DESTINATION include)
install(FILES include/saturation/types.hpp      DESTINATION include/saturation)
install(FILES include/saturation/algorithms.hpp  DESTINATION include/saturation)
install(FILES include/jpeg_codec.h DESTINATION include)
install(FILES include/jpeg_codec/types.hpp      DESTINATION include/jpeg_codec)
install(FILES include/jpeg_codec/algorithms.hpp  DESTINATION include/jpeg_codec)
install(FILES include/calibration.h DESTINATION include)
install(FILES include/calibration/types.hpp      DESTINATION include/calibration)
install(FILES include/calibration/algorithms.hpp  DESTINATION include/calibration)
install(FILES include/black_level.h DESTINATION include)
install(FILES include/black_level/types.hpp      DESTINATION include/black_level)
install(FILES include/black_level/algorithms.hpp  DESTINATION include/black_level)
install(FILES include/defect_correct.h DESTINATION include)
install(FILES include/defect_correct/types.hpp      DESTINATION include/defect_correct)
install(FILES include/defect_correct/algorithms.hpp  DESTINATION include/defect_correct)
install(FILES include/highlight_reconstruct.h DESTINATION include)
install(FILES include/highlight_reconstruct/types.hpp      DESTINATION include/highlight_reconstruct)
install(FILES include/highlight_reconstruct/algorithms.hpp  DESTINATION include/highlight_reconstruct)
install(FILES include/local_contrast.h DESTINATION include)
install(FILES include/local_contrast/types.hpp      DESTINATION include/local_contrast)
install(FILES include/local_contrast/algorithms.hpp  DESTINATION include/local_contrast)
install(FILES include/color_calibration.h DESTINATION include)
install(FILES include/color_calibration/types.hpp      DESTINATION include/color_calibration)
install(FILES include/color_calibration/algorithms.hpp  DESTINATION include/color_calibration)
install(FILES include/hdr.h DESTINATION include)
install(FILES include/hdr/types.hpp      DESTINATION include/hdr)
install(FILES include/hdr/algorithms.hpp  DESTINATION include/hdr)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ImOperatorConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ImOperatorConfigVersion.cmake"
    DESTINATION lib/cmake/ImOperator
)
