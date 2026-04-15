cmake_minimum_required(VERSION 3.14)
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW) 
endif()
# MultiCam: On-the-fly Multi-Camera Pose Estimation Using Spatiotemporal Overlaps of Known Objects
# =============================================================================
project(M3T)

# Options
# =============================================================================
option(USE_AZURE_KINECT "Use Azure Kinect" ON)
option(USE_HOLOLENS "Use HOLOLENS" ON)
option(USE_REALSENSE "Use RealSense D435" ON)
option(USE_GTEST "Use gtest" ON)


# Libraries
# =============================================================================
set(CMAKE_CXX_STANDARD 17)
# Build static on Windows
#if (WIN32)
#    set(GLEW_STATIC true)
#    set(OpenCV_STATIC true)
#    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
#    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -MTd")
#    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -MT")
#endif ()

# Used libraries
set(MMDEPLOY_VERSION_MAJOR 1.3.1)
set(MMDEPLOY_VERSION_MINOR 1.3.1)
set(MMDEPLOY_VERSION_PATCH 1.3.1)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
if (MSVC)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
else ()
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif ()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(realsense2_DIR path/to/realsense2)
set(OpenCV_DIR path/to/opencv-4.7.0/build)
set(Torch_DIR third_party/libtorch/share/cmake/Torch)
set(TRT_DIR "third_party/TensorRT-8.6.1.6")
set(TensorRT_INCLUDE_DIRS third_party/TensorRT-8.6.1.6/include)
set(TensorRT_LIBRARIES third_party/TensorRT-8.6.1.6/lib)
set(pplcv_DIR path/to/third_party/ppl.cv/pplcv-build/install/lib/cmake/ppl)
set(ONNXRUNTIME_DIR path/to/third_party/onnxruntime-win-x64-gpu-1.10.0)
set(MMDEPLOY_DIR third_party/mmdeploy)
set(Open3D_DIR path/to/Open3D/CMake)
set(Pangolin_DIR third_party/Pangolin/cmake)
set(PalSigslot_DIR path/to/palsigslot_x64-windows/share/PalSigslot)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
option(MMDEPLOY_SHARED_LIBS "build shared libs" OFF)
option(MMDEPLOY_BUILD_SDK "build MMDeploy SDK" ON)
option(MMDEPLOY_DYNAMIC_BACKEND "dynamic load backend" OFF)
option(MMDEPLOY_BUILD_SDK_MONOLITHIC "build single lib for SDK API" ON)
option(MMDEPLOY_BUILD_TEST "build unittests" OFF)
option(MMDEPLOY_BUILD_SDK_PYTHON_API "build SDK Python API" OFF)
option(MMDEPLOY_BUILD_SDK_CSHARP_API "build SDK C# API support" OFF)
option(MMDEPLOY_BUILD_SDK_JAVA_API "build SDK JAVA API" OFF)
option(MMDEPLOY_BUILD_EXAMPLES "build examples" ON)
option(MMDEPLOY_SPDLOG_EXTERNAL "use external spdlog" OFF)
option(MMDEPLOY_ZIP_MODEL "support SDK model in zip format" OFF)
option(MMDEPLOY_COVERAGE "build SDK for coverage" OFF)
option(MMDEPLOY_USE_MSCV_STATIC "statically linked CRT" OFF)
option(MMDEPLOY_ELENA_FUSION "use elena to fuse preprocess" OFF)

## Open3D
find_package(Open3D REQUIRED)
if(WIN32)
    find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/CMake)
else()
    find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/lib/CMake)
    list(APPEND Open3D_LIBRARIES dl)
endif()

link_directories(${Open3D_LIBRARY_DIRS})

## Pangolin
find_package(Threads REQUIRED)
find_package(Pangolin CONFIG REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})


#ffmpeg
include_directories(path/to/ffmpeg-6.1/ffmpeg_6.1_install/include)
link_directories(path/to/ffmpeg-6.1/ffmpeg_6.1_install/bin)

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA Libs: \n${CUDA_LIBRARIES}\n")
message(STATUS "CUDA Headers: \n${CUDA_INCLUDE_DIRS}\n")
list(APPEND INCLUDE_DIRS
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        ./include
        )

list(APPEND ALL_LIBS
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${TensorRT_LIBRARIES}
        )

## Libtorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

## mmdeploy
set(MMDEPLOY_TARGET_DEVICES "cuda" CACHE STRING "target devices to support")
set(MMDEPLOY_TARGET_BACKENDS "ort" CACHE STRING "target inference engines to support")
set(MMDEPLOY_CODEBASES "all" CACHE STRING "select OpenMMLab codebases")

if ((NOT MMDEPLOY_BUILD_SDK_MONOLITHIC) AND MMDEPLOY_DYNAMIC_BACKEND)
    set(MMDEPLOY_DYNAMIC_BACKEND OFF)
endif ()

if (MMDEPLOY_SHARED_LIBS)
    set(MMDEPLOY_LIB_TYPE SHARED)
else ()
    set(MMDEPLOY_LIB_TYPE STATIC)
endif ()

set(MMDEPLOY_TASKS "" CACHE INTERNAL "")

if (MMDEPLOY_COVERAGE)
    add_compile_options(-coverage -fprofile-arcs -ftest-coverage)
    add_link_options(-coverage -lgcov)
endif ()

# when CUDA devices are enabled, the environment variable ASAN_OPTIONS=protect_shadow_gap=0
# must be set at runtime
if (MMDEPLOY_ASAN_ENABLE)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>)
    add_link_options(-fsanitize=address)
endif ()

# notice that ubsan has linker issues for ubuntu < 18.04, see
# https://stackoverflow.com/questions/50024731/ld-unrecognized-option-push-state-no-as-needed
if (MMDEPLOY_UBSAN_ENABLE)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=undefined>)
    add_link_options(-fsanitize=undefined)
endif ()

if (MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/diagnostics:classic>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/wd4251>)
    if (MMDEPLOY_USE_MSCV_STATIC)
        foreach(lang C CXX)
            string(REPLACE /MD /MT CMAKE_${lang}_FLAGS_DEBUG "${CMAKE_${lang}_FLAGS_DEBUG}")
            string(REPLACE /MD /MT CMAKE_${lang}_FLAGS_RELEASE "${CMAKE_${lang}_FLAGS_RELEASE}")
        endforeach()
    endif ()
endif ()

if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fobjc-arc")
endif()

add_library(MMDeployStaticModules INTERFACE)
add_library(MMDeployDynamicModules INTERFACE)
add_library(MMDeployLibs INTERFACE)

if ((cuda IN_LIST MMDEPLOY_TARGET_DEVICES) OR (trt IN_LIST MMDEPLOY_TARGET_BACKENDS))
    include(cmake/cuda.cmake NO_POLICY_SCOPE)
endif ()

# this must come after including cuda.cmake because policies in function scope is captured
# at function definition
include(cmake/MMDeploy.cmake)

add_subdirectory(csrc/mmdeploy)

if (MMDEPLOY_BUILD_SDK)
    if (NOT MMDEPLOY_BUILD_SDK_MONOLITHIC)
        install(TARGETS MMDeployStaticModules
                MMDeployDynamicModules
                MMDeployLibs
                EXPORT MMDeployTargets)
    endif ()

    if (MMDEPLOY_BUILD_TEST)
        add_subdirectory(tests/test_csrc)
    endif ()

    if (MMDEPLOY_BUILD_EXAMPLES)
        include(${CMAKE_SOURCE_DIR}/cmake/opencv.cmake)
        add_subdirectory(demo/csrc)
    endif ()

    # export MMDeploy package
    install(EXPORT MMDeployTargets
            FILE MMDeployTargets.cmake
            DESTINATION lib/cmake/MMDeploy)

    if (MMDEPLOY_SPDLOG_EXTERNAL)
        set(SPDLOG_DEPENDENCY "find_package(spdlog QUIET)")
    endif ()
    # append backend deps
    mmdeploy_add_deps(trt BACKENDS ${MMDEPLOY_TARGET_BACKENDS} DEPS TENSORRT CUDNN)
    mmdeploy_add_deps(ort BACKENDS ${MMDEPLOY_TARGET_BACKENDS} DEPS ONNXRUNTIME)
    mmdeploy_add_deps(ncnn BACKENDS ${MMDEPLOY_TARGET_BACKENDS} DEPS ncnn)
    mmdeploy_add_deps(openvino BACKENDS ${MMDEPLOY_TARGET_BACKENDS} DEPS InferenceEngine)
    if (NOT MMDEPLOY_SHARED_LIBS)
        mmdeploy_add_deps(pplnn BACKENDS ${MMDEPLOY_TARGET_BACKENDS} DEPS pplnn)
    endif ()
    mmdeploy_add_deps(snpe BACKENDS ${MMDEPLOY_TARGET_BACKENDS} DEPS snpe)
    mmdeploy_add_deps(rknn BACKENDS ${MMDEPLOY_TARGET_BACKENDS} DEPS rknn)

    include(CMakePackageConfigHelpers)
    # generate the config file that is includes the exports
    configure_package_config_file(${CMAKE_SOURCE_DIR}/cmake/MMDeployConfig.cmake.in
            "${CMAKE_CURRENT_BINARY_DIR}/MMDeployConfig.cmake"
            INSTALL_DESTINATION "lib/cmake"
            NO_SET_AND_CHECK_MACRO
            NO_CHECK_REQUIRED_COMPONENTS_MACRO
            )

    write_basic_package_version_file(
            "${CMAKE_CURRENT_BINARY_DIR}/MMDeployConfigVersion.cmake"
            VERSION "${MMDeploy_VERSION_MAJOR}.${MMDeploy_VERSION_MINOR}"
            COMPATIBILITY AnyNewerVersion
    )

    install(FILES
            ${CMAKE_CURRENT_BINARY_DIR}/MMDeployConfig.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/MMDeployConfigVersion.cmake
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/MMDeploy.cmake
            DESTINATION lib/cmake/MMDeploy
            )

    if (MSVC)
        install(FILES
                ${CMAKE_CURRENT_SOURCE_DIR}/cmake/loader.cpp.in
                DESTINATION lib/cmake/MMDeploy
                )
    endif ()

    install(DIRECTORY
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules
            DESTINATION lib/cmake/MMDeploy
            )

    if (${CMAKE_VERSION} VERSION_LESS "3.17.0")
        install(SCRIPT cmake/post-install.cmake)
    endif ()
endif ()


include_directories(${INCLUDE_DIRS})
find_package(Eigen3 3.3.2 REQUIRED)
find_package(GLEW REQUIRED)
if (UNIX)
    set(OpenGL_GL_PREFERENCE LEGACY)
endif ()
find_package(OpenGL REQUIRED)
find_path(RESTBED_INCLUDE_DIRS "restbed")
find_library(RESTBED_LIBRARY NAMES "librestbed.a" "restbed.lib")
find_package(glfw3 3.1.2 REQUIRED)
#find_package(OpenCV 4.3.0 REQUIRED COMPONENTS core imgproc highgui imgcodecs calib3d features2d xfeatures2d OPTIONAL_COMPONENTS cudafeatures2d)

if (USE_AZURE_KINECT)
    find_package(k4a 1.3.0 REQUIRED)
endif ()

if (USE_REALSENSE)
    find_package(realsense2 REQUIRED)
endif ()

find_package(OpenMP)
if (OPENMP_FOUND)
    if (WIN32)
        set(CMAKE_CXX_FLAGS_RELEASE
                "${CMAKE_CXX_FLAGS_RELEASE} -openmp:experimental")
        set(CMAKE_C_FLAGS_RELEASE
                "${CMAKE_C_FLAGS_RELEASE} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS_RELEASE
                "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS
                "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    else ()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS
                "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endIf ()
endif ()


# Compiler Settings
# =============================================================================

# Default settings

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-march HAS_MARCH)
check_cxx_compiler_flag(-mtune HAS_MTUNE)
if (HAS_MARCH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif ()
if (HAS_MTUNE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native")
endif ()

# Debug settings
check_cxx_compiler_flag(-ggdb HAS_GGDB)
check_cxx_compiler_flag(-Z7 HAS_Z7)
check_cxx_compiler_flag(-ftrapv HAS_FTRAPV)
check_cxx_compiler_flag(-Od HAS_OD)
if (HAS_GGDB)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb")
elseif (HAS_Z7)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Z7")
endif ()
if (HAS_FTRAPV)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ftrapv")
endif ()
if (HAS_OD)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Od")
endif ()
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")

# Release settings
check_cxx_compiler_flag(-Ob3 HAS_OB3)
check_cxx_compiler_flag(-O3 HAS_O3)
check_cxx_compiler_flag(-Ob2 HAS_OB2)
check_cxx_compiler_flag(-O2 HAS_O2)
if (HAS_OB3)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ob3")
elseif (HAS_O3)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
elseif (HAS_OB2)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ob2")
elseif (HAS_O2)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
endif ()
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")


# Definitions
# =============================================================================
if (USE_AZURE_KINECT)
    add_definitions( -DUSE_AZURE_KINECT=TRUE )
endif ()
if (USE_REALSENSE)
    add_definitions( -DUSE_REALSENSE=TRUE )
endif ()
if(OpenCV_CUDA_VERSION)
    add_definitions( -DUSE_CUDA=TRUE ) 
endif ()
if (USE_GTEST)
    set (GTEST_LIBRARY path/to/googletest/build/lib/Release/gtest.lib)
    set (GTEST_MAIN_LIBRARY path/to/googletest/build/lib/Release/gtest_main.lib)
    set (GTEST_INCLUDE_DIR path/to/googletest/googletest/include)
    add_definitions( -DUSE_GTEST=TRUE )
endif ()


# Directories
# =============================================================================
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(examples)
add_subdirectory(doc)
