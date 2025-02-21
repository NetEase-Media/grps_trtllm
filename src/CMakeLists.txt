# build grps server user customized library and unit test binary.

######################## Build user customized lib [BEGIN] ########################
include(${CMAKE_CURRENT_SOURCE_DIR}/set_ifndef.cmake)

set_ifndef(TRTLLM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/TensorRT-LLM)
include(${TRTLLM_DIR}/cpp/cmake/modules/find_library_create_target.cmake)

add_compile_options("-DENABLE_MULTI_DEVICE=1")
# https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html
option(USE_CXX11_ABI "Using CXX11 ABI of libstdc++" ON)
message(STATUS "USE_CXX11_ABI: ${USE_CXX11_ABI}")
if (USE_CXX11_ABI)
    add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=1")
else ()
    add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=0")
endif ()

find_library(
        CUDNN_LIB cudnn
        HINTS ${CUDA_TOOLKIT_ROOT_DIR} ${CUDNN_ROOT_DIR}
        PATH_SUFFIXES lib64 lib)
find_library(
        CUBLAS_LIB cublas
        HINTS ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib64 lib lib/stubs)
find_library(
        CUBLASLT_LIB cublasLt
        HINTS ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib64 lib lib/stubs)
find_library(
        CUDART_LIB cudart
        HINTS ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64)
find_library(
        CUDA_DRV_LIB cuda
        HINTS ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/stubs lib64/stubs)
find_library(
        NVIDIA_ML_LIB nvidia-ml
        HINTS ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/stubs lib64/stubs)
set(CUDA_LIBRARIES ${CUDART_LIB} ${NVIDIA_ML_LIB})

# NCCL dependencies
set_ifndef(NCCL_LIB_DIR /usr/lib/x86_64-linux-gnu/)
set_ifndef(NCCL_INCLUDE_DIR /usr/include/)
find_library(NCCL_LIB nccl HINTS ${NCCL_LIB_DIR})

# TRT_LIB_DIR and TRT_INCLUDE_DIR should be aligned with the path in the
# environment_setup.sh script
set_ifndef(TRT_LIB_DIR
        /usr/local/tensorrt/targets/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/lib)
set_ifndef(
        TRT_INCLUDE_DIR
        /usr/local/tensorrt/include)

set(TRT_LIB nvinfer)
find_library_create_target(${TRT_LIB} nvinfer SHARED ${TRT_LIB_DIR})

file(STRINGS "${TRT_INCLUDE_DIR}/NvInferVersion.h" VERSION_STRINGS
        REGEX "#define NV_TENSORRT_.*")
foreach (TYPE MAJOR MINOR PATCH BUILD)
    string(REGEX MATCH "NV_TENSORRT_${TYPE} [0-9]+" TRT_TYPE_STRING
            ${VERSION_STRINGS})
    string(REGEX MATCH "[0-9]+" TRT_${TYPE} ${TRT_TYPE_STRING})
endforeach (TYPE)

foreach (TYPE MAJOR MINOR PATCH)
    string(REGEX MATCH "NV_TENSORRT_${TYPE} [0-9]+" TRT_TYPE_STRING
            ${VERSION_STRINGS})
    string(REGEX MATCH "[0-9]+" TRT_SO_${TYPE} ${TRT_TYPE_STRING})
endforeach (TYPE)

set(TRT_VERSION
        "${TRT_MAJOR}.${TRT_MINOR}.${TRT_PATCH}"
        CACHE STRING "TensorRT project version")
set(TRT_SOVERSION
        "${TRT_SO_MAJOR}"
        CACHE STRING "TensorRT library so version")
message(
        STATUS
        "Building for TensorRT version: ${TRT_VERSION}, library version: ${TRT_SOVERSION}"
)
include_directories(${TRT_INCLUDE_DIR})
message(STATUS "TRT_INCLUDE_DIR: ${TRT_INCLUDE_DIR}, TRT_LIB_DIR: ${TRT_LIB_DIR}, TRT_LIB: ${TRT_LIB}")

if (${TRT_MAJOR} GREATER_EQUAL 10)
    add_definitions("-DTRT_LLM_USE_DIM64")
    message(
            STATUS "TensorRT version ${TRT_MAJOR} >= 10, int64 dimension is enabled")
endif ()

#
# Dependencies
#
# FetchContent requires us to include the transitive closure of all repos that
# we depend on so that we can override the tags.
#
include(FetchContent)

set(COMMON_HEADER_DIRS ${PROJECT_SOURCE_DIR}/src ${CUDA_PATH}/include)
list(APPEND COMMON_HEADER_DIRS ${TORCH_INCLUDE_DIRS} ${TRT_INCLUDE_DIR})
message(STATUS "COMMON_HEADER_DIRS: ${COMMON_HEADER_DIRS}")

set(COMMON_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/constants.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/llm_styler.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/trtllm_model_state.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/trtllm_model_instance.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/tensorrt/trt_inferer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/vit/vit.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/vit/internvl2_vit.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/vit/qwenvl_vit.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/vit/qwen2vl_vit.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/vit/janus_pro_vit.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer.cc
)

add_library(grps-tensorrt-llm-common SHARED ${COMMON_SRCS})

target_include_directories(
        grps-tensorrt-llm-common
        PUBLIC ${TRTLLM_DIR}/cpp
        ${TRTLLM_DIR}/cpp/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CUDA_INCLUDE_DIRS}
        ${CUDNN_ROOT_DIR}/include
        ${NCCL_INCLUDE_DIR}
        ${3RDPARTY_DIR}/cutlass/include
        ${MPI_INCLUDE_PATH}
        ${COMMON_HEADER_DIRS}
        # ${jinja2cpp_INCLUDE_DIRS}
)

target_compile_features(grps-tensorrt-llm-common PRIVATE cxx_std_17)

set(COMPILE_OPTIONS
        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
        -Wall
        -Wextra
        -Wno-unused-parameter
        -Wno-deprecated-declarations
        -Wno-type-limits>
        $<$<CXX_COMPILER_ID:MSVC>:/Wall
        /D_WIN32_WINNT=0x0A00
        /EHsc>)

target_compile_options(grps-tensorrt-llm-common PRIVATE ${COMPILE_OPTIONS})

add_library(tensorrt_llm SHARED IMPORTED)
set_property(
        TARGET tensorrt_llm
        PROPERTY IMPORTED_LOCATION
        "/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/libtensorrt_llm.so")

add_library(nvinfer_plugin_tensorrt_llm SHARED IMPORTED)
set_property(
        TARGET nvinfer_plugin_tensorrt_llm
        PROPERTY
        IMPORTED_LOCATION
        "/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/libnvinfer_plugin_tensorrt_llm.so"
)

target_link_libraries(grps-tensorrt-llm-common PUBLIC
        tensorrt_llm
        tokenizers_cpp
        ${MPI_LIBRARIES}
        ${CUDA_LIBRARIES}
        nvinfer
        nvinfer_plugin_tensorrt_llm
        opencv_core
        opencv_imgproc
        opencv_highgui
        opencv_imgcodecs
)
if (PILLOW_RESIZE_ENABLE)
    target_link_libraries(grps-tensorrt-llm-common PUBLIC PillowResize)
endif ()

set_target_properties(
        grps-tensorrt-llm-common PROPERTIES POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME grps_tensorrtllm_common)

add_library(grps-server-customized SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/customized_inferer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/grps_server_customized.cc
)

target_compile_options(grps-server-customized PRIVATE ${COMPILE_OPTIONS})
set(THIRD_PARTY_LIB_DIR ${CMAKE_SOURCE_DIR}/third_party/lib/)
target_link_directories(grps-server-customized BEFORE PRIVATE ${THIRD_PARTY_LIB_DIR} ${DEPEND_LINK_DIRECTORIES})
target_link_libraries(grps-server-customized PRIVATE
        grps-tensorrt-llm-common
        libgrps-server-framework.a
        ${MPI_LIBRARIES}
        brpc
        grpc++_unsecure
        gtest
        protobuf
        yaml-cpp
        log4cxx
        aprutil-1
        apr-1
        pthread
        boost_system
        boost_thread
        # jinja2cpp
        dl
)

if (CUDA_ENABLE)
    target_link_libraries(grps-server-customized PRIVATE ${CUDA_LIBRARIES} ${CUDA_NVML_LIBRARIES})
    set_target_properties(grps-server-customized PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()
if (TF_ENABLE)
    target_link_directories(grps-server-customized BEFORE PUBLIC ${TF_LIB_DIR})
    target_link_libraries(grps-server-customized PRIVATE ${TF_LIBRARIES})
endif ()
if (TORCH_ENABLE)
    target_link_libraries(grps-server-customized PRIVATE ${TORCH_LIBRARIES})
endif ()
if (TRT_ENABLE)
    target_link_libraries(grps-server-customized PRIVATE ${TRT_LIBRARIES})
endif ()

set_target_properties(
        grps-server-customized
        PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME grps-server-customized
        LINK_FLAGS
        "-Wl,--no-undefined"
)

# install grps-server-customized lib.
install(TARGETS grps-server-customized grps-tensorrt-llm-common
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
)

# install third party lib.
install(DIRECTORY ${THIRD_PARTY_LIB_DIR}
        DESTINATION lib
        FILES_MATCHING PATTERN "*.so*"
)

######################## Build user customized lib [END] ########################

######################## Build unittest [BEGIN] ########################
add_executable(unit_test
        main.cc
)

# -flto -fno-fat-lto-objects
# set_property(TARGET unit_test PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
target_link_directories(unit_test BEFORE PUBLIC ${DEPEND_LINK_DIRECTORIES})
target_link_libraries(unit_test PUBLIC
        "-Wl,--whole-archive"
        grps-server-customized
        libgrps-server-framework.a
        "-Wl,--no-whole-archive"
        tokenizers_cpp
        ${MPI_LIBRARIES}
        brpc
        grpc++_unsecure
        gtest
        protobuf
        yaml-cpp
        log4cxx
        aprutil-1
        apr-1
        pthread
        boost_system
        boost_thread
        # jinja2cpp
        dl
)
if (CUDA_ENABLE)
    target_link_libraries(unit_test PUBLIC ${CUDA_LIBRARIES} ${CUDA_NVML_LIBRARIES})
endif ()
if (TF_ENABLE)
    target_link_directories(unit_test BEFORE PUBLIC ${TF_LIB_DIR})
    target_link_libraries(unit_test PUBLIC ${TF_LIBRARIES})
endif ()
if (TORCH_ENABLE)
    target_link_libraries(unit_test PUBLIC ${TORCH_LIBRARIES})
endif ()
if (TRT_ENABLE)
    target_link_libraries(unit_test PUBLIC ${TRT_LIBRARIES})
endif ()


target_link_options(unit_test BEFORE PUBLIC
)

install(TARGETS unit_test
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
)
######################## Build unittest [END] ########################
