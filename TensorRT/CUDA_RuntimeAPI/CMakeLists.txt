project(CudaRuntime
    VERSION 1.0.1
    DESCRIPTION "The CUDA Runtime API via Modern C++"
    HOMEPAGE_URL "https://github.com/2694048168/C-and-C-plus-plus"
    LANGUAGES CXX C CUDA
)

add_executable(00_checkDeviceError 00_checkDeviceError.cpp)
target_link_libraries(00_checkDeviceError
PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
)

add_executable(01_memoryAlloc 01_memoryAlloc.cpp)
target_link_libraries(01_memoryAlloc
PRIVATE
    CUDA::cudart
)

add_executable(02_asyncStream 02_asyncStream.cpp)
target_link_libraries(02_asyncStream
PRIVATE
    CUDA::cudart
)

# add_library(kernel SHARED 03_kernel.cu)
add_library(kernel 03_kernel.cu)
target_link_libraries(kernel CUDA::cudart)
add_executable(03_kernelFunction 03_kernelFunction.cpp)
target_link_libraries(03_kernelFunction
PRIVATE
    kernel
)

add_library(shared_kernel 04_sharedMemory.cu)
target_link_libraries(shared_kernel CUDA::cudart)
add_executable(04_sharedMemory 04_sharedMemory.cpp)
target_link_libraries(04_sharedMemory
PRIVATE
    shared_kernel
)

message(STATUS "----> the config build type == ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Debug"))
    set(OpenCV_LIB_LINK opencv_world4100d)
elseif(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Release"))
    set(OpenCV_LIB_LINK opencv_world4100)
else()
    set(OpenCV_LIB_LINK opencv_world4100d)
endif()
# add_library(warpAffine 05_warpAffine.cu)
# target_link_libraries(warpAffine CUDA::cudart)
add_executable(05_warpAffine 05_warpAffine.cpp)
target_link_libraries(05_warpAffine
PRIVATE
    # warpAffine
    CUDA::cudart
    ${OpenCV_LIB_LINK}
)

add_library(cudaKernel STATIC 06_cudaError.cu)
target_link_libraries(cudaKernel CUDA::cudart)
add_executable(06_cudaError 06_cudaError.cpp)
target_link_libraries(06_cudaError
PRIVATE
    cudaKernel
    CUDA::cudart
)
