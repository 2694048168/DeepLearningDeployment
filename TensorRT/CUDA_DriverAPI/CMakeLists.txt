project(CudaDriver
    VERSION 1.0.1
    DESCRIPTION "The CUDA Driver API via Modern C++"
    HOMEPAGE_URL "https://github.com/2694048168/C-and-C-plus-plus"
    LANGUAGES CXX C
)

add_executable(00_cuinit 00_cuinit.cpp)
target_link_libraries(00_cuinit
PRIVATE
    CUDA::cuda_driver
)

add_executable(01_checkDeviceError 01_checkDeviceError.cpp)
target_link_libraries(01_checkDeviceError
PRIVATE
    CUDA::cuda_driver
)

add_executable(02_checkDeviceErrorAdvanced 02_checkDeviceErrorAdvanced.cpp)
target_link_libraries(02_checkDeviceErrorAdvanced
PRIVATE
    CUDA::cuda_driver
)

add_executable(03_deviceContext 03_deviceContext.cpp)
target_link_libraries(03_deviceContext
PRIVATE
    CUDA::cuda_driver
)

add_executable(04_memoryAlloc 04_memoryAlloc.cpp)
target_link_libraries(04_memoryAlloc
PRIVATE
    CUDA::cuda_driver
)
