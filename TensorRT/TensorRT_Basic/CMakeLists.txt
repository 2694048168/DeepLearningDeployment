project(TensorBasic
    VERSION 1.0.1
    DESCRIPTION "The basic TensorRT API via Modern C++"
    HOMEPAGE_URL "https://github.com/2694048168/C-and-C-plus-plus"
    LANGUAGES CXX C CUDA
)

add_executable(00_basicTensorRT 00_basicTensorRT.cpp)
target_link_libraries(00_basicTensorRT
PRIVATE
    nvinfer
)

add_executable(01_inferEngine 01_inferEngine.cpp)
target_link_libraries(01_inferEngine
PRIVATE
    nvinfer
    CUDA::cudart
)

add_executable(02_dynamicShape 02_dynamicShape.cpp)
target_link_libraries(02_dynamicShape
PRIVATE
    nvinfer
    CUDA::cudart
)

add_executable(05_onnxParse 05_onnxParse.cpp)
target_link_libraries(05_onnxParse
PRIVATE
    nvinfer
    nvonnxparser
    CUDA::cudart
)

add_executable(06_pluginImpl 06_pluginImpl.cpp)
target_link_libraries(06_pluginImpl
PRIVATE
    nvinfer
    nvonnxparser
    CUDA::cudart
)

add_executable(07_int8_quantization 07_int8_quantization.cpp)
target_link_libraries(07_int8_quantization
PRIVATE
    nvinfer
    nvonnxparser
    CUDA::cudart
)
