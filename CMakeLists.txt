cmake_minimum_required(VERSION 3.25.0)

if(UNIX)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

if(WIN32)
    set(CMAKE_CUDA_COMPILER "C://Program Files//NVIDIA GPU Computing Toolkit//CUDA/v12.4//bin//nvcc.exe")
endif()

# 判断操作系统
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    message(STATUS "Linux platform: ${CMAKE_SYSTEM_NAME}")
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(STATUS "Windows platform: ${CMAKE_SYSTEM_NAME}")
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    message(STATUS "Darwin platform: ${CMAKE_SYSTEM_NAME}")
else()
    message(STATUS "other platform: ${CMAKE_SYSTEM_NAME}")
endif(CMAKE_SYSTEM_NAME MATCHES "Linux")

project(ModelNetworkDeployment
    VERSION 1.0.1
    DESCRIPTION "The Models and Networks Deployment"
    HOMEPAGE_URL "https://github.com/2694048168/ModelDeployment"
    LANGUAGES C CXX CUDA
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 只启用 ISO C++ 标准的编译器标志, 而不使用特定编译器的扩展
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# 只启用 ISO C++ 标准的编译器标志, 而不使用特定编译器的扩展
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 只启用 CUDA 标准的编译器标志, 而不使用特定编译器的扩展
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# add_compile_definitions(<definition> ...)
# 将预编译参数添加到源文件的编译中，对下级子目录同样有效
# 预编译命令会添加到COMPILE_DEFINITIONS目录属性中

# 判断编译器
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    message(STATUS "LLVM Clang compiler: ${CMAKE_CXX_COMPILER_ID}")
    # lots of warnings and all warnings as errors
    # add_compile_options(-Wall -Wextra -pedantic -Werror -finput-charset=utf-8 -fexec-charset=utf-8)
    # add_compile_options(-finput-charset=utf-8 -fexec-charset=utf-8)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    message(STATUS "GNU GCC compiler: ${CMAKE_CXX_COMPILER_ID}")
    # lots of warnings and all warnings as errors
    # add_compile_options(-Wall -Wextra -pedantic -Werror -finput-charset=utf-8 -fexec-charset=utf-8)
    # add_compile_options(-finput-charset=utf-8 -fexec-charset=utf-8)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    message(STATUS "Intel C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    message(STATUS "Visual Studio C++ compiler(MSVC): ${CMAKE_CXX_COMPILER_ID}")
    # add_compile_options(<option> ...)
    # option:编译选项,注意对于不同编译器,支持的选项可能不一样
    # 将编译器选项添加到当前及子目录的源文件的编译中
    # warning level 4 and all warnings as errors
    # =====If you want to set both the source character set and
    # the execution character set to UTF-8, you can use the /utf-8
    # compiler option as a shortcut. It's equivalent to
    # /source-charset:utf-8 /execution-charset:utf-8 on the command line.
    # add_compile_options(/W4 /WX /source-charset:utf-8 /execution-charset:utf-8)
    # add_compile_options(/W4 /WX /utf-8)
    # add_compile_options(/utf-8)
endif()

# ---------- 条件判断 Debug & Release 模式下的编译指令 ----------
# if(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Debug"))
# elseif(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Release"))
# else()
# endif()

# ------------------------------------------------------
# 设置一系列的变量, 顶层 CMake 的变量可以被底层 CMake 访问
# 设置输出二进制路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# ------- OpenCV ---------
# 要检查 CUDA 是否可用(nvcc), 可使用 CheckLanguage:
include(CheckLanguage)
check_language(CUDA)
message(STATUS "--->There is CUDA toolkits: ${CMAKE_CUDA_COMPILER}")
message(STATUS "--->The CUDA toolkits is: ${CMAKE_CUDA_COMPILER_ID}")
message(STATUS "--->The CUDA Version is: ${CMAKE_CUDA_COMPILER_VERSION}")
 
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4")
set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4")
find_package(CUDAToolkit)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
message(STATUS "--->CUDA include: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

# -------------------------------------------
# external 第三方库
# ------- OpenCV ---------
set(OpenCV_Include "D:/DevelopTools/OpenCV410/opencv/build/include")
set(OpenCV_Lib "D:/DevelopTools/OpenCV410/opencv/build/x64/vc16/lib")
include_directories(${OpenCV_Include})
link_directories(${OpenCV_Lib})
file(GLOB OpenCV_DLL_FILE
"D:/DevelopTools/OpenCV410/opencv/build/x64/vc16/bin/opencv_world4100d.dll"
"D:/DevelopTools/OpenCV410/opencv/build/x64/vc16/bin/opencv_world4100.dll"
)

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(STATUS "====> On Windows")
    if(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Debug"))
        file(COPY ${OpenCV_DLL_FILE} DESTINATION "${EXECUTABLE_OUTPUT_PATH}/Debug")
    elseif(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Release"))
        file(COPY ${OpenCV_DLL_FILE} DESTINATION "${EXECUTABLE_OUTPUT_PATH}/Release")
    else()
        file(COPY ${OpenCV_DLL_FILE} DESTINATION "${EXECUTABLE_OUTPUT_PATH}/Debug")
    endif()
else() # Linux or Darwin(Mac OS)
    message(STATUS "====> On Linux or Mac")
    file(COPY ${OpenCV_DLL_FILE} DESTINATION "${EXECUTABLE_OUTPUT_PATH}")
endif()
# --------------------------------------------------
# ------- TensorRT ---------
set(TensorRT_Include "external/TensorRT/include")
set(TensorRT_Lib "external/TensorRT/lib")
include_directories(${TensorRT_Include})
link_directories(${TensorRT_Lib})
file(GLOB TensorRT_DLL_FILE
"${TensorRT_Lib}/*.dll"
)

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(STATUS "====> On Windows")
    if(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Debug"))
        file(COPY ${TensorRT_DLL_FILE} DESTINATION "${EXECUTABLE_OUTPUT_PATH}/Debug")
    elseif(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Release"))
        file(COPY ${TensorRT_DLL_FILE} DESTINATION "${EXECUTABLE_OUTPUT_PATH}/Release")
    else()
        file(COPY ${TensorRT_DLL_FILE} DESTINATION "${EXECUTABLE_OUTPUT_PATH}/Debug")
    endif()
else() # Linux or Darwin(Mac OS)
    message(STATUS "====> On Linux or Mac")
    file(COPY ${TensorRT_DLL_FILE} DESTINATION "${EXECUTABLE_OUTPUT_PATH}")
endif()
# --------------------------------------------------

# 添加 CMake 子目录, 寻找该目录下的 CMakeLists.txt
add_subdirectory(TensorRT/CUDA_DriverAPI/)
add_subdirectory(TensorRT/CUDA_RuntimeAPI/)
add_subdirectory(TensorRT/TensorRT_Basic/)
