cmake_minimum_required(VERSION 3.10.1)
project(yolov5_plate)

set(CMAKE_CXX_STANDARD 17)
add_definitions(-std=c++17)
add_definitions(-pthread)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# 设置 Release 优化
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

# OpenMP 支持
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message("OpenMP found")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# OpenCV
find_package(OpenCV REQUIRED)

# ncnn
set(ncnn_DIR "/home/dell/Softwares/ncnn/build/install/lib/cmake/ncnn")
find_package(ncnn REQUIRED)
if(NOT TARGET ncnn)
    message(WARNING "ncnn NOT FOUND! Please set ncnn_DIR environment variable")
else()
    message("ncnn found")
endif()

# Include paths
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

# 源文件收集
file(GLOB SRC_FILES src/*.cpp)

set(SOURCE_FILES src/main.cpp src/platedetector_yolov5.cpp)

add_library(yolov5_plate STATIC ${SOURCE_FILES})

target_link_libraries(
    yolov5_plate
    ${OpenCV_LIBS}
    ncnn
    OpenMP::OpenMP_CXX
)

target_include_directories(yolov5_plate PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)