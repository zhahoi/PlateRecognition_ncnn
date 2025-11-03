cmake_minimum_required(VERSION 3.10.1)
project(plate_detection_recognition)

set(CMAKE_CXX_STANDARD 17)

# 依赖查找
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

set(ncnn_DIR "/home/dell/Softwares/ncnn/build/install/lib/cmake/ncnn")
find_package(ncnn REQUIRED)

# 设置模型文件路径
set(PLATE_RECOGNITION_PARAM "${CMAKE_SOURCE_DIR}/../plate_recognition/weights/plate_recognition_color.ncnn.param")
set(PLATE_RECOGNITION_BIN "${CMAKE_SOURCE_DIR}/../plate_recognition/weights/plate_recognition_color.ncnn.bin")
set(YOLOV5_PARAM "${CMAKE_SOURCE_DIR}/../yolov5_platedetector/weights/yolov5plate.ncnn.param")
set(YOLOV5_BIN "${CMAKE_SOURCE_DIR}/../yolov5_platedetector/weights/yolov5plate.ncnn.bin")
set(YOLOV7_PARAM "${CMAKE_SOURCE_DIR}/../yolov7_platedetector/weights/yolov7plate.ncnn.param")
set(YOLOV7_BIN "${CMAKE_SOURCE_DIR}/../yolov7_platedetector/weights/yolov7plate.ncnn.bin")

# 传递路径给编译器
add_definitions(-DPLATE_RECOGNITION_PARAM="${PLATE_RECOGNITION_PARAM}")
add_definitions(-DPLATE_RECOGNITION_BIN="${PLATE_RECOGNITION_BIN}")
add_definitions(-DYOLOV5_PARAM="${YOLOV5_PARAM}")
add_definitions(-DYOLOV5_BIN="${YOLOV5_BIN}")
add_definitions(-DYOLOV7_PARAM="${YOLOV7_PARAM}")
add_definitions(-DYOLOV7_BIN="${YOLOV7_BIN}")

# 添加子项目（确保它们编译为库）
add_subdirectory(../yolov5_platedetector yolov5_platedetector)
add_subdirectory(../yolov7_platedetector yolov7_platedetector)
add_subdirectory(../plate_recognition plate_recognition)

# 可执行文件
add_executable(plate_detection_recognition 
    src/main.cpp 
    src/plate_detection_recognition.cpp
)

# 编译选项
target_compile_options(plate_detection_recognition PRIVATE -O2 -pthread)
target_include_directories(plate_detection_recognition PRIVATE 
    include
    ../yolov5_platedetector/include
    ../yolov7_platedetector/include
    ../plate_recognition/include
)

# 链接库
target_link_libraries(plate_detection_recognition PRIVATE
    ${OpenCV_LIBS}
    ncnn
    OpenMP::OpenMP_CXX
    yolov5_plate
    yolov7_plate
    plate_recognition
)

# 复制资源文件
file(COPY font/ data/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR})