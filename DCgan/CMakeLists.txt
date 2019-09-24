cmake_minimum_required(VERSION 3.7)
project(DCgan)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(DCgan ${SOURCE_FILES})

set(CMAKE_PREFIX_PATH
        "/home/lab-xjh/app/libtorch-shared/libtorch")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
target_link_libraries(DCgan ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})
include_directories(DCgan ${TORCH_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

