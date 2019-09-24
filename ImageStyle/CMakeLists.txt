cmake_minimum_required(VERSION 3.7)
project(ImageStyle)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(ImageStyle ${SOURCE_FILES})
set(CMAKE_PREFIX_PATH
        "/home/lab-xjh/app/libtorch-shared/libtorch")

find_package(Torch REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS})
target_link_libraries(ImageStyle ${TORCH_LIBRARIES})

find_package(OpenCV REQUIRED )

foreach(lp ${OpenCV_INCLUDE_DIRS})
    message(STATUS "lib:" ${lp})
endforeach(lp)

target_link_libraries(ImageStyle ${OpenCV_LIBRARIES} )