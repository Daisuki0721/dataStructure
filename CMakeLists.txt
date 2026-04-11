set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

cmake_minimum_required(VERSION 3.10)
project(dataStructure)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

file(GLOB SRCS "src/*.cpp" "main.cpp")

file(GLOB HDRS "include/*.h" "include/*.hpp")

include_directories(include ${OpenCV_INCLUDE_DIRS})

add_executable(task ${SRCS} ${HDRS})

target_link_libraries(task ${OpenCV_LIBS})