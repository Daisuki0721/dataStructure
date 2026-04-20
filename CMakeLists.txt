set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

cmake_minimum_required(VERSION 3.10)
project(dataStructure)

set(CMAKE_CXX_STANDARD 17)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 查找 Qt5 - 支持自定义路径
if(NOT Qt5_DIR)
    if(APPLE)
        # macOS 下自动查找 Homebrew 安装的 Qt5
        execute_process(
            COMMAND brew --prefix qt@5
            OUTPUT_VARIABLE QT5_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(QT5_PATH)
            set(Qt5_DIR "${QT5_PATH}/lib/cmake/Qt5")
        endif()
    endif()
endif()

find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)

file(GLOB HDRS "include/*.hpp")

include_directories(include ${OpenCV_INCLUDE_DIRS})

# 命令行版本
add_executable(task src/task1.cpp)
target_link_libraries(task ${OpenCV_LIBS})

# 任务2命令行版本
add_executable(task2 src/task2.cpp)
target_link_libraries(task2 ${OpenCV_LIBS})

# 任务3命令行版本（排序查找 + 哈夫曼）
add_executable(task3 src/task3.cpp)
target_link_libraries(task3 ${OpenCV_LIBS})

# GUI 版本
set(GUI_HEADERS include/image_gui.hpp)
set(GUI_SOURCES src/main_gui.cpp src/image_gui.cpp)

# 为 GUI 版本启用 MOC
set_source_files_properties(${GUI_HEADERS} PROPERTIES AUTOMOC ON)

add_executable(task_gui ${GUI_SOURCES} ${GUI_HEADERS})
set_target_properties(task_gui PROPERTIES AUTOMOC ON)
target_link_libraries(task_gui ${OpenCV_LIBS} Qt5::Core Qt5::Gui Qt5::Widgets)