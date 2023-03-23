cmake_minimum_required(VERSION 3.1)
#创建工程
project(puzzle)
enable_language(CUDA)#开启CUDA的支持选项
set(CMAKE_CXX_FLAGS "-std=c++14")#C++版本为14
#set(CMAKE_BUILD_TYPE "Debug")#debug模式
#set(CMAKE_CUDA_FLAGS "-g -G")
set(CMAKE_BUILD_TYPE "Realease")
find_package(OpenCV REQUIRED) #找OpenCV安装路径
find_package(CUDA REQUIRED) #找CUDA安装路径

include_directories(${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) #加载OpenCV的头文件

# add_executable(test main.cpp) #将程序生成可执行文件
# target_link_libraries(test ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES}) #链接lib文件到可执行文件中

add_executable(${PROJECT_NAME} puzzle.cu) #将程序生成可执行文件
target_link_libraries(${PROJECT_NAME}
    ${CUDA_LIBRARIES} 
    ${OpenCV_LIBRARIES}
    )#链接lib文件到可执行文件中
