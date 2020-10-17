project(lab-4)
cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(SOURCES main.cpp Dijkstra.cpp)

add_executable(lab-4 ${SOURCES})