cmake_minimum_required(VERSION 3.15)
project(matrix_mult)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
add_executable(matrix_mult Main.cpp)