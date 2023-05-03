cmake_minimum_required(VERSION 3.2)

project(LAB6)

set(CMAKE_CXX_COMPILER clang++)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -pedantic -O0 -fstack-protector-all -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -Wextra -pedantic -O3")

add_executable(24697 24697.cpp)
