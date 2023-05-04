cmake_minimum_required(VERSION 3.2)

project(LAB6)

<<<<<<< HEAD
set(CMAKE_CXX_COMPILER clang++)
=======
#set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_COMPILER g++-12)
>>>>>>> 39ae558f7f62c32f1ca7af80926efdfff52340ec

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

<<<<<<< HEAD
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -pedantic -O0 -fstack-protector-all -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -Wextra -pedantic -O3")
=======
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -pedantic -O0 -fopenmp -fstack-protector-all -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -Wextra -pedantic -fopenmp -O3")
>>>>>>> 39ae558f7f62c32f1ca7af80926efdfff52340ec

add_executable(24697 24697.cpp)
