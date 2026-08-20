cmake_minimum_required(VERSION 3.2)

project(LAB6)

<<<<<<< HEAD
<<<<<<< HEAD
=======
set(CMAKE_CXX_COMPILER clang++)

=======
<<<<<<< HEAD
>>>>>>> 39ae558f7f62c32f1ca7af80926efdfff52340ec
=======
set(CMAKE_CXX_COMPILER clang++)

>>>>>>> coursework3_array
<<<<<<< HEAD
=======
>>>>>>> hand-in
>>>>>>> 39ae558f7f62c32f1ca7af80926efdfff52340ec
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -pedantic -O0 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -Wextra -pedantic -O3")

add_executable(main main.cpp)
