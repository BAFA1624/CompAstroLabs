# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.26.3/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.26.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/ben/Documents/gitrepos.nosync/CompAstroLabs/coursework3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/ben/Documents/gitrepos.nosync/CompAstroLabs/coursework3/release

# Include any dependencies generated for this target.
include CMakeFiles/24697.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/24697.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/24697.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/24697.dir/flags.make

CMakeFiles/24697.dir/24697.cpp.o: CMakeFiles/24697.dir/flags.make
CMakeFiles/24697.dir/24697.cpp.o: /Users/ben/Documents/gitrepos.nosync/CompAstroLabs/coursework3/24697.cpp
CMakeFiles/24697.dir/24697.cpp.o: CMakeFiles/24697.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/ben/Documents/gitrepos.nosync/CompAstroLabs/coursework3/release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/24697.dir/24697.cpp.o"
	g++-12 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/24697.dir/24697.cpp.o -MF CMakeFiles/24697.dir/24697.cpp.o.d -o CMakeFiles/24697.dir/24697.cpp.o -c /Users/ben/Documents/gitrepos.nosync/CompAstroLabs/coursework3/24697.cpp

CMakeFiles/24697.dir/24697.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/24697.dir/24697.cpp.i"
	g++-12 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/ben/Documents/gitrepos.nosync/CompAstroLabs/coursework3/24697.cpp > CMakeFiles/24697.dir/24697.cpp.i

CMakeFiles/24697.dir/24697.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/24697.dir/24697.cpp.s"
	g++-12 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/ben/Documents/gitrepos.nosync/CompAstroLabs/coursework3/24697.cpp -o CMakeFiles/24697.dir/24697.cpp.s

# Object files for target 24697
24697_OBJECTS = \
"CMakeFiles/24697.dir/24697.cpp.o"

# External object files for target 24697
24697_EXTERNAL_OBJECTS =

24697: CMakeFiles/24697.dir/24697.cpp.o
24697: CMakeFiles/24697.dir/build.make
24697: CMakeFiles/24697.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/ben/Documents/gitrepos.nosync/CompAstroLabs/coursework3/release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable 24697"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/24697.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/24697.dir/build: 24697
.PHONY : CMakeFiles/24697.dir/build

CMakeFiles/24697.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/24697.dir/cmake_clean.cmake
.PHONY : CMakeFiles/24697.dir/clean

CMakeFiles/24697.dir/depend:
	cd /Users/ben/Documents/gitrepos.nosync/CompAstroLabs/coursework3/release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/ben/Documents/gitrepos.nosync/CompAstroLabs/coursework3 /Users/ben/Documents/gitrepos.nosync/CompAstroLabs/coursework3 /Users/ben/Documents/gitrepos.nosync/CompAstroLabs/coursework3/release /Users/ben/Documents/gitrepos.nosync/CompAstroLabs/coursework3/release /Users/ben/Documents/gitrepos.nosync/CompAstroLabs/coursework3/release/CMakeFiles/24697.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/24697.dir/depend

