# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/workspaces/python/pyg/gnn_general_ext/build

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/workspaces/python/pyg/gnn_general_ext/build

# Include any dependencies generated for this target.
include CMakeFiles/gnn_ext.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/gnn_ext.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/gnn_ext.dir/flags.make

CMakeFiles/gnn_ext.dir/root/workspaces/python/pyg/gnn_general_ext/src/main.cpp.o: CMakeFiles/gnn_ext.dir/flags.make
CMakeFiles/gnn_ext.dir/root/workspaces/python/pyg/gnn_general_ext/src/main.cpp.o: /root/workspaces/python/pyg/gnn_general_ext/src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/workspaces/python/pyg/gnn_general_ext/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/gnn_ext.dir/root/workspaces/python/pyg/gnn_general_ext/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gnn_ext.dir/root/workspaces/python/pyg/gnn_general_ext/src/main.cpp.o -c /root/workspaces/python/pyg/gnn_general_ext/src/main.cpp

CMakeFiles/gnn_ext.dir/root/workspaces/python/pyg/gnn_general_ext/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gnn_ext.dir/root/workspaces/python/pyg/gnn_general_ext/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/workspaces/python/pyg/gnn_general_ext/src/main.cpp > CMakeFiles/gnn_ext.dir/root/workspaces/python/pyg/gnn_general_ext/src/main.cpp.i

CMakeFiles/gnn_ext.dir/root/workspaces/python/pyg/gnn_general_ext/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gnn_ext.dir/root/workspaces/python/pyg/gnn_general_ext/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/workspaces/python/pyg/gnn_general_ext/src/main.cpp -o CMakeFiles/gnn_ext.dir/root/workspaces/python/pyg/gnn_general_ext/src/main.cpp.s

CMakeFiles/gnn_ext.dir/root/workspaces/python/pyg/gnn_general_ext/src/main.cpp.o.requires:

.PHONY : CMakeFiles/gnn_ext.dir/root/workspaces/python/pyg/gnn_general_ext/src/main.cpp.o.requires

CMakeFiles/gnn_ext.dir/root/workspaces/python/pyg/gnn_general_ext/src/main.cpp.o.provides: CMakeFiles/gnn_ext.dir/root/workspaces/python/pyg/gnn_general_ext/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/gnn_ext.dir/build.make CMakeFiles/gnn_ext.dir/root/workspaces/python/pyg/gnn_general_ext/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/gnn_ext.dir/root/workspaces/python/pyg/gnn_general_ext/src/main.cpp.o.provides

CMakeFiles/gnn_ext.dir/root/workspaces/python/pyg/gnn_general_ext/src/main.cpp.o.provides.build: CMakeFiles/gnn_ext.dir/root/workspaces/python/pyg/gnn_general_ext/src/main.cpp.o


# Object files for target gnn_ext
gnn_ext_OBJECTS = \
"CMakeFiles/gnn_ext.dir/root/workspaces/python/pyg/gnn_general_ext/src/main.cpp.o"

# External object files for target gnn_ext
gnn_ext_EXTERNAL_OBJECTS = \
"/root/workspaces/python/pyg/gnn_general_ext/build/CMakeFiles/cuda_impl.dir/root/workspaces/python/pyg/gnn_general_ext/src/cuda/sage.cu.o"

CMakeFiles/gnn_ext.dir/cmake_device_link.o: CMakeFiles/gnn_ext.dir/root/workspaces/python/pyg/gnn_general_ext/src/main.cpp.o
CMakeFiles/gnn_ext.dir/cmake_device_link.o: CMakeFiles/cuda_impl.dir/root/workspaces/python/pyg/gnn_general_ext/src/cuda/sage.cu.o
CMakeFiles/gnn_ext.dir/cmake_device_link.o: CMakeFiles/gnn_ext.dir/build.make
CMakeFiles/gnn_ext.dir/cmake_device_link.o: /usr/lib/gcc/x86_64-linux-gnu/7/libgomp.so
CMakeFiles/gnn_ext.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libpthread.so
CMakeFiles/gnn_ext.dir/cmake_device_link.o: CMakeFiles/gnn_ext.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/workspaces/python/pyg/gnn_general_ext/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/gnn_ext.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gnn_ext.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/gnn_ext.dir/build: CMakeFiles/gnn_ext.dir/cmake_device_link.o

.PHONY : CMakeFiles/gnn_ext.dir/build

# Object files for target gnn_ext
gnn_ext_OBJECTS = \
"CMakeFiles/gnn_ext.dir/root/workspaces/python/pyg/gnn_general_ext/src/main.cpp.o"

# External object files for target gnn_ext
gnn_ext_EXTERNAL_OBJECTS = \
"/root/workspaces/python/pyg/gnn_general_ext/build/CMakeFiles/cuda_impl.dir/root/workspaces/python/pyg/gnn_general_ext/src/cuda/sage.cu.o"

gnn_ext.cpython-37m-x86_64-linux-gnu.so: CMakeFiles/gnn_ext.dir/root/workspaces/python/pyg/gnn_general_ext/src/main.cpp.o
gnn_ext.cpython-37m-x86_64-linux-gnu.so: CMakeFiles/cuda_impl.dir/root/workspaces/python/pyg/gnn_general_ext/src/cuda/sage.cu.o
gnn_ext.cpython-37m-x86_64-linux-gnu.so: CMakeFiles/gnn_ext.dir/build.make
gnn_ext.cpython-37m-x86_64-linux-gnu.so: /usr/lib/gcc/x86_64-linux-gnu/7/libgomp.so
gnn_ext.cpython-37m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpthread.so
gnn_ext.cpython-37m-x86_64-linux-gnu.so: CMakeFiles/gnn_ext.dir/cmake_device_link.o
gnn_ext.cpython-37m-x86_64-linux-gnu.so: CMakeFiles/gnn_ext.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/workspaces/python/pyg/gnn_general_ext/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared module gnn_ext.cpython-37m-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gnn_ext.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/strip /root/workspaces/python/pyg/gnn_general_ext/build/gnn_ext.cpython-37m-x86_64-linux-gnu.so

# Rule to build all files generated by this target.
CMakeFiles/gnn_ext.dir/build: gnn_ext.cpython-37m-x86_64-linux-gnu.so

.PHONY : CMakeFiles/gnn_ext.dir/build

CMakeFiles/gnn_ext.dir/requires: CMakeFiles/gnn_ext.dir/root/workspaces/python/pyg/gnn_general_ext/src/main.cpp.o.requires

.PHONY : CMakeFiles/gnn_ext.dir/requires

CMakeFiles/gnn_ext.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/gnn_ext.dir/cmake_clean.cmake
.PHONY : CMakeFiles/gnn_ext.dir/clean

CMakeFiles/gnn_ext.dir/depend:
	cd /root/workspaces/python/pyg/gnn_general_ext/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/workspaces/python/pyg/gnn_general_ext/build /root/workspaces/python/pyg/gnn_general_ext/build /root/workspaces/python/pyg/gnn_general_ext/build /root/workspaces/python/pyg/gnn_general_ext/build /root/workspaces/python/pyg/gnn_general_ext/build/CMakeFiles/gnn_ext.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/gnn_ext.dir/depend
