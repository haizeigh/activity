# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

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
CMAKE_COMMAND = /home/westwell/Downloads/CLion-2021.2.3/clion-2021.2.3/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/westwell/Downloads/CLion-2021.2.3/clion-2021.2.3/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/westwell/Documents/project/activity

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/westwell/Documents/project/activity/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/activity1.dir/depend.make
# Include the progress variables for this target.
include CMakeFiles/activity1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/activity1.dir/flags.make

CMakeFiles/activity1.dir/demos/test_train.cpp.o: CMakeFiles/activity1.dir/flags.make
CMakeFiles/activity1.dir/demos/test_train.cpp.o: ../demos/test_train.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/westwell/Documents/project/activity/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/activity1.dir/demos/test_train.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/activity1.dir/demos/test_train.cpp.o -c /home/westwell/Documents/project/activity/demos/test_train.cpp

CMakeFiles/activity1.dir/demos/test_train.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/activity1.dir/demos/test_train.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/westwell/Documents/project/activity/demos/test_train.cpp > CMakeFiles/activity1.dir/demos/test_train.cpp.i

CMakeFiles/activity1.dir/demos/test_train.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/activity1.dir/demos/test_train.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/westwell/Documents/project/activity/demos/test_train.cpp -o CMakeFiles/activity1.dir/demos/test_train.cpp.s

CMakeFiles/activity1.dir/src/DataReader.cpp.o: CMakeFiles/activity1.dir/flags.make
CMakeFiles/activity1.dir/src/DataReader.cpp.o: ../src/DataReader.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/westwell/Documents/project/activity/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/activity1.dir/src/DataReader.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/activity1.dir/src/DataReader.cpp.o -c /home/westwell/Documents/project/activity/src/DataReader.cpp

CMakeFiles/activity1.dir/src/DataReader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/activity1.dir/src/DataReader.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/westwell/Documents/project/activity/src/DataReader.cpp > CMakeFiles/activity1.dir/src/DataReader.cpp.i

CMakeFiles/activity1.dir/src/DataReader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/activity1.dir/src/DataReader.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/westwell/Documents/project/activity/src/DataReader.cpp -o CMakeFiles/activity1.dir/src/DataReader.cpp.s

CMakeFiles/activity1.dir/src/DataProc.cpp.o: CMakeFiles/activity1.dir/flags.make
CMakeFiles/activity1.dir/src/DataProc.cpp.o: ../src/DataProc.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/westwell/Documents/project/activity/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/activity1.dir/src/DataProc.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/activity1.dir/src/DataProc.cpp.o -c /home/westwell/Documents/project/activity/src/DataProc.cpp

CMakeFiles/activity1.dir/src/DataProc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/activity1.dir/src/DataProc.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/westwell/Documents/project/activity/src/DataProc.cpp > CMakeFiles/activity1.dir/src/DataProc.cpp.i

CMakeFiles/activity1.dir/src/DataProc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/activity1.dir/src/DataProc.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/westwell/Documents/project/activity/src/DataProc.cpp -o CMakeFiles/activity1.dir/src/DataProc.cpp.s

CMakeFiles/activity1.dir/src/FeatureExtractor.cpp.o: CMakeFiles/activity1.dir/flags.make
CMakeFiles/activity1.dir/src/FeatureExtractor.cpp.o: ../src/FeatureExtractor.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/westwell/Documents/project/activity/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/activity1.dir/src/FeatureExtractor.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/activity1.dir/src/FeatureExtractor.cpp.o -c /home/westwell/Documents/project/activity/src/FeatureExtractor.cpp

CMakeFiles/activity1.dir/src/FeatureExtractor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/activity1.dir/src/FeatureExtractor.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/westwell/Documents/project/activity/src/FeatureExtractor.cpp > CMakeFiles/activity1.dir/src/FeatureExtractor.cpp.i

CMakeFiles/activity1.dir/src/FeatureExtractor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/activity1.dir/src/FeatureExtractor.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/westwell/Documents/project/activity/src/FeatureExtractor.cpp -o CMakeFiles/activity1.dir/src/FeatureExtractor.cpp.s

CMakeFiles/activity1.dir/src/FeatureAnalyse.cpp.o: CMakeFiles/activity1.dir/flags.make
CMakeFiles/activity1.dir/src/FeatureAnalyse.cpp.o: ../src/FeatureAnalyse.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/westwell/Documents/project/activity/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/activity1.dir/src/FeatureAnalyse.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/activity1.dir/src/FeatureAnalyse.cpp.o -c /home/westwell/Documents/project/activity/src/FeatureAnalyse.cpp

CMakeFiles/activity1.dir/src/FeatureAnalyse.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/activity1.dir/src/FeatureAnalyse.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/westwell/Documents/project/activity/src/FeatureAnalyse.cpp > CMakeFiles/activity1.dir/src/FeatureAnalyse.cpp.i

CMakeFiles/activity1.dir/src/FeatureAnalyse.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/activity1.dir/src/FeatureAnalyse.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/westwell/Documents/project/activity/src/FeatureAnalyse.cpp -o CMakeFiles/activity1.dir/src/FeatureAnalyse.cpp.s

CMakeFiles/activity1.dir/src/Classfier.cpp.o: CMakeFiles/activity1.dir/flags.make
CMakeFiles/activity1.dir/src/Classfier.cpp.o: ../src/Classfier.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/westwell/Documents/project/activity/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/activity1.dir/src/Classfier.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/activity1.dir/src/Classfier.cpp.o -c /home/westwell/Documents/project/activity/src/Classfier.cpp

CMakeFiles/activity1.dir/src/Classfier.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/activity1.dir/src/Classfier.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/westwell/Documents/project/activity/src/Classfier.cpp > CMakeFiles/activity1.dir/src/Classfier.cpp.i

CMakeFiles/activity1.dir/src/Classfier.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/activity1.dir/src/Classfier.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/westwell/Documents/project/activity/src/Classfier.cpp -o CMakeFiles/activity1.dir/src/Classfier.cpp.s

# Object files for target activity1
activity1_OBJECTS = \
"CMakeFiles/activity1.dir/demos/test_train.cpp.o" \
"CMakeFiles/activity1.dir/src/DataReader.cpp.o" \
"CMakeFiles/activity1.dir/src/DataProc.cpp.o" \
"CMakeFiles/activity1.dir/src/FeatureExtractor.cpp.o" \
"CMakeFiles/activity1.dir/src/FeatureAnalyse.cpp.o" \
"CMakeFiles/activity1.dir/src/Classfier.cpp.o"

# External object files for target activity1
activity1_EXTERNAL_OBJECTS =

activity1: CMakeFiles/activity1.dir/demos/test_train.cpp.o
activity1: CMakeFiles/activity1.dir/src/DataReader.cpp.o
activity1: CMakeFiles/activity1.dir/src/DataProc.cpp.o
activity1: CMakeFiles/activity1.dir/src/FeatureExtractor.cpp.o
activity1: CMakeFiles/activity1.dir/src/FeatureAnalyse.cpp.o
activity1: CMakeFiles/activity1.dir/src/Classfier.cpp.o
activity1: CMakeFiles/activity1.dir/build.make
activity1: CMakeFiles/activity1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/westwell/Documents/project/activity/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable activity1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/activity1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/activity1.dir/build: activity1
.PHONY : CMakeFiles/activity1.dir/build

CMakeFiles/activity1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/activity1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/activity1.dir/clean

CMakeFiles/activity1.dir/depend:
	cd /home/westwell/Documents/project/activity/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/westwell/Documents/project/activity /home/westwell/Documents/project/activity /home/westwell/Documents/project/activity/cmake-build-debug /home/westwell/Documents/project/activity/cmake-build-debug /home/westwell/Documents/project/activity/cmake-build-debug/CMakeFiles/activity1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/activity1.dir/depend

