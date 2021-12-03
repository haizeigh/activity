cmake_minimum_required(VERSION 3.16)
project(activity1)


set( mlpack_dir  /usr/include/mlpack)
set( fann_dir  /usr/include)
set( arm_dir  /usr/include/armadillo_bits)

#fann
set( mlpackc_dir  /usr/lib/x86_64-linux-gnu /usr/lib )
set( armc_dir  /usr/lib )

include_directories(${mlpack_dir} ${fann_dir} ${arm_dir}  ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include/ActivityRecognition)
link_directories(${mlpackc_dir}  ${armc_dir} ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/demos)
link_libraries(mlpack fann armadillo)
#find_package(MLPACK REQUIRED)
#link_directories(${MLPACK_LIBRARY_DIRS})


set(CMAKE_CXX_STANDARD 14)


file(GLOB SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)
#list()

#add_executable(activity1 main.cpp)
#add_executable(activity1 demos/test_readDir.cpp ${SRC_FILES}  )
#add_executable(activity1  demos/test_reader.cpp
add_executable(activity1  demos/test_train.cpp
        ${CMAKE_SOURCE_DIR}/src/DataReader.cpp
        ${CMAKE_SOURCE_DIR}/src/DataProc.cpp
        ${CMAKE_SOURCE_DIR}/src/FeatureExtractor.cpp
        ${CMAKE_SOURCE_DIR}/src/FeatureAnalyse.cpp
        )
target_link_libraries(activity1 -lmlpack -larmadillo -larmadillo)
