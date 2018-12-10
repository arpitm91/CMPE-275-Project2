cmake_minimum_required(VERSION 3.3)

project (../contrast)


set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Modules)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)


# openmp flags
SET(OPENMP_CXX_FLAGS "-O3 -fopenmp")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${OPENMP_CXX_FLAGS}" )

find_package( OpenCV REQUIRED )

# app targets
add_executable(contrast-seq ../contrast/seq.cpp)
target_link_libraries(contrast-seq ${OpenCV_LIBS} )

add_executable(contrast-omp ../contrast/openmp.cpp)
target_link_libraries(contrast-omp ${OpenCV_LIBS} )