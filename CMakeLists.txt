cmake_minimum_required(VERSION 3.3)

project (image_processing)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Modules)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

# Boost flags
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(Boost COMPONENTS chrono)

# openmp flags
set(OPENMP_CXX_FLAGS "-O3 -fopenmp")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${OPENMP_CXX_FLAGS}")

find_package( OpenCV REQUIRED )

# app targets
add_executable(contrast-seq ../src/contrast/seq.cpp)
target_link_libraries(contrast-seq ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable(contrast-omp ../src/contrast/openmp.cpp)
target_link_libraries(contrast-omp ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable(sharpen-seq ../src/sharpening/seq.cpp)
target_link_libraries(sharpen-seq ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable(sharpen-omp ../src/sharpening/openmp.cpp)
target_link_libraries(sharpen-omp ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable(blur-seq ../src/blur/seq.cpp)
target_link_libraries(blur-seq ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable(blur-omp ../src/blur/openmp.cpp)
target_link_libraries(blur-omp ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable(openmp_performance_measure ../src/blur/openmp_performance_measure.cpp)
target_link_libraries(openmp_performance_measure ${OpenCV_LIBS} ${Boost_LIBRARIES})

# add_executable(openmp_performance_measure-contrast ../src/contrast/openmp_performance_measure.cpp)
# target_link_libraries(openmp_performance_measure-contrast ${OpenCV_LIBS} ${Boost_LIBRARIES})
