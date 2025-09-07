cmake_minimum_required(VERSION 3.12)
project(control_cpp)

set(CMAKE_CXX_STANDARD 14)

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)

pybind11_add_module(control_cpp get_line_edges_at_y.cpp)
target_link_libraries(control_cpp PRIVATE ${OpenCV_LIBS})

pybind11_add_module(control_cpp_bottle find_bottle_center.cpp)
target_link_libraries(control_cpp_bottle PRIVATE ${OpenCV_LIBS})
