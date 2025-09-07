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

pybind11_add_module(control_cpp_virtual get_virtual_line_target_x.cpp)
target_link_libraries(control_cpp_virtual PRIVATE ${OpenCV_LIBS})

pybind11_add_module(control_cpp_blue_target find_blue_target_center.cpp)
target_link_libraries(control_cpp_blue_target PRIVATE ${OpenCV_LIBS})

pybind11_add_module(control_cpp_blue_line get_is_blue_line_at_y.cpp)
target_link_libraries(control_cpp_blue_line PRIVATE ${OpenCV_LIBS})

pybind11_add_module(control_cpp_x320_blue is_x320_on_blue_target.cpp)
target_link_libraries(control_cpp_x320_blue PRIVATE ${OpenCV_LIBS})

pybind11_add_module(control_cpp_x320_red is_x320_on_red_target.cpp)
target_link_libraries(control_cpp_x320_red PRIVATE ${OpenCV_LIBS})

pybind11_add_module(control_cpp_red_center get_red_target_center_x.cpp)
target_link_libraries(control_cpp_red_center PRIVATE ${OpenCV_LIBS})

pybind11_add_module(control_cpp_left_black is_left_black_line_detected.cpp)
target_link_libraries(control_cpp_left_black PRIVATE ${OpenCV_LIBS})

pybind11_add_module(control_cpp_lower_horizontal is_lower_horizontal_line_detected.cpp)
target_link_libraries(control_cpp_lower_horizontal PRIVATE ${OpenCV_LIBS})

pybind11_add_module(control_cpp_vertical_black is_vertical_black_line_detected.cpp)
target_link_libraries(control_cpp_vertical_black PRIVATE ${OpenCV_LIBS})

pybind11_add_module(control_cpp_upper_horizontal is_upper_horizontal_line_detected.cpp)
target_link_libraries(control_cpp_upper_horizontal PRIVATE ${OpenCV_LIBS})

pybind11_add_module(control_cpp_blue_pixel get_blue_line_pixel.cpp)
target_link_libraries(control_cpp_blue_pixel PRIVATE ${OpenCV_LIBS})

pybind11_add_module(control_cpp_fast_corner is_fast_corner_detected.cpp)
target_link_libraries(control_cpp_fast_corner PRIVATE ${OpenCV_LIBS})
