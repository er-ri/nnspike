#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>

namespace py = pybind11;

// 青ライン有無判定
bool get_is_blue_line_at_y(py::array image_np, int target_y = 470, int min_run = 30) {
    py::buffer_info buf = image_np.request();
    int height = buf.shape[0];
    int width = buf.shape[1];
    int channels = (buf.ndim == 3) ? buf.shape[2] : 1;
    cv::Mat img;
    if (channels == 3)
        img = cv::Mat(height, width, CV_8UC3, buf.ptr);
    else
        img = cv::Mat(height, width, CV_8UC1, buf.ptr);
    if (img.empty() || target_y < 0 || target_y >= img.rows)
        return false;
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask;
    cv::inRange(hsv, cv::Scalar(95, 100, 50), cv::Scalar(145, 255, 255), mask);
    cv::Mat line_mask = mask.row(target_y);
    int max_run = 0, current_run = 0;
    for (int x = 0; x < line_mask.cols; ++x) {
        if (line_mask.at<uchar>(0, x)) {
            current_run++;
            if (current_run > max_run) max_run = current_run;
        } else {
            current_run = 0;
        }
    }
    return max_run >= min_run;
}

PYBIND11_MODULE(control_cpp_blue_line, m) {
    m.def("get_is_blue_line_at_y", &get_is_blue_line_at_y, "Blue line presence detector", py::arg("image"), py::arg("target_y") = 470, py::arg("min_run") = 30);
}
