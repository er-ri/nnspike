#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <tuple>

namespace py = pybind11;

std::tuple<py::object, py::object, py::object>
get_line_edges_at_y(py::array image_np, std::tuple<int, int, int, int> roi, int target_y, int threshold_value = 80) {
    py::buffer_info buf = image_np.request();
    int height = buf.shape[0];
    int width = buf.shape[1];
    int channels = (buf.ndim == 3) ? buf.shape[2] : 1;
    cv::Mat img;
    if (channels == 3)
        img = cv::Mat(height, width, CV_8UC3, buf.ptr);
    else
        img = cv::Mat(height, width, CV_8UC1, buf.ptr);

    int x1, y1, x2, y2;
    std::tie(x1, y1, x2, y2) = roi;
    if (target_y < y1 || target_y >= y2)
        return std::make_tuple(py::none(), py::none(), py::none());

    cv::Mat gray;
    if (img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img;
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
    cv::Mat binary;
    cv::threshold(gray, binary, threshold_value, 255, cv::THRESH_BINARY_INV);

    cv::Mat roi_img = binary(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    int roi_row = target_y - y1;
    if (roi_row < 0 || roi_row >= roi_img.rows)
        return std::make_tuple(py::none(), py::none(), py::none());

    cv::Mat row = roi_img.row(roi_row);
    std::vector<int> white_pixels;
    for (int x = 0; x < row.cols; ++x) {
        if (row.at<uchar>(0, x) == 255)
            white_pixels.push_back(x);
    }
    if (!white_pixels.empty()) {
        int left_x = x1 + white_pixels.front();
        int right_x = x1 + white_pixels.back();
        int line_width = right_x - left_x + 1;
        // int→py::objectに変換して返す
        return std::make_tuple(py::cast(left_x), py::cast(right_x), py::cast(line_width));
    }
    return std::make_tuple(py::none(), py::none(), py::none());
}

PYBIND11_MODULE(control_cpp, m) {
    m.def("get_line_edges_at_y", &get_line_edges_at_y, "Line edge detector",
        py::arg("image"), py::arg("roi"), py::arg("target_y"), py::arg("threshold_value") = 80);
}
