#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>

namespace py = pybind11;

// 画像中央付近で赤ターゲット抽出
bool is_x320_on_red_target(py::array image_np, int x_tolerance = 60) {
    py::buffer_info buf = image_np.request();
    int height = buf.shape[0];
    int width = buf.shape[1];
    int channels = (buf.ndim == 3) ? buf.shape[2] : 1;
    cv::Mat img;
    if (channels == 3)
        img = cv::Mat(height, width, CV_8UC3, buf.ptr);
    else
        img = cv::Mat(height, width, CV_8UC1, buf.ptr);
    if (img.empty())
        return false;
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask1, mask2, mask;
    cv::inRange(hsv, cv::Scalar(0, 90, 60), cv::Scalar(15, 255, 210), mask1);
    cv::inRange(hsv, cv::Scalar(175, 90, 60), cv::Scalar(180, 255, 210), mask2);
    cv::bitwise_or(mask1, mask2, mask);
    cv::medianBlur(mask, mask, 5);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Point2f best_center;
    double max_area = 0;
    for (const auto& cnt : contours) {
        if (cnt.size() >= 5) {
            double area = cv::contourArea(cnt);
            if (area < 300) continue;
            cv::RotatedRect ellipse;
            try { ellipse = cv::fitEllipse(cnt); } catch (...) { continue; }
            cv::Rect rect = cv::boundingRect(cnt);
            double rect_area = rect.width * rect.height;
            double rect_ratio = (rect_area > 0) ? area / rect_area : 0;
            if (rect_ratio < 0.4) continue;
            if (area > max_area) {
                max_area = area;
                best_center = ellipse.center;
            }
        }
    }
    if (max_area == 0) return false;
    int cx = (int)best_center.x;
    return std::abs(cx - 320) <= x_tolerance;
}

PYBIND11_MODULE(control_cpp_x320_red, m) {
    m.def("is_x320_on_red_target", &is_x320_on_red_target, "Is x=320 on red target", py::arg("image"), py::arg("x_tolerance") = 60);
}
