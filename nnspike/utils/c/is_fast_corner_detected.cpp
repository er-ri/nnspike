#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <algorithm>

namespace py = pybind11;

// コーナー抽出
bool is_fast_corner_detected(py::array image_np, std::tuple<int,int,int,int> roi, std::string course = "right") {
    py::buffer_info buf = image_np.request();
    int height = buf.shape[0];
    int width = buf.shape[1];
    int channels = (buf.ndim == 3) ? buf.shape[2] : 1;
    cv::Mat img;
    if (channels == 3)
        img = cv::Mat(height, width, CV_8UC3, buf.ptr);
    else
        img = cv::Mat(height, width, CV_8UC1, buf.ptr);
    if (img.empty()) return false;
    int x1, y1, x2, y2;
    std::tie(x1, y1, x2, y2) = roi;
    int center_x = 320, center_y = 200, left_x = 50, x_tolerance = 100, min_area = 10000;
    if (course == "left")
        cv::flip(img, img, 1);
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    cv::Mat green_mask;
    cv::inRange(hsv, cv::Scalar(35, 120, 60), cv::Scalar(90, 255, 220), green_mask);
    img.setTo(cv::Scalar(255,255,255), green_mask);
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8,8));
    clahe->apply(gray, gray);
    cv::medianBlur(gray, gray, 7);
    cv::Mat binary;
    cv::threshold(gray, binary, 120, 255, cv::THRESH_BINARY_INV);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    cv::Mat mask_roi = cv::Mat::zeros(binary.size(), binary.type());
    binary(cv::Rect(x1, y1, x2-x1, y2-y1)).copyTo(mask_roi(cv::Rect(x1, y1, x2-x1, y2-y1)));
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_roi, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    int roi_x_min = std::max(center_x - x_tolerance, x1);
    int roi_x_max = std::min(center_x + x_tolerance, x2-1);
    for (const auto& cnt : contours) {
        double area = cv::contourArea(cnt);
        if (area < min_area) continue;
        bool crosses_x_hit = cv::countNonZero(mask_roi.colRange(roi_x_min, roi_x_max)) > 0;
        bool cond_area = area >= min_area;
        cv::Mat y_line = mask_roi.rowRange(center_y, y2).col(left_x);
        int max_run = 0, run = 0;
        for (int i = 0; i < y_line.rows; ++i) {
            if (y_line.at<uchar>(i, 0) == 255) {
                run++;
                if (run > max_run) max_run = run;
            } else {
                run = 0;
            }
        }
        bool crosses_y_hit = max_run >= 10;
        bool all_conditions = cond_area && crosses_x_hit && crosses_y_hit;
        if (all_conditions) return true;
    }
    return false;
}

PYBIND11_MODULE(control_cpp_corner, m) {
    m.def("is_fast_corner_detected", &is_fast_corner_detected, "Fast corner detector", py::arg("image"), py::arg("roi"), py::arg("course") = "right");
}
