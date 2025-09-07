#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <string>

namespace py = pybind11;

// 左黒ライン抽出
bool is_left_black_line_detected(py::array image_np, std::string course) {
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
    int x1 = 0, y1 = 0, x2 = 160, y2 = 480; // ROI_LINE_LEFT例
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
    cv::dilate(binary, binary, kernel);
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    cv::Mat mask_roi = cv::Mat::zeros(binary.size(), binary.type());
    binary(cv::Rect(x1, y1, x2-x1, y2-y1)).copyTo(mask_roi(cv::Rect(x1, y1, x2-x1, y2-y1)));
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_roi, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    int min_width = 60, min_height = 150, min_area = 8000;
    double min_aspect = 2.0;
    for (const auto& cnt : contours) {
        cv::Rect rect = cv::boundingRect(cnt);
        double aspect = (double)rect.height / (rect.width + 1e-5);
        double area = cv::contourArea(cnt);
        if (rect.width >= min_width && rect.height >= min_height && aspect >= min_aspect && area >= min_area)
            return true;
    }
    return false;
}

PYBIND11_MODULE(control_cpp_left_black, m) {
    m.def("is_left_black_line_detected", &is_left_black_line_detected, "Left black line detector", py::arg("image"), py::arg("course"));
}
