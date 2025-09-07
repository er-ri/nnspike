#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>

namespace py = pybind11;

// 青オブジェクト面積抽出
int get_blue_line_pixel(py::array image_np) {
    py::buffer_info buf = image_np.request();
    int height = buf.shape[0];
    int width = buf.shape[1];
    int channels = (buf.ndim == 3) ? buf.shape[2] : 1;
    cv::Mat img;
    if (channels == 3)
        img = cv::Mat(height, width, CV_8UC3, buf.ptr);
    else
        img = cv::Mat(height, width, CV_8UC1, buf.ptr);
    if (img.empty()) return 0;
    int x1 = 160, y1 = 200, x2 = 480, y2 = 400; // ROI_LOOP例
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask;
    cv::inRange(hsv, cv::Scalar(95, 100, 50), cv::Scalar(145, 255, 255), mask);
    cv::medianBlur(mask, mask, 7);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::dilate(mask, mask, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::Mat roi_mask = mask(cv::Rect(x1, y1, x2-x1, y2-y1));
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(roi_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    int max_area = 0;
    for (const auto& cnt : contours) {
        double area = cv::contourArea(cnt);
        if (area > 300 && area > max_area)
            max_area = (int)area;
    }
    return max_area;
}

PYBIND11_MODULE(control_cpp_blue_pixel, m) {
    m.def("get_blue_line_pixel", &get_blue_line_pixel, "Blue object pixel area extractor", py::arg("image"));
}
