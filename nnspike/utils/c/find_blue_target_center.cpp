#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <tuple>

namespace py = pybind11;

// 青い的（楕円）の中心座標・面積・青ピクセル数を返す
std::tuple<py::object, py::object, int>
find_blue_target_center(py::array image_np) {
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
        return std::make_tuple(py::none(), py::none(), 0);
    // 青色抽出（target）
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask;
    cv::inRange(hsv, cv::Scalar(100, 80, 80), cv::Scalar(140, 255, 255), mask);
    // メディアンブラー＋クロージング（5x5楕円）
    cv::medianBlur(mask, mask, 7);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    // 輪郭検出
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    double max_area = 0;
    cv::RotatedRect best_ellipse;
    bool found = false;
    cv::Point2f best_center;
    for (const auto& cnt : contours) {
        if (cnt.size() >= 5) {
            double area = cv::contourArea(cnt);
            if (area < 300) continue;
            cv::RotatedRect ellipse;
            try {
                ellipse = cv::fitEllipse(cnt);
            } catch (...) { continue; }
            cv::Rect rect = cv::boundingRect(cnt);
            double rect_area = rect.width * rect.height;
            double rect_ratio = (rect_area > 0) ? area / rect_area : 0;
            if (rect_ratio < 0.4) continue;
            if (area > max_area) {
                max_area = area;
                best_ellipse = ellipse;
                best_center = ellipse.center;
                found = true;
            }
        }
    }
    int blue_pixel_count = cv::countNonZero(mask);
    if (found) {
        return std::make_tuple(py::make_tuple((int)best_center.x, (int)best_center.y), py::cast(max_area), blue_pixel_count);
    }
    return std::make_tuple(py::none(), py::none(), 0);
}

PYBIND11_MODULE(control_cpp_blue_target, m) {
    m.def("find_blue_target_center", &find_blue_target_center, "Blue target center detector", py::arg("image"));
}
