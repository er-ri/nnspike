#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <string>

namespace py = pybind11;

std::tuple<py::object, py::object, int>
find_bottle_center(py::array image_np, std::string color, std::tuple<int, int, int, int> roi) {
    // Python→cv::Mat変換
    py::buffer_info buf = image_np.request();
    int height = buf.shape[0];
    int width = buf.shape[1];
    int channels = (buf.ndim == 3) ? buf.shape[2] : 1;
    cv::Mat img;
    if (channels == 3)
        img = cv::Mat(height, width, CV_8UC3, buf.ptr);
    else
        img = cv::Mat(height, width, CV_8UC1, buf.ptr);

    int min_area = 490;
    if (img.empty() || (color != "yellow" && color != "blue" && color != "red"))
        return std::make_tuple(py::none(), py::none(), 0);

    // --- HSV色範囲定義 ---
    cv::Scalar lower, upper;
    cv::Mat mask;
    if (color == "yellow") {
        lower = cv::Scalar(15, 100, 100);
        upper = cv::Scalar(35, 255, 255);
        cv::Mat hsv;
        cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, lower, upper, mask);
    } else if (color == "blue") {
        lower = cv::Scalar(90, 60, 40);
        upper = cv::Scalar(140, 255, 255);
        cv::Mat hsv;
        cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, lower, upper, mask);
    } else if (color == "red") {
        cv::Mat hsv;
        cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
        cv::Mat mask1, mask2;
        cv::inRange(hsv, cv::Scalar(0, 90, 60), cv::Scalar(12, 255, 255), mask1);
        cv::inRange(hsv, cv::Scalar(170, 90, 60), cv::Scalar(180, 255, 255), mask2);
        cv::bitwise_or(mask1, mask2, mask);
    }
    // --- メディアンブラー＋ノイズ除去（クロージング7x7） ---
    cv::medianBlur(mask, mask, 7);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // --- ROI適用 ---
    int x1, y1, x2, y2;
    std::tie(x1, y1, x2, y2) = roi;
    cv::Mat mask_roi = cv::Mat::zeros(mask.size(), mask.type());
    mask(cv::Rect(x1, y1, x2 - x1, y2 - y1)).copyTo(mask_roi(cv::Rect(x1, y1, x2 - x1, y2 - y1)));

    // --- 輪郭検出 ---
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_roi, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty())
        return std::make_tuple(py::none(), py::none(), 0);

    double max_area = 0;
    cv::Rect best_rect;
    int color_pixel_count = 0;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < min_area)
            continue;
        cv::Rect rect = cv::boundingRect(contour);
        double rect_area = rect.width * rect.height;
        if (rect_area == 0)
            continue;
        double ratio = area / rect_area;
        if (ratio <= 0.5)
            continue;
        cv::Mat rect_mask = mask(rect);
        int rect_color_pixel_count = cv::countNonZero(rect_mask);
        if (area <= rect_color_pixel_count / 2)
            continue;
        if (area > max_area) {
            max_area = area;
            best_rect = rect;
            color_pixel_count = rect_color_pixel_count;
        }
    }
    if (max_area == 0)
        return std::make_tuple(py::none(), py::none(), 0);
    double cx = best_rect.x + best_rect.width / 2.0;
    double cy = best_rect.y + best_rect.height / 2.0;
    return std::make_tuple(py::make_tuple(cx, cy), py::cast(max_area), color_pixel_count);
}

PYBIND11_MODULE(control_cpp_bottle, m) {
    m.def("find_bottle_center", &find_bottle_center, "Bottle center detector",
        py::arg("image"), py::arg("color"), py::arg("roi"));
}
