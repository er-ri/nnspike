#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <tuple>

namespace py = pybind11;

// 仮想ライン中心抽出
int get_virtual_line_target_x(py::array image_np) {
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
        return 320;
    // ROI: (x1, y1, x2, y2) = (160, 200, 480, 400)（例: ROI_VIRTUAL）
    int x1 = 160, y1 = 200, x2 = 480, y2 = 400;
    cv::Mat gray;
    if (img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img;
    // CLAHE
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8,8));
    clahe->apply(gray, gray);
    cv::medianBlur(gray, gray, 7);
    cv::Mat binary;
    cv::threshold(gray, binary, 120, 255, cv::THRESH_BINARY_INV);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::dilate(binary, binary, kernel);
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    cv::Mat roi_img = binary(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(roi_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    int min_area = 500;
    double max_aspect = 5.0;
    struct Candidate {
        int center_x, center_y, left_edge_x, right_edge_x, area;
    };
    std::vector<Candidate> candidates;
    for (const auto& cnt : contours) {
        cv::Rect rect = cv::boundingRect(cnt);
        int area = rect.width * rect.height;
        double aspect = (rect.height > 0) ? (double)rect.width / rect.height : 0;
        if (area < min_area) continue;
        if (aspect > max_aspect) continue;
        if (area >= 12000) continue;
        int cx = rect.x + rect.width / 2;
        int cy = rect.y + rect.height / 2;
        candidates.push_back({cx, cy, rect.x, rect.x + rect.width, area});
    }
    // 下から順に最初の物体を選択
    Candidate* selected = nullptr;
    int max_cy = -1;
    for (auto& c : candidates) {
        if (c.center_y > max_cy) {
            max_cy = c.center_y;
            selected = &c;
        }
    }
    int target_x = 320;
    if (selected) {
        int center_x = x1 + selected->center_x;
        int left_edge_x = x1 + selected->left_edge_x;
        int right_edge_x = x1 + selected->right_edge_x;
        if (center_x < 320)
            target_x = right_edge_x + 170;
        else
            target_x = left_edge_x - 170;
    }
    return target_x;
}

PYBIND11_MODULE(control_cpp_get_virtual_line_target_x, m) {
    m.def("get_virtual_line_target_x", &get_virtual_line_target_x, "Virtual line target x extractor", py::arg("image"));
}
