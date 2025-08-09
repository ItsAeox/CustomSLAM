#pragma once
#include <opencv2/core.hpp>
#include <vector>

struct Frame {
    double timestamp{0.0};
    cv::Mat gray;                       // CV_8UC1
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;                       // CV_8U for ORB
};
