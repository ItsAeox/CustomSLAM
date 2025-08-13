#pragma once
#include <Eigen/Core>
#include <opencv2/core.hpp>

struct MapPoint {
  Eigen::Vector3d Xw;   // world coords
  cv::Mat desc;         // ORB descriptor (1x32 CV_8U)
  int obs = 1;          // how many times re‑observed (future use)
};
