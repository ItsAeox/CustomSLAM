#pragma once
#include <opencv2/core.hpp>
#include <vector>

struct ImuMeas {
  double t;
  cv::Vec3d acc;   // m/s^2
  cv::Vec3d gyro;  // rad/s
};

struct ImuPreint {
  // Preintegrated deltas from i -> j in i-frame
  cv::Matx33d dR = cv::Matx33d::eye();
  cv::Vec3d   dv = cv::Vec3d(0,0,0);
  cv::Vec3d   dp = cv::Vec3d(0,0,0);
  double      dt = 0.0;

  // Jacobians wrt biases (needed for tight coupling)
  cv::Matx33d J_dR_bg = cv::Matx33d::zeros();
  cv::Matx33d J_dv_bg = cv::Matx33d::zeros();
  cv::Matx33d J_dv_ba = cv::Matx33d::zeros();
  cv::Matx33d J_dp_bg = cv::Matx33d::zeros();
  cv::Matx33d J_dp_ba = cv::Matx33d::zeros();

  // Covariance (15x15) would go here (recommended). Start with diagonal if needed.
};

cv::Matx33d so3Exp(const cv::Vec3d& w);
cv::Matx33d skew(const cv::Vec3d& v);

ImuPreint preintegrateImu(const std::vector<ImuMeas>& meas,
                          double t0, double t1,
                          const cv::Vec3d& bg, const cv::Vec3d& ba);
