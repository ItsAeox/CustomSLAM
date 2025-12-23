#pragma once
#include <opencv2/core.hpp>

struct VioCalib {
  cv::Matx33d K;        // camera intrinsics
  cv::Matx33d R_ci;     // camera-from-imu rotation
  cv::Vec3d   t_ci;     // camera-from-imu translation (meters)
  cv::Vec3d   g_w;      // gravity in world (m/s^2), e.g. (0, 0, -9.81)
  double gyro_noise = 1e-3;
  double acc_noise  = 1e-2;
  double gyro_rw    = 1e-5;
  double acc_rw     = 1e-4;
};

struct VioState {
  // World-from-IMU (NOT camera): R_wi, p_wi, v_wi
  cv::Matx33d R_wi = cv::Matx33d::eye();
  cv::Vec3d   p_wi = cv::Vec3d(0,0,0);
  cv::Vec3d   v_wi = cv::Vec3d(0,0,0);
  cv::Vec3d   b_g  = cv::Vec3d(0,0,0);
  cv::Vec3d   b_a  = cv::Vec3d(0,0,0);
  double ts = 0.0;
};
