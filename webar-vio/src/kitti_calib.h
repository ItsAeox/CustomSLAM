#pragma once
#include <opencv2/core.hpp>
#include <string>

struct KittiCalibOut {
  cv::Matx33d K = cv::Matx33d::eye();
  cv::Matx33d R_ci = cv::Matx33d::eye();  // camera-from-imu
  cv::Vec3d   t_ci = cv::Vec3d(0,0,0);
};

bool parseKittiCalibFromTexts_Image02(
  const std::string& calib_cam_to_cam_txt,
  const std::string& calib_velo_to_cam_txt,
  const std::string& calib_imu_to_velo_txt,
  KittiCalibOut& out
);
