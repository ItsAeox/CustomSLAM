#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include "imu_preint.h"
#include "vio_types.h"

struct InitKf {
  double ts = 0.0;
  cv::Matx33d Rwc_vo = cv::Matx33d::eye(); // world-from-camera in VO world
  cv::Vec3d   twc_vo = cv::Vec3d(0,0,0);   // VO units
};

struct VioInitResult {
  bool ok = false;
  double scale = 1.0;             // meters per VO unit
  cv::Vec3d g_w = cv::Vec3d(0,-9.81,0);
  cv::Vec3d b_g = cv::Vec3d(0,0,0);
  cv::Vec3d b_a = cv::Vec3d(0,0,0);
  std::vector<cv::Vec3d> v_w;     // velocity per KF (meters/sec)
};

class VioInitializer {
public:
  void reset() { kfs_.clear(); }
  void push(const InitKf& kf) { kfs_.push_back(kf); }
  int  size() const { return (int)kfs_.size(); }

  bool ready(double minSpanSec=2.0, int minKfs=12) const;

  VioInitResult solve(const std::vector<ImuMeas>& imu,
                      const VioCalib& calib,
                      int maxOuterIters=5);

private:
  std::vector<InitKf> kfs_;

  bool solveScaleGravityVelocity(const std::vector<ImuMeas>& imu,
                                 const VioCalib& calib,
                                 const cv::Vec3d& bg,
                                 double& s_out,
                                 cv::Vec3d& g_out,
                                 std::vector<cv::Vec3d>& v_out);
};
