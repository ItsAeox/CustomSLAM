#pragma once
#include "vio_types.h"
#include "imu_preint.h"
#include <opencv2/core.hpp>
#include <vector>

struct FeatureObs {
    int mp_id;           // map point id (optional for debugging)
    cv::Vec3d Xw;        // world point (meters, in your current world)
    double u, v;         // pixel (full-res)
  };

class VioBackend {
public:
  void setCalib(const VioCalib& c) { calib_ = c; }

  // Add a keyframe node with observations and IMU factor from prev KF
  void addKeyframe(const VioState& init,
                   const std::vector<FeatureObs>& obs,
                   const ImuPreint& pim);

  // Optimize sliding window and output latest state (and optionally all)
  bool optimize(int max_iters = 10);

  const VioState& latest() const { return states_.back(); }

private:
  VioCalib calib_;
  int window_ = 10;
  std::vector<VioState> states_;
  std::vector<std::vector<FeatureObs>> obs_;
  std::vector<ImuPreint> pim_; // between states_[i-1] -> states_[i]
};
