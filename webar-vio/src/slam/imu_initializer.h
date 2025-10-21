// slam/imu_initializer.h
#pragma once
#include <vector>
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <sophus/se3.hpp>
#include "slam/imu_types.h"

struct ScaleInitResult {
    double scale = 1.0;
    Eigen::Vector3d gravity = Eigen::Vector3d(0, -9.8, 0);
    bool success = false;
};

class ScaleInitializer {
public:
    // Estimate scale and gravity vector from two poses and IMU samples between them
    static ScaleInitResult EstimateScaleAndGravity(
        const Sophus::SE3d& T0_wc,
        const Sophus::SE3d& T1_wc,
        const std::vector<IMUSample>& imu_span);
};
