// slam/imu_types.h
#pragma once
#include <Eigen/Core>

struct IMUSample {
    double t;
    Eigen::Vector3d acc;
    Eigen::Vector3d gyro;
};
