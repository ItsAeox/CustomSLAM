#pragma once
#include "vio_types.h"
#include <vector>

struct ImuMeas;

bool vioInitFromAccelMean(const std::vector<ImuMeas>& meas,
                          double t0, double t1,
                          const cv::Vec3d& g_w,
                          VioState& out);
