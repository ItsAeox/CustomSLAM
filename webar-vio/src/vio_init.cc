#include "vio_init.h"
#include "imu_preint.h"
#include <cmath>

static inline cv::Vec3d normed(const cv::Vec3d& v){
  double n = cv::norm(v);
  if (n < 1e-9) return cv::Vec3d(0,0,0);
  return v*(1.0/n);
}

static inline cv::Matx33d rotFromTwoVectors(const cv::Vec3d& a, const cv::Vec3d& b){
  // returns R such that R*a ~= b
  cv::Vec3d v = a.cross(b);
  double c = a.dot(b);
  double s = cv::norm(v);
  if (s < 1e-9) return cv::Matx33d::eye();
  cv::Matx33d vx(0, -v[2], v[1],
                 v[2], 0, -v[0],
                 -v[1], v[0], 0);
  cv::Matx33d I = cv::Matx33d::eye();
  // Rodrigues formula
  return I + vx + (vx*vx) * ((1.0 - c) / (s*s));
}

bool vioInitFromAccelMean(const std::vector<ImuMeas>& meas,
                          double t0, double t1,
                          const cv::Vec3d& g_w,
                          VioState& out)
{
  if (meas.empty() || t1 <= t0) return false;

  cv::Vec3d aSum(0,0,0);
  int cnt = 0;
  for (const auto& m : meas) {
    if (m.t < t0 || m.t > t1) continue;
    aSum += m.acc;
    cnt++;
  }
  if (cnt < 10) return false;

  cv::Vec3d aMean = aSum * (1.0 / cnt);
  // accel measures (gravity + linear accel). Over a short stationary-ish window, mean ~= gravity direction.
  // We align IMU so measured accel direction aligns with -g_w direction depending on your convention.
  cv::Vec3d gdir = normed(g_w);
  cv::Vec3d am   = normed(aMean);

  // Many rigs: accel points "down" (same direction as gravity). Your g_w is likely (0,-9.81,0).
  // So align am -> gdir.
  cv::Matx33d R_wi = rotFromTwoVectors(am, gdir);

  out.R_wi = R_wi;
  out.p_wi = cv::Vec3d(0,0,0);
  out.v_wi = cv::Vec3d(0,0,0);
  out.b_g  = cv::Vec3d(0,0,0);
  out.b_a  = cv::Vec3d(0,0,0);
  out.ts   = t1;
  return true;
}
