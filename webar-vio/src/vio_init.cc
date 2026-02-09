#include "vio_init.h"
#include "imu_preint.h"
#include <cmath>

static inline cv::Vec3d normed(const cv::Vec3d& v){
  double n = cv::norm(v);
  if (n < 1e-9) return cv::Vec3d(0,0,0);
  return v*(1.0/n);
}

static inline cv::Matx33d rotFromTwoVectors(const cv::Vec3d& a_in, const cv::Vec3d& b_in){
  // returns R such that R*a ~= b
  cv::Vec3d a = normed(a_in);
  cv::Vec3d b = normed(b_in);
  if (cv::norm(a) < 1e-9 || cv::norm(b) < 1e-9) return cv::Matx33d::eye();

  double c = std::clamp(a.dot(b), -1.0, 1.0);

  // If vectors are nearly the same
  if (c > 1.0 - 1e-9) return cv::Matx33d::eye();

  // If vectors are nearly opposite, choose an arbitrary orthogonal axis
  if (c < -1.0 + 1e-9) {
    cv::Vec3d axis = std::abs(a[0]) < 0.9 ? cv::Vec3d(1,0,0) : cv::Vec3d(0,1,0);
    axis = normed(a.cross(axis));
    // 180-deg rotation: R = I + 2*skew(axis)^2
    cv::Matx33d K(0, -axis[2], axis[1],
                  axis[2], 0, -axis[0],
                  -axis[1], axis[0], 0);
    cv::Matx33d I = cv::Matx33d::eye();
    return I + 2.0 * (K * K);
  }

  cv::Vec3d v = a.cross(b);
  double s = cv::norm(v);

  cv::Matx33d vx(0, -v[2], v[1],
                 v[2], 0, -v[0],
                 -v[1], v[0], 0);

  cv::Matx33d I = cv::Matx33d::eye();
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
  cv::Vec3d am = normed(aMean);

  // For specific-force accelerometers, when stationary: am ≈ -gdir (points up).
  // We want IMU orientation such that measured accel aligns with -gravity direction.
  cv::Matx33d R_wi = rotFromTwoVectors(am, -gdir);  

  out.R_wi = R_wi;
  out.p_wi = cv::Vec3d(0,0,0);
  out.v_wi = cv::Vec3d(0,0,0);
  out.b_g  = cv::Vec3d(0,0,0);
  out.b_a  = cv::Vec3d(0,0,0);
  out.ts   = t1;
  return true;
}
