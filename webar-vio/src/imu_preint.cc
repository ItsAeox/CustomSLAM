#include "imu_preint.h"
#include <opencv2/calib3d.hpp>   // <-- add this for cv::Rodrigues
#include <algorithm>
#include <cmath>

cv::Matx33d skew(const cv::Vec3d& v){
  return cv::Matx33d( 0, -v[2], v[1],
                      v[2], 0, -v[0],
                     -v[1], v[0], 0 );
}

cv::Matx33d so3Exp(const cv::Vec3d& w){
  cv::Mat rvec = (cv::Mat_<double>(3,1) << w[0], w[1], w[2]);
  cv::Mat R;
  cv::Rodrigues(rvec, R);
  cv::Matx33d out;
  for(int r=0;r<3;r++) for(int c=0;c<3;c++) out(r,c)=R.at<double>(r,c);
  return out;
}

ImuPreint preintegrateImu(const std::vector<ImuMeas>& meas,
                          double t0, double t1,
                          const cv::Vec3d& bg, const cv::Vec3d& ba)
{
  ImuPreint P;
  if (t1 <= t0) return P;
  if (meas.empty()) return P;

  // Collect samples in [t0, t1]
  std::vector<ImuMeas> S;
  S.reserve(256);
  for (auto& m: meas) if (m.t >= t0 && m.t <= t1) S.push_back(m);
  if (S.size() < 2) return P;

  cv::Matx33d dR = cv::Matx33d::eye();
  cv::Vec3d dv(0,0,0), dp(0,0,0);
  double dtSum = 0.0;

  // Simple midpoint integration (upgradeable)
  for (size_t k=1;k<S.size();k++){
    const double dt = S[k].t - S[k-1].t;
    if (dt <= 0) continue;

    const cv::Vec3d w0 = S[k-1].gyro - bg;
    const cv::Vec3d w1 = S[k].gyro   - bg;
    const cv::Vec3d w  = 0.5*(w0+w1);

    const cv::Vec3d a0 = S[k-1].acc - ba;
    const cv::Vec3d a1 = S[k].acc   - ba;
    const cv::Vec3d a  = 0.5*(a0+a1);

    // Update rotation
    const cv::Matx33d dRk = so3Exp(w * dt);
    // Use current dR to rotate accel into i-frame
    const cv::Vec3d a_i = dR * a;

    dp += dv * dt + 0.5 * a_i * (dt*dt);
    dv += a_i * dt;
    dR = dR * dRk;

    dtSum += dt;
  }

  P.dR = dR;
  P.dv = dv;
  P.dp = dp;
  P.dt = dtSum;

  // NOTE: For full tight coupling, fill Jacobians + covariance.
  // We can add that once you confirm noise params + calib.
  return P;
}
