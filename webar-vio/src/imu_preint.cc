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

  // ---- Collect samples around [t0, t1] (need bracketing for interpolation) ----
  // We will build S such that S[0].t==t0 and S.back().t==t1.
  std::vector<ImuMeas> S;
  S.reserve(256);

  // Find first index with meas[idx].t >= t0
  int i0 = -1, i1 = -1;
  for (int i=0; i<(int)meas.size(); ++i){
    if (meas[i].t >= t0) { i0 = i; break; }
  }
  for (int i=0; i<(int)meas.size(); ++i){
    if (meas[i].t >= t1) { i1 = i; break; }
  }
  if (i0 < 0 || i1 < 0) return P;  // not found at all → bail
  
  // If i0 == 0 there's no prior sample for interpolation at t0.
  // Just start from the first available sample.
  auto lerp = [](const cv::Vec3d& a, const cv::Vec3d& b, double u){
    return (1.0-u)*a + u*b;
  };

  auto interpAt = [&](double t, int idx_hi)->ImuMeas{
    // idx_hi is first index with meas[idx_hi].t >= t, so bracket is (idx_hi-1, idx_hi)
    const auto& A = meas[idx_hi-1];
    const auto& B = meas[idx_hi];
    const double denom = std::max(1e-12, (B.t - A.t));
    const double u = (t - A.t) / denom;
    ImuMeas M;
    M.t = t;
    M.gyro = lerp(A.gyro, B.gyro, u);
    M.acc  = lerp(A.acc , B.acc , u);
    return M;
  };

  if (i0 == 0) {
    if (std::abs(meas[0].t - t0) < 1e-9) {
        S.push_back(meas[0]);
    } else {
        return P;
    }
  } else {
      S.push_back(interpAt(t0, i0));
  }

  // Add all real samples strictly inside (t0, t1)
  for (int i=i0; i<i1; ++i){
    if (meas[i].t > t0 && meas[i].t < t1) S.push_back(meas[i]);
  }

  // Add interpolated t1 sample (if i1 is exactly at t1, interpAt still works fine)
  if (i1 == 0) return P;
  S.push_back(interpAt(t1, i1));

  if (S.size() < 2) return P;

  cv::Matx33d dR = cv::Matx33d::eye();
  cv::Vec3d dv(0,0,0), dp(0,0,0);

  for (size_t k=1; k<S.size(); ++k){
    const double dt = S[k].t - S[k-1].t;
    if (dt <= 0) continue;

    const cv::Vec3d w0 = S[k-1].gyro - bg;
    const cv::Vec3d w1 = S[k].gyro   - bg;
    const cv::Vec3d w  = 0.5*(w0+w1);

    const cv::Vec3d a0 = S[k-1].acc - ba;
    const cv::Vec3d a1 = S[k].acc   - ba;
    const cv::Vec3d a  = 0.5*(a0+a1);

    const cv::Matx33d dRk = so3Exp(w * dt);

    // IMPORTANT:
    // Treat accelerometer as measuring "specific force" (includes gravity in sensor frame).
    // Then the propagation in System adds g_world_ separately.
    // If your accel samples are actually "acc including gravity", this is correct.
    //
    // But if your accel samples are "linear acceleration" (gravity removed),
    // you MUST NOT add g_world_ in propagation (System), OR you will double-count gravity.
    //
    // We'll keep the standard VIO assumption here: accel == specific force.
    const cv::Vec3d a_i = dR * a;

    dp += dv * dt + 0.5 * a_i * (dt*dt);
    dv += a_i * dt;
    dR  = dR * dRk;

    // keep preintegrated rotation numerically clean
    cv::Mat M(3,3,CV_64F);
    for (int r=0;r<3;r++) for (int c=0;c<3;c++) M.at<double>(r,c) = dR(r,c);
    cv::SVD svd(M);
    cv::Mat Rn = svd.u * svd.vt;
    if (cv::determinant(Rn) < 0.0) {
      cv::Mat U = svd.u.clone();
      U.col(2) *= -1.0;
      Rn = U * svd.vt;
    }
    for (int r=0;r<3;r++) for (int c=0;c<3;c++) dR(r,c) = Rn.at<double>(r,c);
  }

  P.dR = dR;
  P.dv = dv;
  P.dp = dp;
  P.dt = (t1 - t0);

  // NOTE: For full tight coupling, fill Jacobians + covariance.
  // We can add that once you confirm noise params + calib.
  return P;
}
