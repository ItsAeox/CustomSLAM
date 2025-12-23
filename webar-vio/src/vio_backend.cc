#include "vio_backend.h"
#include <opencv2/calib3d.hpp>
#include <cmath>
#include <algorithm>

// ---- SO(3) helpers ---------------------------------------------------------
static inline cv::Matx33d ortho(const cv::Matx33d& Rm) {
  cv::Mat R(3,3,CV_64F);
  for (int r=0;r<3;r++) for (int c=0;c<3;c++) R.at<double>(r,c)=Rm(r,c);
  cv::SVD svd(R);
  cv::Mat U=svd.u, Vt=svd.vt;
  cv::Mat RR = U*Vt;
  if (cv::determinant(RR) < 0) {
    U.col(2) *= -1;
    RR = U*Vt;
  }
  cv::Matx33d out;
  for (int r=0;r<3;r++) for (int c=0;c<3;c++) out(r,c)=RR.at<double>(r,c);
  return out;
}

static inline cv::Vec3d so3Log(const cv::Matx33d& Rm) {
  cv::Mat R(3,3,CV_64F);
  for (int r=0;r<3;r++) for (int c=0;c<3;c++) R.at<double>(r,c)=Rm(r,c);
  cv::Mat rvec;
  cv::Rodrigues(R, rvec);
  return cv::Vec3d(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2));
}

// ---- camera-from-imu -> world-from-camera conversion ------------------------
// state is world-from-imu: R_wi, p_wi
// calib has camera-from-imu: R_ci, t_ci
// Then world-from-camera:
//   R_wc = R_wi * R_ci^T
//   p_wc = p_wi - R_wi * R_ci^T * t_ci
static inline void imuStateToCamPose(const VioCalib& c, const VioState& s,
                                    cv::Matx33d& R_wc, cv::Vec3d& p_wc) {
  const cv::Matx33d R_ic = c.R_ci.t();
  R_wc = s.R_wi * R_ic;
  p_wc = s.p_wi - s.R_wi * (R_ic * c.t_ci);
}

// project Xw with Twc
static inline bool project(const VioCalib& c,
                           const cv::Matx33d& R_wc, const cv::Vec3d& p_wc,
                           const cv::Vec3d& Xw,
                           double& u, double& v) {
  // Xc = Rcw*(Xw - p_wc)
  cv::Matx33d R_cw = R_wc.t();
  cv::Vec3d Xc = R_cw * (Xw - p_wc);
  if (Xc[2] <= 1e-6) return false;
  const double fx = c.K(0,0), fy = c.K(1,1), cx = c.K(0,2), cy = c.K(1,2);
  u = fx*(Xc[0]/Xc[2]) + cx;
  v = fy*(Xc[1]/Xc[2]) + cy;
  return true;
}

void VioBackend::addKeyframe(const VioState& init,
                            const std::vector<FeatureObs>& obs,
                            const ImuPreint& pim)
{
  if (states_.empty()) {
    states_.push_back(init);
    obs_.push_back(obs);
    // dummy IMU
    pim_.push_back(ImuPreint{});
    return;
  }
  states_.push_back(init);
  obs_.push_back(obs);
  pim_.push_back(pim);

  // keep window bounded
  while ((int)states_.size() > window_) {
    states_.erase(states_.begin());
    obs_.erase(obs_.begin());
    pim_.erase(pim_.begin());
  }
}

bool VioBackend::optimize(int max_iters)
{
  if (states_.size() < 2) return false;

  // Optimize ONLY the latest state against:
  //  - IMU preintegration factor from previous state
  //  - Reprojection residuals using map points
  //
  // This is tight-coupled (IMU + reprojection in same solve),
  // but cheap enough for WASM.

  const int k = (int)states_.size() - 1;
  const VioState s_prev = states_[k-1];
  VioState s = states_[k];
  const ImuPreint& P = pim_[k];
  const auto& OBS = obs_[k];

  if (P.dt <= 1e-6) {
    // no IMU span, still allow reprojection-only refine
  }

  // weights (tune later)
  const double w_imu_R = 1.0;      // rad
  const double w_imu_v = 1.0;      // m/s
  const double w_imu_p = 1.0;      // m
  const double w_px    = 1.0 / 2.0; // 2px sigma

  // variables: x = [dtheta(3), dp(3), dv(3)] (9-dim)
  auto applyDelta = [&](const cv::Mat& dx) {
    cv::Vec3d dth(dx.at<double>(0), dx.at<double>(1), dx.at<double>(2));
    cv::Vec3d dp (dx.at<double>(3), dx.at<double>(4), dx.at<double>(5));
    cv::Vec3d dv (dx.at<double>(6), dx.at<double>(7), dx.at<double>(8));
    s.R_wi = ortho(s.R_wi * so3Exp(dth));
    s.p_wi += dp;
    s.v_wi += dv;
  };

  auto buildResidual = [&](VioState ss, cv::Mat& r) {
    // IMU residuals
    // Predicted from prev using preint (ignoring bias jacobians for now)
    const double dt = P.dt;
    const cv::Vec3d g = calib_.g_w;

    cv::Matx33d R_pred = s_prev.R_wi * P.dR;
    cv::Vec3d v_pred = s_prev.v_wi + g*dt + s_prev.R_wi * P.dv;
    cv::Vec3d p_pred = s_prev.p_wi + s_prev.v_wi*dt + 0.5*g*(dt*dt) + s_prev.R_wi * P.dp;

    cv::Matx33d dR = R_pred.t() * ss.R_wi;
    cv::Vec3d rR = so3Log(dR);
    cv::Vec3d rv = (ss.v_wi - v_pred);
    cv::Vec3d rp = (ss.p_wi - p_pred);

    // Reprojection residuals
    cv::Matx33d R_wc; cv::Vec3d p_wc;
    imuStateToCamPose(calib_, ss, R_wc, p_wc);

    // residual vector: [imu(9), reproj(2*N)]
    const int M = 9 + 2*(int)OBS.size();
    r = cv::Mat(M, 1, CV_64F);
    int row = 0;

    // IMU (weighted)
    r.at<double>(row++) = w_imu_R * rR[0];
    r.at<double>(row++) = w_imu_R * rR[1];
    r.at<double>(row++) = w_imu_R * rR[2];
    r.at<double>(row++) = w_imu_v * rv[0];
    r.at<double>(row++) = w_imu_v * rv[1];
    r.at<double>(row++) = w_imu_v * rv[2];
    r.at<double>(row++) = w_imu_p * rp[0];
    r.at<double>(row++) = w_imu_p * rp[1];
    r.at<double>(row++) = w_imu_p * rp[2];

    for (const auto& ob : OBS) {
      double uhat=0, vhat=0;
      bool ok = project(calib_, R_wc, p_wc, ob.Xw, uhat, vhat);
      if (!ok) {
        // push big residual to discourage this state, but not explode
        r.at<double>(row++) = w_px * 50.0;
        r.at<double>(row++) = w_px * 50.0;
      } else {
        r.at<double>(row++) = w_px * (uhat - ob.u);
        r.at<double>(row++) = w_px * (vhat - ob.v);
      }
    }
  };

  // Gauss-Newton with numeric Jacobian (9 vars)
  for (int it=0; it<max_iters; ++it) {
    cv::Mat r0;
    buildResidual(s, r0);

    const int m = r0.rows;
    const int n = 9;
    cv::Mat J(m, n, CV_64F);

    const double eps = 1e-5;
    for (int j=0;j<n;++j) {
      VioState sp = s;
      cv::Mat dx = cv::Mat::zeros(n,1,CV_64F);
      dx.at<double>(j) = eps;

      // apply small delta to copy
      {
        cv::Vec3d dth(dx.at<double>(0), dx.at<double>(1), dx.at<double>(2));
        cv::Vec3d dp (dx.at<double>(3), dx.at<double>(4), dx.at<double>(5));
        cv::Vec3d dv (dx.at<double>(6), dx.at<double>(7), dx.at<double>(8));
        sp.R_wi = ortho(sp.R_wi * so3Exp(dth));
        sp.p_wi += dp;
        sp.v_wi += dv;
      }

      cv::Mat rp;
      buildResidual(sp, rp);
      cv::Mat col = (rp - r0) * (1.0/eps);
      col.copyTo(J.col(j));
    }

    // Solve normal equations
    cv::Mat H = J.t() * J;
    cv::Mat b = J.t() * r0;

    // Damping (LM-lite)
    for (int d=0; d<n; ++d) H.at<double>(d,d) += 1e-3;

    cv::Mat dx;
    if (!cv::solve(H, -b, dx, cv::DECOMP_SVD)) break;

    // step
    applyDelta(dx);

    const double stepNorm = cv::norm(dx);
    if (stepNorm < 1e-6) break;
  }

  states_[k] = s;
  return true;
}
