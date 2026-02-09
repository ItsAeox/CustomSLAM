#include "vio_initializer.h"
#include <opencv2/calib3d.hpp>
#include <cmath>
#include <algorithm>

static inline cv::Vec3d safeNormed(const cv::Vec3d& v, double eps=1e-9){
  double n = cv::norm(v);
  if (n < eps) return cv::Vec3d(0,0,0);
  return v*(1.0/n);
}

static inline cv::Vec3d so3Log(const cv::Matx33d& Rm){
  cv::Mat R(3,3,CV_64F);
  for(int r=0;r<3;r++) for(int c=0;c<3;c++) R.at<double>(r,c)=Rm(r,c);
  cv::Mat rv; cv::Rodrigues(R, rv);
  return cv::Vec3d(rv.at<double>(0),rv.at<double>(1),rv.at<double>(2));
}

bool VioInitializer::ready(double minSpanSec, int minKfs) const {
  if ((int)kfs_.size() < minKfs) return false;
  const double span = kfs_.back().ts - kfs_.front().ts;
  return span >= minSpanSec;
}

// --- Step 1: gyro bias from rotation-only consistency -----------------------
// We want IMU preintegrated ΔR(bg) to match visual relative rotation between KFs.
// Visual relative (camera): Rc_i^c_{i+1}. You have Rwc, so Rcw = Rwc^T.
// Relative: R_cj_ci = Rcw_j * Rwc_i

bool VioInitializer::solveScaleGravityVelocity(const std::vector<ImuMeas>& imu,
                                               const VioCalib& calib,
                                               const cv::Vec3d& bg,
                                               double& s_out,
                                               cv::Vec3d& g_out,
                                               std::vector<cv::Vec3d>& v_out)
{
  const int N = (int)kfs_.size();
  if (N < 4) return false;

  // Unknowns: v0..v_{N-1} (3N), gravity g (3), scale s (1)
  const int nv = 3*N;
  const int ng = 3;
  const int ns = 1;
  const int dim = nv + ng + ns;

  // Equations per edge i->j: 6 (velocity + position)
  const int M = 6*(N-1);
  cv::Mat A = cv::Mat::zeros(M, dim, CV_64F);
  cv::Mat b = cv::Mat::zeros(M, 1, CV_64F);

  auto vIndex = [&](int i, int k){ return 3*i + k; };
  const int gIndex = nv;
  const int sIndex = nv + ng;

  // We treat camera pose as “IMU pose” rotation-wise for init (common approximation),
  // and ignore lever arm t_ci for init solve.
  // If you want, we can incorporate t_ci later, but it’s not the main failure right now.
  for (int i=0;i<N-1;++i){
    const auto& Ki = kfs_[i];
    const auto& Kj = kfs_[i+1];
    const double dt = Kj.ts - Ki.ts;
    if (dt <= 1e-4) continue;

    ImuPreint P = preintegrateImu(imu, Ki.ts, Kj.ts, bg, cv::Vec3d(0,0,0));
    if (P.dt <= 1e-6) continue;

    // Approx world-from-IMU using camera pose and extrinsic:
    // R_ci = camera-from-IMU  =>  R_wi ≈ R_wc * R_ci
    cv::Matx33d Rwi = Ki.Rwc_vo * calib.R_ci;

    // Visual delta position in VO world
    cv::Vec3d dp_vo = (Kj.twc_vo - Ki.twc_vo);

    // Row base
    const int r0 = 6*i;

    // --- Velocity equation: v_j - v_i - g*dt = Rwi * dv_ij
    // Put v_j ( +1 ), v_i ( -1 ), g ( -dt )
    for(int k=0;k<3;k++){
      A.at<double>(r0 + k, vIndex(i+1,k)) =  1.0;
      A.at<double>(r0 + k, vIndex(i  ,k)) = -1.0;
      A.at<double>(r0 + k, gIndex + k)    = -dt;
      // RHS
      cv::Vec3d rhs = Rwi * P.dv;
      b.at<double>(r0 + k) = rhs[k];
    }

    // --- Position equation:
    // s*(p_vo_j - p_vo_i) = v_i*dt + 0.5*g*dt^2 + Rwi*dp_ij
    for(int k=0;k<3;k++){
      // v_i*dt
      A.at<double>(r0 + 3 + k, vIndex(i,k)) = dt;
      // g
      A.at<double>(r0 + 3 + k, gIndex + k)  = 0.5 * dt * dt;
      // scale term goes to LHS => -dp_vo * s on LHS, so move to A
      A.at<double>(r0 + 3 + k, sIndex)      = -dp_vo[k];
      // RHS
      cv::Vec3d rhs = Rwi * P.dp;
      b.at<double>(r0 + 3 + k) = rhs[k];
    }
  }

  // Solve least squares
  cv::Mat x;
  if (!cv::solve(A, b, x, cv::DECOMP_SVD)) return false;

  // Extract
  v_out.resize(N);
  for (int i=0;i<N;i++){
    v_out[i] = cv::Vec3d(x.at<double>(vIndex(i,0)),
                         x.at<double>(vIndex(i,1)),
                         x.at<double>(vIndex(i,2)));
  }
  g_out = cv::Vec3d(x.at<double>(gIndex+0),
                    x.at<double>(gIndex+1),
                    x.at<double>(gIndex+2));
  s_out = x.at<double>(sIndex);

  // Normalize gravity magnitude to 9.81 and rescale velocities accordingly
  const double gmag = cv::norm(g_out);
  if (gmag > 1e-6) {
    const double k = 9.81 / gmag;
    g_out *= k;
    for (auto& v: v_out) v *= k;
    s_out *= k;
  }

  // Scale sanity
  if (!std::isfinite(s_out) || s_out <= 1e-6) return false;

  return true;
}

VioInitResult VioInitializer::solve(const std::vector<ImuMeas>& imu,
                                    const VioCalib& calib,
                                    int maxOuterIters)
{
  VioInitResult out;
  if (!ready()) return out;

  // Outer iteration: bg -> (s,g,v) -> (optional refine)
  cv::Vec3d bg(0,0,0);
  cv::Vec3d ba(0,0,0);
  double s = 1.0;
  cv::Vec3d g(0,-9.81,0);
  std::vector<cv::Vec3d> v;

  // --- Solve gyro bias by 1D GN on bg (simple numeric) ---
  // We minimize sum || log( dR_cam(bg)^T * R_vis ) ||^2
  auto cost = [&](const cv::Vec3d& bgTest){
    double C = 0.0;
    for (size_t i=0;i+1<kfs_.size();++i){
      const auto& A = kfs_[i];
      const auto& B = kfs_[i+1];
      ImuPreint P = preintegrateImu(imu, A.ts, B.ts, bgTest, ba);
      if (P.dt <= 1e-6) continue;

      // visual relative rotation camera i->j
      cv::Matx33d Rcw_i = A.Rwc_vo.t();
      cv::Matx33d Rcw_j = B.Rwc_vo.t();
      cv::Matx33d R_cj_ci = Rcw_j * A.Rwc_vo;

      // imu delta in camera frame
      cv::Matx33d dR_cam = calib.R_ci * P.dR * calib.R_ci.t();

      cv::Vec3d e = so3Log(dR_cam.t() * R_cj_ci);
      C += e.dot(e);
    }
    return C;
  };

  for (int it=0; it<10; ++it){
    const double eps = 1e-5;
    const double c0 = cost(bg);
    cv::Mat J(1,3,CV_64F);
    for (int k=0;k<3;k++){
      cv::Vec3d bp = bg; bp[k] += eps;
      double cp = cost(bp);
      J.at<double>(0,k) = (cp - c0)/eps;
    }
    // small gradient step (safe for WASM)
    cv::Vec3d grad(J.at<double>(0,0),J.at<double>(0,1),J.at<double>(0,2));
    bg -= 0.05 * grad; // step size (tune 0.01..0.1)
    if (cv::norm(grad) < 1e-6) break;
  }

  // --- Solve scale + gravity + velocities (linear LS) ---
  if (!solveScaleGravityVelocity(imu, calib, bg, s, g, v)) return out;

  out.ok = true;
  out.scale = s;
  out.g_w = g;
  out.b_g = bg;
  out.b_a = ba;
  out.v_w = v;
  return out;
}
