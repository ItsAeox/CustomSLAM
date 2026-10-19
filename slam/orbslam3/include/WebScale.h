// web port (Web.Scale.Online, claude/PLAN.md 4d b2): metric scale, gravity direction and accelerometer bias of the
// current map, online and loosely coupled — the C++ form of eval/scale_from_accel.py (after Nuetzi et al. 2011).
// Over windows of the recent tracked trajectory (IMU clock):
//     s (v(t1) - v(t0)) = sum R_wb (f_b - b_a) dt + g_w (t1 - t0)
// v: velocity of the visual camera centre (map units / s; positions smoothed over +-2 frames, central differences),
// R_wb = R_wc R_cb (visual rotation, interpolated at each IMU sample), f_b the accelerometer; unknowns s (metres per
// map unit), g_w (gravity in the map frame, 3) and b_a (accelerometer bias, body frame, 3). Linear least squares, then
// |g_w| = 9.81 with a Huber loss (iteratively reweighted Gauss-Newton). The camera-IMU lever arm is ignored. Restarts
// when the map changes. Offline on EuRoC / TUM-VI (scale_check): median 1.2 % (2 s windows), 3.1 % (1 s).
#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cmath>
#include <deque>
#include <vector>

namespace ORB_SLAM3 {

class WebScaleEstimator {
 public:
  double window = 2.0;   // s
  double keep = 30.0;    // s of trajectory and IMU kept
  // last solution
  bool valid = false;    // a solution exists (>= 10 windows)
  double s = 0, relStd = 1e9, rms = 0, gLinear = 0, span = 0;  // relStd: 1-sigma of s / s from the linear system
  Eigen::Vector3d g = Eigen::Vector3d::Zero(), ba = Eigen::Vector3d::Zero();
  int windows = 0;
  long solves = 0;

  void reset() { poses_.clear(); imu_.clear(); valid = false; s = 0; relStd = 1e9; windows = 0; }
  void setRcb(const Eigen::Matrix3d& Rcb) { Rcb_ = Rcb; }  // R_cb = R_bc^T: body-frame vectors into the camera frame
  void addImu(double t, const Eigen::Vector3d& a) {
    if (!imu_.empty() && t <= imu_.back().t) return;
    imu_.push_back({t, a});
    while (!imu_.empty() && imu_.front().t < t - keep - 5) imu_.pop_front();
  }
  // a tracked frame: time on the IMU clock, camera centre and camera-to-world rotation in the map frame
  void addPose(double t, const Eigen::Vector3d& c, const Eigen::Matrix3d& Rwc) {
    if (!poses_.empty() && t <= poses_.back().t + 1e-6) return;
    poses_.push_back({t, c, Eigen::Quaterniond(Rwc).normalized()});
    while (!poses_.empty() && poses_.front().t < t - keep) poses_.pop_front();
  }

  bool solve() {
    solves++;
    rows_.clear(); rhs_.clear();
    const size_t n = poses_.size();
    size_t s0 = 0;
    for (size_t i = 1; i <= n; i++)  // segments without tracking gaps (> 0.2 s)
      if (i == n || poses_[i].t - poses_[i - 1].t > 0.2) {
        segment(s0, i);
        s0 = i;
      }
    windows = (int)rows_.size() / 3;
    if (windows < 10) { valid = false; return false; }
    // linear least squares on the normal equations
    Eigen::Matrix<double, 7, 7> AtA = Eigen::Matrix<double, 7, 7>::Zero();
    Eigen::Matrix<double, 7, 1> Atb = Eigen::Matrix<double, 7, 1>::Zero();
    for (size_t k = 0; k < rows_.size(); k++) { AtA += rows_[k].transpose() * rows_[k]; Atb += rows_[k].transpose() * rhs_[k]; }
    Eigen::Matrix<double, 7, 1> x = AtA.ldlt().solve(Atb);
    if (!x.allFinite()) { valid = false; return false; }
    gLinear = x.segment<3>(1).norm();
    // quality gate: the free linear solution must find gravity (scale_online_1019: good runs 9.1-10.1 m/s^2, the two
    // failures V1_03 / V2_03 4.6 / 5.8 with 5-40 % scale errors) — otherwise the window set does not constrain s
    if (std::abs(gLinear - 9.81) > 1.5) { valid = false; return false; }
    double r2 = 0;
    for (size_t k = 0; k < rows_.size(); k++) { const double r = rows_[k].dot(x) - rhs_[k]; r2 += r * r; }
    const double sigma2 = r2 / std::max<double>(1, (double)rows_.size() - 7);
    const Eigen::Matrix<double, 7, 7> cov = AtA.inverse() * sigma2;
    // refinement: |g| = 9.81 (soft, weight 10), Huber 0.1 on the window residuals (IRLS Gauss-Newton)
    if (gLinear > 1e-6) x.segment<3>(1) *= 9.81 / gLinear;
    if (x[0] <= 0) x[0] = std::abs(x[0]) + 1e-3;
    for (int it = 0; it < 15; it++) {
      Eigen::Matrix<double, 7, 7> H = Eigen::Matrix<double, 7, 7>::Zero();
      Eigen::Matrix<double, 7, 1> b = Eigen::Matrix<double, 7, 1>::Zero();
      for (size_t k = 0; k < rows_.size(); k++) {
        const double r = rows_[k].dot(x) - rhs_[k], ar = std::abs(r), w = ar <= 0.1 ? 1.0 : 0.1 / ar;
        H += w * rows_[k].transpose() * rows_[k];
        b -= w * rows_[k].transpose() * r;
      }
      const double gn = x.segment<3>(1).norm();
      if (gn > 1e-9) {  // 10 (|g| - 9.81)
        Eigen::Matrix<double, 1, 7> J = Eigen::Matrix<double, 1, 7>::Zero();
        J.segment<3>(1) = 10.0 * x.segment<3>(1).transpose() / gn;
        H += J.transpose() * J;
        b -= J.transpose() * (10.0 * (gn - 9.81));
      }
      const Eigen::Matrix<double, 7, 1> dx = H.ldlt().solve(b);
      if (!dx.allFinite()) break;
      x += dx;
      if (dx.norm() < 1e-7) break;
    }
    double rr = 0;
    for (size_t k = 0; k < rows_.size(); k++) { const double r = rows_[k].dot(x) - rhs_[k]; rr += r * r; }
    if (!x.allFinite() || x[0] <= 0) { valid = false; return false; }
    s = x[0]; g = x.segment<3>(1); ba = x.segment<3>(4);
    rms = std::sqrt(rr / rows_.size());
    relStd = std::sqrt(std::max(0.0, cov(0, 0))) / s;
    span = poses_.back().t - poses_.front().t;
    valid = true;
    return true;
  }

 private:
  struct P { double t; Eigen::Vector3d c; Eigen::Quaterniond q; };
  struct I { double t; Eigen::Vector3d a; };
  std::deque<P> poses_;
  std::deque<I> imu_;
  Eigen::Matrix3d Rcb_ = Eigen::Matrix3d::Identity();
  std::vector<Eigen::Matrix<double, 1, 7>> rows_;
  std::vector<double> rhs_;

  Eigen::Quaterniond rotAt(size_t a, double t) const {  // slerp between the poses around t (a: index with t_a <= t)
    const P& p0 = poses_[a];
    const P& p1 = poses_[a + 1];
    const double u = (t - p0.t) / (p1.t - p0.t);
    return p0.q.slerp(std::min(1.0, std::max(0.0, u)), p1.q);
  }

  void segment(size_t s0, size_t s1) {  // poses [s0, s1)
    const int k = 2;
    const size_t m = s1 - s0;
    if (m < 2 * k + 3 || poses_[s1 - 1].t - poses_[s0].t < 2 * window) return;
    std::vector<double> t(m);
    std::vector<Eigen::Vector3d> ps(m), v(m);
    for (size_t i = 0; i < m; i++) {
      t[i] = poses_[s0 + i].t;
      Eigen::Vector3d sum = Eigen::Vector3d::Zero();
      int c = 0;
      for (int j = -k; j <= k; j++) {
        const long q = (long)i + j;
        if (q >= 0 && q < (long)m) { sum += poses_[s0 + q].c; c++; }
      }
      ps[i] = sum / c;
    }
    for (size_t i = 0; i < m; i++) {  // np.gradient: central differences, one-sided at the ends
      if (i == 0) v[i] = (ps[1] - ps[0]) / (t[1] - t[0]);
      else if (i == m - 1) v[i] = (ps[m - 1] - ps[m - 2]) / (t[m - 1] - t[m - 2]);
      else v[i] = (ps[i + 1] - ps[i - 1]) / (t[i + 1] - t[i - 1]);
    }
    size_t i0 = k;
    while (true) {
      size_t i1 = i0;
      while (i1 < m && t[i1] < t[i0] + window) i1++;
      if (i1 >= m - k) break;
      // IMU samples in [t0, t1]
      std::vector<const I*> sm;
      for (const I& x : imu_) if (x.t >= t[i0] && x.t <= t[i1]) sm.push_back(&x);
      if (sm.size() >= 5) {
        Eigen::Vector3d Af = Eigen::Vector3d::Zero();
        Eigen::Matrix3d AR = Eigen::Matrix3d::Zero();
        double T = 0;
        size_t a = s0 + i0;
        for (size_t j = 0; j < sm.size(); j++) {
          const double dt = sm.size() == 1 ? 0 : (j == 0 ? sm[1]->t - sm[0]->t : j == sm.size() - 1 ? sm[j]->t - sm[j - 1]->t
                                                                                                       : 0.5 * (sm[j + 1]->t - sm[j - 1]->t));
          while (a + 2 < s0 + m && poses_[a + 1].t < sm[j]->t) a++;
          const Eigen::Matrix3d Rwb = rotAt(a, sm[j]->t).toRotationMatrix() * Rcb_;
          Af += Rwb * sm[j]->a * dt;
          AR += Rwb * dt;
          T += dt;
        }
        const Eigen::Vector3d dv = v[i1] - v[i0];
        for (int r = 0; r < 3; r++) {  // s dv - g T + A_R b = A_f
          Eigen::Matrix<double, 1, 7> row = Eigen::Matrix<double, 1, 7>::Zero();
          row[0] = dv[r];
          row[1 + r] = -T;
          row.segment<3>(4) = AR.row(r);
          rows_.push_back(row);
          rhs_.push_back(Af[r]);
        }
      } else {
        size_t n2 = i0;
        while (n2 < m && t[n2] < t[i0] + window / 2) n2++;
        i0 = n2;
        continue;
      }
      size_t n2 = i0;
      while (n2 < m && t[n2] < t[i0] + window / 2) n2++;
      i0 = std::max(i0 + 1, n2);
    }
  }
};

}  // namespace ORB_SLAM3
