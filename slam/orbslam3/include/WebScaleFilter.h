// web port (Web.Scale.Online 1 + Web.Scale.Method 1, claude/PLAN.md 4d approach B): metric scale, gravity, velocity and
// accelerometer bias of the current map by a loosely coupled visual-inertial filter — the C++ form of eval/vio_filter.py
// (bank method, defaults of that prototype). After Weiss & Siegwart (ICRA 2011) / Nuetzi et al. (JIRS 2011), who put
// the visual scale into an EKF; an EKF on the scale converges confidently to wrong values from a poor start (vio_filter.py
// --method ekf --init prior: -9 to -22 % on ground-truth poses), so the scale is not a state here but a grid:
//   per map, in the map frame, metric units (camera centre; the camera-IMU lever arm is ignored):
//     p' = v,  v' = R_vb (f - b_a) + g      R_vb = visual camera rotation x R_cb, carried by the gyro between frames
//     tracked frame: p_vis = p / s          p_vis in map units (noise sigVis), s = metres per map unit
//   For a fixed s this is linear in x = (p, v, g, b_a): one Kalman filter per (s, accelerometer noise level) gives the
//   exact likelihood of that pair; the noise level is marginalised (an assumed level that is too high biases s upwards).
//   |g| = 9.81 is enforced after every update (a constraint, not evidence). Estimate: parabola peak of the marginal
//   log-likelihood over ln s; 1-sigma: the posterior spread of ln s (at least half a grid step).
// Banks start when the map has settled (>= 10 keyframes, >= 2 s: early bundle adjustment rescales the map), a new bank
// every restart / 2 s, banks live restart s (the oldest reports: windows of restart/2 .. restart s follow scale changes
// of the map); a visual update whose NIS on the most likely hypothesis exceeds the gate is skipped (relocalisation jumps),
// a bank that skips 10 in a row is dropped. The IMU between two frames is the same for every hypothesis, so the
// transition (Phi, input u) and the noise (Q = q QA + QO) are accumulated once per frame; hypotheses whose
// log-likelihood falls 100 below the best are pruned from that bank (the next bank starts with the full grid).
// Results on the synthetic set (EXPERIMENTS vio_filter_sweep, prototype): converged in 7 / 9 / 6 / 11 of 12 runs of ar /
// gesture / orbit / shake (online window estimator 3 / 5 / 0 / 0), median scale error 0.7-2.5 %.
#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <deque>
#include <memory>
#include <vector>

namespace ORB_SLAM3 {

class WebScaleFilter {
 public:
  using M12 = Eigen::Matrix<double, 12, 12>;
  using V12 = Eigen::Matrix<double, 12, 1>;
  using M3 = Eigen::Matrix3d;
  using V3 = Eigen::Vector3d;

  // settings (vio_filter.py defaults, the noise-grid variant)
  int nScales = 100;
  double sMin = 0.05, sMax = 200;
  std::vector<double> sigA{0.03, 0.1, 0.3};  // accelerometer + model noise per sample (m/s^2), marginalised
  double sigVis = 0.005;                      // visual position noise (map units)
  double sigBa = 0.1, baWalk = 0.002;         // bias prior (m/s^2), random walk (m/s^2 / sqrt(s))
  double restart = 30, gate = 16.27;
  double prune = 100;  // log-likelihood below the best at which a hypothesis is dropped from its bank (0 = never). 30 was
                     // safe on the synthetic set but not on real phone data (rec_20261021_070221: relative likelihoods swing,
                     // pruned scales would have come back -> falsely narrow posterior, a 3.7-4.8x scale claimed); 60 and
                     // 100 gave the unpruned results there and on the synthetic set, at 1/3-1/6 of its cost
  double stableBack = 3.0;  // s: the reported 1-sigma is at least the relative change of the estimate over this time (0 = off)
  int maxSkips = 10, startKFs = 10;
  double startTime = 2.0;

  // last estimate (the fields WebScaleEstimator offers, so trace / page code reads either)
  bool valid = false;
  double s = 0, relStd = 1e9;
  int windows = 0;  // here: visual updates of the reporting bank
  V3 g = V3::Zero();

  void reset() {
    banks_.clear(); imu_.clear(); recent_.clear(); hist_.clear(); valid = false; s = 0; relStd = 1e9; windows = 0;
    haveR_ = false; tMap0_ = -1;
  }
  void setRcb(const M3& Rcb) { Rcb_ = Rcb; }
  void addImu(double t, const V3& a, const V3& w) {
    if ((!imu_.empty() && t <= imu_.back().t) || (!recent_.empty() && t <= recent_.back().t)) return;
    imu_.push_back({t, a, w});
    recent_.push_back({t, a, w});  // the last 0.5 s (start value of gravity)
    while (recent_.size() > 1 && recent_.front().t < t - 0.5) recent_.pop_front();
    if (nImu_++ == 0) tImu0_ = t;
    tImu1_ = t;
  }

  // a tracked frame (time on the IMU clock, camera centre and camera-to-world rotation in the map frame, keyframes in
  // the map); call only for frames tracked OK, in time order; reset() when the map changes
  void addFrame(double t, const V3& c, const M3& Rwc, int mapKFs) {
    if (tMap0_ < 0 && mapKFs > 0) tMap0_ = t;
    // IMU up to the frame: shared transition for every bank, rotation carried by the gyro (midpoint rule)
    Phi_.setIdentity(); u_.setZero(); QA_.setZero(); QO_.setZero();
    double tl = tLast_;
    size_t k = 0;
    for (; k < imu_.size() && imu_[k].t <= t; k++) {
      if (haveR_ && imu_[k].t > tRot_) {
        const double dt = imu_[k].t - tRot_;
        const V3 w = 0.5 * (prevW_ + imu_[k].w), f = 0.5 * (prevA_ + imu_[k].a);
        const M3 Rmid = Rvb_ * expSO3(w * dt / 2);
        step(Rmid, f, imu_[k].t - tl);
        tl = imu_[k].t;
        Rvb_ = Rvb_ * expSO3(w * dt);
        tRot_ = imu_[k].t;
      }
      prevW_ = imu_[k].w; prevA_ = imu_[k].a; haveImu_ = true;
    }
    imu_.erase(imu_.begin(), imu_.begin() + k);
    Rvb_ = Rwc * Rcb_;  // the visual rotation at the frame
    if (haveR_ && haveImu_ && t > tl) step(Rvb_, prevA_, t - tl);  // last stretch to the frame time
    haveR_ = true; tRot_ = t; tLast_ = t;
    for (auto& b : banks_) b->predictUpdate(*this, c);
    // drop banks that keep rejecting the visual pose (the map jumped) or are older than restart
    for (size_t i = 0; i < banks_.size();) {
      if (banks_[i]->skips > maxSkips || (restart > 0 && t - banks_[i]->t0 > restart)) banks_.erase(banks_.begin() + i);
      else i++;
    }
    const bool settled = tMap0_ >= 0 && mapKFs >= startKFs && t - tMap0_ >= startTime;
    if (settled && haveImu_ && (banks_.empty() || (restart > 0 && t - banks_.back()->t0 >= restart / 2)))
      banks_.push_back(std::make_unique<Bank>(*this, t, c, fMean(t)));
    if (banks_.empty()) { valid = false; return; }
    double sd = 0;
    bool edge = false;
    banks_.front()->estimate(*this, s, sd, edge);
    // reported 1-sigma: at least the estimate's change over the last stableBack s (the posterior alone is over-confident
    // early: on the synthetic set 'stable within 3 % over 3 s' cut the median error of converged frames, e.g. ar 11.8 -> 3.7 %,
    // scripts/experiments/2026-10-30/conv_rule.py); a reader's 'relStd < 5 %' then also means 'not moving'
    hist_.push_back({t, s});
    while (hist_.size() > 1 && hist_[1].first <= t - stableBack) hist_.pop_front();
    const double change = (stableBack > 0 && hist_.front().first <= t - stableBack + 0.5)
                              ? std::abs(std::log(s / hist_.front().second)) : (stableBack > 0 ? 1e9 : 0.0);
    relStd = edge ? 1e9 : std::max(sd, change);
    g = banks_.front()->gBest(*this);
    windows = banks_.front()->nUpd;
    valid = !edge && banks_.front()->nUpd >= 10;
  }

 private:
  struct Imu { double t; V3 a, w; };

  static M3 expSO3(const V3& w) {
    const double th = w.norm();
    if (th < 1e-12) return M3::Identity();
    return Eigen::AngleAxisd(th, w / th).toRotationMatrix();
  }

  // one IMU step for all hypotheses: x' = F x + B f dt-terms, accumulated as Phi, u; Q = q QA + QO
  void step(const M3& R, const V3& f, double dt) {
    if (dt <= 0) return;
    M12 F = M12::Identity();
    F.block<3, 3>(0, 3) = M3::Identity() * dt;
    F.block<3, 3>(0, 6) = M3::Identity() * 0.5 * dt * dt;
    F.block<3, 3>(0, 9) = -0.5 * dt * dt * R;
    F.block<3, 3>(3, 6) = M3::Identity() * dt;
    F.block<3, 3>(3, 9) = -dt * R;
    V12 in = V12::Zero();  // the measured specific force rotated into the map frame
    const V3 Rf = R * f;
    in.segment<3>(0) = 0.5 * dt * dt * Rf;
    in.segment<3>(3) = dt * Rf;
    M12 Qa = M12::Zero(), Qo = M12::Zero();
    Qa.block<3, 3>(0, 0) = M3::Identity() * dt * dt * dt / 3;
    Qa.block<3, 3>(0, 3) = Qa.block<3, 3>(3, 0) = M3::Identity() * dt * dt / 2;
    Qa.block<3, 3>(3, 3) = M3::Identity() * dt;
    Qo.block<3, 3>(6, 6) = M3::Identity() * 1e-6 * dt;
    Qo.block<3, 3>(9, 9) = M3::Identity() * baWalk * baWalk * dt;
    Phi_ = F * Phi_;
    u_ = F * u_ + in;
    QA_ = F * QA_ * F.transpose() + Qa;
    QO_ = F * QO_ * F.transpose() + Qo;
  }

  V3 fMean(double t) const {  // mean specific force over the 0.5 s before t (the gravity start value)
    V3 m = V3::Zero();
    int n = 0;
    for (const auto& r : recent_)
      if (r.t > t - 0.5 && r.t <= t) { m += r.a; n++; }
    return n ? V3(m / n) : prevA_;
  }

  struct Bank {
    std::vector<V12> x;
    std::vector<M12> P;
    std::vector<double> L, sj, qj;
    std::vector<char> active;  // pruned: log-likelihood more than prune below the best (weight < e^-30), never updated again
    double t0;
    int nUpd = 0, skips = 0;

    Bank(const WebScaleFilter& F, double t, const V3& c, const V3& fm) : t0(t) {
      const int nq = (int)F.sigA.size(), J = F.nScales * nq;
      const double rate = F.imuRate();
      x.resize(J); P.resize(J); L.assign(J, 0); sj.resize(J); qj.resize(J); active.assign(J, 1);
      V3 g0 = -F.Rvb_ * fm;
      g0 = 9.81 * g0 / std::max(g0.norm(), 1e-6);
      for (int i = 0; i < F.nScales; i++) {
        const double s = std::exp(std::log(F.sMin) + (std::log(F.sMax) - std::log(F.sMin)) * i / (F.nScales - 1));
        for (int k = 0; k < nq; k++) {
          const int j = i * nq + k;
          sj[j] = s;
          const double qa = F.sigA[k] / std::sqrt(rate);  // per-sample sigma -> continuous density
          qj[j] = qa * qa;
          x[j].setZero();
          x[j].segment<3>(0) = s * c;
          x[j].segment<3>(6) = g0;
          P[j].setZero();
          P[j].block<3, 3>(0, 0) = M3::Identity() * (F.sigVis * s) * (F.sigVis * s);
          P[j].block<3, 3>(3, 3) = M3::Identity();
          P[j].block<3, 3>(6, 6) = M3::Identity();
          P[j].block<3, 3>(9, 9) = M3::Identity() * F.sigBa * F.sigBa;
        }
      }
    }

    void predictUpdate(const WebScaleFilter& F, const V3& z) {
      const int J = (int)x.size();
      for (int j = 0; j < J; j++) {
        if (!active[j]) continue;
        x[j] = F.Phi_ * x[j] + F.u_;
        P[j] = F.Phi_ * P[j] * F.Phi_.transpose() + qj[j] * F.QA_ + F.QO_;
      }
      // gate on the most likely hypothesis
      int jb = 0;
      for (int j = 1; j < J; j++) if (active[j] && (!active[jb] || L[j] > L[jb])) jb = j;
      const double r2v = F.sigVis * F.sigVis;
      auto innov = [&](int j, V3& nu, M3& S) {
        nu = z - x[j].segment<3>(0) / sj[j];
        S = P[j].block<3, 3>(0, 0) / (sj[j] * sj[j]) + M3::Identity() * r2v;
      };
      {
        V3 nu; M3 S;
        innov(jb, nu, S);
        if (F.gate > 0 && nUpd >= 30 && nu.dot(S.ldlt().solve(nu)) > F.gate) { skips++; return; }
      }
      skips = 0;
      nUpd++;
      double Lmax = -1e300;
      for (int j = 0; j < J; j++) {
        if (!active[j]) continue;
        V3 nu; M3 S;
        innov(j, nu, S);
        const Eigen::LDLT<M3> ld(S);
        // K = P H^T S^-1 with H = [I/s 0 0 0]
        const Eigen::Matrix<double, 12, 3> PHt = P[j].block<12, 3>(0, 0) / sj[j];
        const Eigen::Matrix<double, 12, 3> K = ld.solve(PHt.transpose()).transpose();
        x[j] += K * nu;
        // Joseph form (I - KH) P (I - KH)^T + K R K^T, expanded: P - K (HP) - (HP)^T K^T + K S K^T (same result, ~3x cheaper)
        const Eigen::Matrix<double, 12, 12> KHP = K * PHt.transpose();
        P[j] += -KHP - KHP.transpose() + K * S * K.transpose();
        L[j] += -0.5 * (nu.dot(ld.solve(nu)) + std::log(std::max(S.determinant(), 1e-300)) + 3 * std::log(2 * M_PI));
        for (int it = 0; it < 2; it++) {  // |g| = 9.81
          const V3 gv = x[j].segment<3>(6);
          const double n = gv.norm();
          if (n < 1e-9) break;
          V12 Hg = V12::Zero();
          Hg.segment<3>(6) = gv / n;
          const V12 PH = P[j] * Hg;
          const double Sg = Hg.dot(PH) + 1e-4;
          const V12 Kg = PH / Sg;
          x[j] += Kg * (9.81 - n);
          P[j] -= Kg * PH.transpose();
        }
        P[j] = 0.5 * (P[j] + P[j].transpose()).eval();
        Lmax = std::max(Lmax, L[j]);
      }
      if (nUpd >= 30) {  // the best scale's neighbours (+-3 grid steps) stay: the parabola peak needs them
        const int nq = (int)F.sigA.size();
        int jb2 = 0;
        for (int j = 1; j < J; j++) if (active[j] && (!active[jb2] || L[j] > L[jb2])) jb2 = j;
        for (int j = 0; j < J; j++)
          if (active[j] && F.prune > 0 && L[j] < Lmax - F.prune && std::abs(j / nq - jb2 / nq) > 3) active[j] = 0;
      }
    }


    // marginal over the noise levels, parabola peak over ln s
    void estimate(const WebScaleFilter& F, double& s, double& sd, bool& edge) const {
      const int nq = (int)F.sigA.size(), ns = F.nScales;
      std::vector<double> Ls(ns);
      double mx = -1e300;  // over the active hypotheses (pruned ones keep a stale value)
      for (size_t j = 0; j < L.size(); j++) if (active[j]) mx = std::max(mx, L[j]);
      for (int i = 0; i < ns; i++) {
        double acc = 0;
        for (int k = 0; k < nq; k++)
          if (active[i * nq + k]) acc += std::exp(L[i * nq + k] - mx);
        Ls[i] = std::log(acc + 1e-300);
      }
      const int jb = (int)(std::max_element(Ls.begin(), Ls.end()) - Ls.begin());
      const double step = (std::log(F.sMax) - std::log(F.sMin)) / (ns - 1);
      auto ls = [&](int i) { return std::log(F.sMin) + step * i; };
      double est = ls(jb);
      if (jb > 0 && jb < ns - 1) {
        const double y0 = Ls[jb - 1], y1 = Ls[jb], y2 = Ls[jb + 1], den = y0 - 2 * y1 + y2;
        if (den < 0) est = ls(jb) + 0.5 * (y0 - y2) / den * step;
      }
      double wsum = 0, mean = 0, var = 0;
      for (int i = 0; i < ns; i++) { const double w = std::exp(Ls[i] - Ls[jb]); wsum += w; mean += w * ls(i); }
      mean /= wsum;
      for (int i = 0; i < ns; i++) { const double w = std::exp(Ls[i] - Ls[jb]); var += w * (ls(i) - mean) * (ls(i) - mean); }
      sd = std::max(std::sqrt(var / wsum), 0.5 * step);
      s = std::exp(est);
      edge = jb == 0 || jb == ns - 1;
    }

    V3 gBest(const WebScaleFilter&) const {
      int jb = 0;
      for (int j = 1; j < (int)L.size(); j++) if (active[j] && (!active[jb] || L[j] > L[jb])) jb = j;
      return x[jb].segment<3>(6);
    }
  };

  double imuRate() const { return nImu_ > 10 && tImu1_ > tImu0_ ? (nImu_ - 1) / (tImu1_ - tImu0_) : 60.0; }  // samples / s

  std::deque<Imu> imu_, recent_;
  std::deque<std::pair<double, double>> hist_;  // (t, s) of the reported estimate over the last stableBack s
  long nImu_ = 0;
  double tImu0_ = 0, tImu1_ = 0;
  std::vector<std::unique_ptr<Bank>> banks_;
  M3 Rcb_ = M3::Identity(), Rvb_ = M3::Identity();
  bool haveR_ = false, haveImu_ = false;
  double tRot_ = 0, tLast_ = 0, tMap0_ = -1;
  V3 prevA_ = V3::Zero(), prevW_ = V3::Zero();
  M12 Phi_ = M12::Identity(), QA_ = M12::Zero(), QO_ = M12::Zero();
  V12 u_ = V12::Zero();
};

}  // namespace ORB_SLAM3
