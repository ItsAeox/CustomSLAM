
#include "system.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <numeric>  
#include <random>  
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>


// Forward-Backward KLT tracking with consistency check and basic gating.
static void trackFbKLT(const std::vector<cv::Mat>& pyrPrev,
                       const std::vector<cv::Mat>& pyrCur,
                       std::vector<cv::Point2f>& ptsPrev,
                       std::vector<cv::Point2f>& ptsCur,
                       std::vector<char>& alive,
                       int win=21, int levels=3,
                       float errMax=20.f, float fbMax=2.0f,
                       const cv::TermCriteria& termcrit=cv::TermCriteria(
                         cv::TermCriteria::COUNT|cv::TermCriteria::EPS,30,0.01))
{
  if (ptsPrev.empty()) return;
  int maxLv = levels;
  if ((int)pyrPrev.size() < 2*(levels+1)) maxLv = std::max(0, (int)pyrPrev.size()/2 - 1);

  std::vector<uchar> st; std::vector<float> err;
  cv::Size winSz(win,win);

  // forward
  ptsCur = ptsPrev;
  cv::calcOpticalFlowPyrLK(pyrPrev, pyrCur, ptsPrev, ptsCur, st, err, winSz, maxLv, termcrit,
                           cv::OPTFLOW_USE_INITIAL_FLOW | cv::OPTFLOW_LK_GET_MIN_EIGENVALS);

  std::vector<cv::Point2f> fwdCur; fwdCur.reserve(ptsPrev.size());
  std::vector<cv::Point2f> bwdPrev; bwdPrev.reserve(ptsPrev.size());
  std::vector<int>         idx;     idx.reserve(ptsPrev.size());
  alive.assign(ptsPrev.size(), 0);

  for (size_t i=0;i<ptsPrev.size();++i) {
    if (!st[i]) continue;
    if (err[i] > errMax) continue;
    const auto& p = ptsCur[i];
    if (p.x < 1 || p.y < 1) continue;
    if (p.x >= pyrCur[0].cols-1 || p.y >= pyrCur[0].rows-1) continue;
    fwdCur.push_back(ptsCur[i]);
    bwdPrev.push_back(ptsPrev[i]);
    idx.push_back((int)i);
    alive[i] = 1;
  }
  if (fwdCur.empty()) return;

  // backward
  st.clear(); err.clear();
  cv::calcOpticalFlowPyrLK(pyrCur, pyrPrev, fwdCur, bwdPrev, st, err, winSz, 0, termcrit,
                           cv::OPTFLOW_USE_INITIAL_FLOW | cv::OPTFLOW_LK_GET_MIN_EIGENVALS);

  for (size_t k=0;k<fwdCur.size();++k) {
    int i = idx[k];
    if (!st[k]) { alive[i]=0; continue; }
    if (cv::norm(ptsPrev[i] - bwdPrev[k]) > fbMax) { alive[i]=0; continue; }
  }
  // --- Photometric gate: normalized cross-correlation (NCC) on 9x9 patches ---
  // Reject tracks whose local appearance doesn't agree (helps kill jittery matches)
  {
    const int half = 4; // 9x9 patch
    // Use base pyramid level (same one LK errors are based on)
    const cv::Mat& Iprev = pyrPrev.empty() ? cv::Mat() : pyrPrev[0];
    const cv::Mat& Icur  = pyrCur .empty() ? cv::Mat() : pyrCur [0];
    if (!Iprev.empty() && !Icur.empty()) {
      for (size_t ii = 0; ii < ptsPrev.size(); ++ii) {
        if (!alive[ii]) continue;

        const cv::Point2f a = ptsPrev[ii];
        const cv::Point2f b = ptsCur [ii];

        if (a.x < half || a.y < half || b.x < half || b.y < half ||
            a.x >= Iprev.cols - half || a.y >= Iprev.rows - half ||
            b.x >= Icur .cols - half || b.y >= Icur .rows - half) {
          alive[ii] = 0; 
          continue;
        }

        cv::Mat pa, pb;
        cv::getRectSubPix(Iprev, cv::Size(2*half+1, 2*half+1), a, pa);
        cv::getRectSubPix(Icur,  cv::Size(2*half+1, 2*half+1), b, pb);

        cv::Mat nccMat;
        cv::matchTemplate(pa, pb, nccMat, cv::TM_CCOEFF_NORMED);
        const float ncc = nccMat.at<float>(0,0);

        if (ncc < 0.85f) {
          alive[ii] = 0; // too different; drop this track
        }
      }
    }
  }

}

// Keep a uniform subset of points: at most one per cell, up to 'maxKeep'
static void thinTracksUniform(std::vector<cv::Point2f>& pts,
                              int imgW, int imgH,
                              int cellSize, int maxKeep)
{
  if ((int)pts.size() <= maxKeep) return;

  const int CW = std::max(1, imgW / cellSize);
  const int CH = std::max(1, imgH / cellSize);
  std::vector<uint8_t> used(CW * CH, 0);

  std::vector<cv::Point2f> kept;
  kept.reserve(std::min((int)pts.size(), maxKeep));

  // greedily keep at most one per cell
  for (const auto& p : pts) {
    int c = std::clamp(int(p.x / cellSize), 0, CW - 1);
    int r = std::clamp(int(p.y / cellSize), 0, CH - 1);
    uint8_t& flag = used[r * CW + c];
    if (!flag) {
      kept.push_back(p);
      flag = 1;
      if ((int)kept.size() >= maxKeep) break;
    }
  }

  if (!kept.empty()) pts.swap(kept);
  // If we still have more than maxKeep 
  // truncate to maxKeep as a final guard:
  if ((int)pts.size() > maxKeep) pts.resize(maxKeep);
}


// Randomized cell order to avoid top-row bias.
static void detectGridShiTomasi(const cv::Mat& img,
                                const std::vector<cv::Point2f>& occupied,
                                int cellSize,
                                int wantTotal,
                                std::vector<cv::Point2f>& out)
{
  out.clear();
  if (img.empty() || wantTotal <= 0) return;
  const int W = img.cols, H = img.rows;
  const int CW = std::max(1, W / cellSize);
  const int CH = std::max(1, H / cellSize);

  // mark occupied cells
  std::vector<uint8_t> occ(CW*CH, 0);
  for (auto& p : occupied) {
    int c = std::clamp(int(p.x / cellSize), 0, CW-1);
    int r = std::clamp(int(p.y / cellSize), 0, CH-1);
    occ[r*CW + c] = 1;
  }

  // precompute score map once
  cv::Mat blur; cv::GaussianBlur(img, blur, cv::Size(3,3), 0.);
  cv::Mat hmap; cv::cornerMinEigenVal(blur, hmap, 3, 3);

  // randomized traversal over all cells
  std::vector<int> order(CW*CH);
  std::iota(order.begin(), order.end(), 0);
  static thread_local std::mt19937 rng{1234567};
  std::shuffle(order.begin(), order.end(), rng);

  out.reserve(std::min(wantTotal, CW*CH));
  for (int idx : order) {
    if ((int)out.size() >= wantTotal) break;
    if (occ[idx]) continue;
    int c = idx % CW, r = idx / CW;
    int x0 = c*cellSize, y0 = r*cellSize;
    int x1 = std::min(x0+cellSize, W), y1 = std::min(y0+cellSize, H);
    if (x1-x0 < 3 || y1-y0 < 3) continue;

    cv::Mat roi = hmap(cv::Rect(x0,y0,x1-x0,y1-y0));
    double minv, maxv; cv::Point maxp;
    cv::minMaxLoc(roi, &minv, &maxv, nullptr, &maxp);

    cv::Point2f p(maxp.x + x0, maxp.y + y0);
    std::vector<cv::Point2f> tmp{p};
    cv::cornerSubPix(img, tmp, cv::Size(3,3), cv::Size(-1,-1),
                     cv::TermCriteria(cv::TermCriteria::EPS|cv::TermCriteria::MAX_ITER, 30, 0.01));
    out.push_back(tmp[0]);
  }
}

// Keep spatially uniform points using a simple grid (bucket) selection.
static void bucketUniform(const std::vector<cv::Point2f>& ptsIn,
                          int W, int H, int cell, int maxKeep,
                          std::vector<int>& keepIdx) {
  const int cols = std::max(1, W / cell), rows = std::max(1, H / cell);
  std::vector<int> taken(cols * rows, 0);
  keepIdx.clear(); keepIdx.reserve(std::min((int)ptsIn.size(), maxKeep));
  for (int i = 0; i < (int)ptsIn.size(); ++i) {
    const auto& p = ptsIn[i];
    int cx = std::min(cols - 1, std::max(0, (int)p.x / cell));
    int cy = std::min(rows - 1, std::max(0, (int)p.y / cell));
    int bin = cy * cols + cx;
    if (!taken[bin]) {
      taken[bin] = 1;
      keepIdx.push_back(i);
      if ((int)keepIdx.size() >= maxKeep) break;
    }
  }
}

// Build a suppression mask so new detections avoid neighborhoods of existing points.
static void buildSuppressionMask(cv::Size sz,
                                 const std::vector<cv::Point2f>& keep,
                                 int radius, cv::Mat& mask) {
  mask = cv::Mat(sz, CV_8UC1, cv::Scalar(255));
  for (const auto& p : keep) {
    cv::circle(mask, cv::Point(cvRound(p.x), cvRound(p.y)), radius,
               cv::Scalar(0), -1, cv::LINE_AA);
  }
}

// Orthonormalize a 3x3 rotation in-place (Matx33d)
static inline void orthoMat(cv::Matx33d& Rm) {
  cv::Mat M(3,3,CV_64F);
  for (int r=0;r<3;++r) for (int c=0;c<3;++c) M.at<double>(r,c) = Rm(r,c);

  cv::SVD svd(M);
  cv::Mat U = svd.u, Vt = svd.vt;
  cv::Mat R = U * Vt;

  // Ensure det(R)=+1 (no reflection)
  if (cv::determinant(R) < 0) {
    U.col(2) *= -1;
    R = U * Vt;
  }

  for (int r=0;r<3;++r) for (int c=0;c<3;++c) Rm(r,c) = R.at<double>(r,c);
}

// === Mapping helpers ========================================================
// Compute ORB descriptors at given processing-scale points (no detection step).
void System::computeORBAtPoints(const cv::Mat& img,
                                const std::vector<cv::Point2f>& pts,
                                cv::Mat& outDesc)
{
  std::vector<cv::KeyPoint> kps; kps.reserve(pts.size());
  for (auto& p : pts) kps.emplace_back(p, (float)orbPatchSize_);
  outDesc.release();
  if (orb_) orb_->compute(img, kps, outDesc);
}

// Collect (Xw ↔ pixel) pairs by projection + small window gating.
// Returns number of correspondences harvested into pnp_points_/pnp_pixels_/pnp_indices_.
int System::harvestPnpCorrespondences(float winPx, int maxTake)
{
  pnp_points_.clear(); pnp_pixels_.clear(); pnp_indices_.clear();
  pnp_points_.reserve(std::min<int>((int)mps_.size(), maxTake));
  pnp_pixels_.reserve(std::min<int>((int)mps_.size(), maxTake));
  pnp_indices_.reserve(std::min<int>((int)mps_.size(), maxTake));

  // Current camera pose (world->camera): Rcw, tcw from Rwc_,twc_
  cv::Matx33d Rcw = Rwc_.t();
  cv::Vec3d   tcw = -(Rwc_.t() * twc_);
  const cv::Matx33d Kd = K();

  // Build a light grid over current 2D tracks to accelerate nearest search (processing scale).
  // Grid cell ~ 12 px at processing scale.
  const int Wp = curProc_.cols, Hp = curProc_.rows;
  const int cell = 12;
  const int gx = std::max(1, Wp / cell), gy = std::max(1, Hp / cell);
  std::vector<std::vector<int>> grid(gx*gy);
  grid.reserve(gx*gy);
  for (int i = 0; i < (int)ptsCur_.size(); ++i) {
    const auto& p = ptsCur_[i];
    int cx = std::clamp(int(p.x / cell), 0, gx-1);
    int cy = std::clamp(int(p.y / cell), 0, gy-1);
    grid[cy*gx + cx].push_back(i);
  }
  auto visitBucket = [&](int cx, int cy, auto&& fn){
    if (cx<0||cy<0||cx>=gx||cy>=gy) return;
    for (int idx : grid[cy*gx + cx]) fn(idx);
  };

  // Reprojection window defined at processing scale:
  const float s = (float)procScale_;
  const float winProc = std::max(2.f, winPx / s);
  const float win2 = winProc * winProc;

  int taken = 0;
  for (int mi = 0; mi < (int)mps_.size(); ++mi) {
    const auto& M = mps_[mi];

    // Project Xw
    cv::Vec3d Xc = Rcw * M.Xw + tcw;
    if (Xc[2] <= 1e-6) continue; // behind

    double u = (Kd(0,0) * (Xc[0]/Xc[2])) + Kd(0,2);
    double v = (Kd(1,1) * (Xc[1]/Xc[2])) + Kd(1,2);

    // Convert full-res pixel to processing scale
    float up = (float)(u / s);
    float vp = (float)(v / s);
    if (up < 2 || vp < 2 || up >= Wp-2 || vp >= Hp-2) continue;

    // Search a few neighbor buckets
    int cx = int(up / cell), cy = int(vp / cell);
    int hits = 0, bestIdx = -1; float bestD2 = win2;

    auto scan = [&](int idx){
      const auto& q = ptsCur_[idx];
      float dx = q.x - up, dy = q.y - vp;
      float d2 = dx*dx + dy*dy;
      if (d2 <= bestD2) { bestD2 = d2; bestIdx = idx; }
    };
    visitBucket(cx,cy,scan);
    visitBucket(cx+1,cy,scan); visitBucket(cx-1,cy,scan);
    visitBucket(cx,cy+1,scan); visitBucket(cx,cy-1,scan);

    // if (bestIdx >= 0) {
    //   pnp_indices_.push_back(mi);
    //   pnp_points_.emplace_back((float)M.Xw[0], (float)M.Xw[1], (float)M.Xw[2]);
    //   // use full-res pixels for PnP
    //   pnp_pixels_.emplace_back(up * s, vp * s);
    //   if (++taken >= maxTake) break;
    // }
    if (bestIdx >= 0) {
      // Optional descriptor verification (if the MapPoint has a descriptor)
      bool pass = true;
      if (!mps_[mi].desc.empty()) {
        // Compute ORB at the candidate 2D point (processing scale)
        std::vector<cv::Point2f> onePt = { ptsCur_[bestIdx] };
        cv::Mat candDesc; computeORBAtPoints(curProc_, onePt, candDesc);
        if (!candDesc.empty()) {
          // Hamming distance gate
          const int dist = cv::norm(candDesc.row(0), mps_[mi].desc, cv::NORM_HAMMING);
          // Typical robust range: 0..256 (ORB 256 bits). Try 50-60 first.
          if (dist > 60) pass = false;
        }
      }
      if (pass) {
        pnp_indices_.push_back(mi);
        pnp_points_.emplace_back((float)mps_[mi].Xw[0], (float)mps_[mi].Xw[1], (float)mps_[mi].Xw[2]);
        pnp_pixels_.emplace_back(up * s, vp * s);
        if (++taken >= maxTake) break;
      }
    }
  }
  return (int)pnp_points_.size();
}

// Two-view init: recover relative pose, triangulate inlier pairs, make 2 KFs + MapPoints.
bool System::tryTwoViewInit(const std::vector<cv::Point2f>& prevProcPts,
                            const std::vector<cv::Point2f>& curProcPts)
{
  if (mapInitialized_) return true;
  if (prevProcPts.size() < 12 || curProcPts.size() < 12) return false;

  // Promote to full-res pixels
  std::vector<cv::Point2f> p0, p1;
  toFullResPixels(prevProcPts, p0);
  toFullResPixels(curProcPts,  p1);

  // E + recoverPose
  cv::Mat mask;
  cv::Mat E = cv::findEssentialMat(p0, p1, fx_, cv::Point2d(cx_, cy_),
                                   cv::RANSAC, 0.999, 1.5, mask);
  if (E.empty()) return false;

  cv::Mat R, t;
  int ninl = cv::recoverPose(E, p0, p1, R, t, fx_, cv::Point2d(cx_, cy_), mask);
  if (ninl < 30) return false;

  // Build normalized points for triangulation
  std::vector<cv::Point2f> n0, n1; n0.reserve(ninl); n1.reserve(ninl);
  cv::Matx33d Ki_ = Ki();
  for (int i=0;i<(int)mask.rows;i++) if (mask.at<uchar>(i)) {
    cv::Vec3d x0(p0[i].x, p0[i].y, 1.0); x0 = Ki_ * x0;
    cv::Vec3d x1(p1[i].x, p1[i].y, 1.0); x1 = Ki_ * x1;
    n0.emplace_back((float)(x0[0]), (float)(x0[1]));
    n1.emplace_back((float)(x1[0]), (float)(x1[1]));
  }

  // Cameras: P0 = [I|0], P1 = [R|t] (camera-1 in camera-0 coords)
  cv::Matx34d P0 = cv::Matx34d::eye();
  cv::Matx34d P1;
  for (int r=0;r<3;++r) for (int c=0;c<3;++c) P1(r,c) = R.at<double>(r,c);
  P1(0,3) = t.at<double>(0); P1(1,3) = t.at<double>(1); P1(2,3) = t.at<double>(2);

  // Triangulate
  cv::Mat X4;
  cv::triangulatePoints(P0, P1, n0, n1, X4);

  // Compose first two KFs in world=cam0 coords
  Keyframe KF0; KF0.id = nextKFId_++; KF0.Rwc = cv::Matx33d::eye(); KF0.twc = cv::Vec3d(0,0,0);
  Keyframe KF1; KF1.id = nextKFId_++; 
  cv::Matx33d R10; for (int r=0;r<3;++r) for (int c=0;c<3;++c) R10(r,c) = R.at<double>(r,c);
  cv::Vec3d   t10(t.at<double>(0), t.at<double>(1), t.at<double>(2));
  KF1.Rwc = R10.t(); KF1.twc = -(R10.t()*t10);

  // Prepare ORB descs at inlier pixels for KF1 (processing scale)
  // Convert full-res inliers to processing points to compute descriptors quickly
  std::vector<cv::Point2f> inlProc1; inlProc1.reserve(n1.size());
  const float s = (float)procScale_;
  for (auto& px : p1) inlProc1.emplace_back(px.x / s, px.y / s);
  computeORBAtPoints(curProc_, inlProc1, KF1.desc);

  // Create MapPoints with cheirality + reprojection + baseline angle checks
  mps_.reserve(mps_.size() + X4.cols);
  int kept = 0;
  for (int i=0;i<X4.cols;++i) {
    double X = X4.at<double>(0,i), Y = X4.at<double>(1,i), Z = X4.at<double>(2,i), W = X4.at<double>(3,i);
    if (W <= 1e-9) continue;
    cv::Vec3d Xc0 = cv::Vec3d(X/W, Y/W, Z/W);
    if (Xc0[2] <= 1e-6) continue; // depth>0 in cam0

    // depth in cam1: R*Xc0 + t
    cv::Vec3d Xc1 = R10 * Xc0 + t10;
    if (Xc1[2] <= 1e-6) continue;

    // simple reprojection check (< 2.5 px) into both cams (full-res)
    auto reprojErr = [&](const cv::Vec3d& Xc, const cv::Point2f& px){
      double u = fx_ * (Xc[0]/Xc[2]) + cx_;
      double v = fy_ * (Xc[1]/Xc[2]) + cy_;
      double du = u - px.x, dv = v - px.y;
      return std::sqrt(du*du + dv*dv);
    };
    if (reprojErr(Xc0, p0[i]) > 2.5) continue;
    if (reprojErr(Xc1, p1[i]) > 2.5) continue;

    MapPoint M;
    M.Xw = Xc0; // world=cam0
    M.hostKF = KF0.id;
    if (!KF1.desc.empty() && i < KF1.desc.rows) {
      M.desc = KF1.desc.row(i).clone();
    }
    mps_.push_back(std::move(M));
    kept++;
  }

  kfs_.push_back(std::move(KF0));
  kfs_.push_back(std::move(KF1));
  mapInitialized_ = (kept >= 50);
  lastKFTs_ = lastTS_;
  lastKFInliers_ = kept;

  // Initialize global pose with KF1 (so trail keeps continuity)
  if (mapInitialized_) {
    Rwc_ = kfs_.back().Rwc;
    twc_ = kfs_.back().twc;
  }
  return mapInitialized_;
}

// PnP on harvested correspondences; refines pose; returns true on success.
bool System::trackWithPnP()
{
  const int N = harvestPnpCorrespondences(/*winPx=*/8.f, /*maxTake=*/800);
  if (N < 20) return false;

  // Build vectors cv::Mat-friendly
  cv::Mat rvec, tvec;
  // Start from predicted pose (cw), using rotation prior if available
  {
    // Predict world-from-camera by applying the cached delta once
    cv::Matx33d Rwc_pred = Rwc_ * R_delta_prior_;
    cv::Matx33d Rcw_init = Rwc_pred.t();
    cv::Rodrigues(Rcw_init, rvec);
  
    // Translation seed consistent with predicted rotation
    cv::Vec3d tcw_init = -(Rwc_pred.t() * twc_);
    tvec = (cv::Mat_<double>(3,1) << tcw_init[0], tcw_init[1], tcw_init[2]);
  
    // Consume the prior (one-shot)
    R_delta_prior_ = cv::Matx33d::eye();
  }
  

  cv::Mat inliers;
  const cv::Mat Kcv = (cv::Mat_<double>(3,3) << fx_,0,cx_, 0,fy_,cy_, 0,0,1);
  bool ok = cv::solvePnPRansac(
              pnp_points_, pnp_pixels_, Kcv, cv::noArray(),
              rvec, tvec, /*useExtrinsicGuess=*/true,
              200, 2.0, 0.99, inliers, cv::SOLVEPNP_EPNP);
  if (!ok || inliers.empty() || inliers.rows < 20) return false;

  // === Update MP stats and gently prune weak correspondences ===
  std::vector<char> isInl(mps_.size(), 0);
  for (int i = 0; i < inliers.rows; ++i) {
    int mi = pnp_indices_[inliers.at<int>(i)];
    if (mi >= 0 && mi < (int)mps_.size()) isInl[mi] = 1;
  }
  for (size_t i = 0; i < pnp_indices_.size(); ++i) {
    int mi = pnp_indices_[i];
    if (mi >= 0 && mi < (int)mps_.size()) {
      mps_[mi].seen++;
      if (isInl[mi]) mps_[mi].found++;
    }
  }
  // Cull a small number per frame to avoid bursts
  int removed = 0;
  for (size_t i = 0; i < mps_.size() && removed < 64; ) {
    const auto& M = mps_[i];
    if (M.seen >= 10 && (double)M.found / std::max(1, M.seen) < 0.25) {
      mps_.erase(mps_.begin() + i);
      ++removed;
    } else {
      ++i;
    }
  }

  // Motion-only refine (optional LM)
  cv::solvePnP(pnp_points_, pnp_pixels_, Kcv, cv::noArray(),
               rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);

  // Update Twc
  cv::Mat Rcv; cv::Rodrigues(rvec, Rcv);
  cv::Matx33d Rcw;
  for (int r=0;r<3;++r) for (int c=0;c<3;++c) Rcw(r,c) = Rcv.at<double>(r,c);
  cv::Vec3d tcw(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
  Rwc_ = Rcw.t();
  twc_ = -(Rcw.t() * tcw);
  orthoMat(Rwc_);  

  // Append trail position (keeps your on-screen path in sync)
  path_.emplace_back((float)twc_[0], (float)twc_[1], (float)twc_[2]);
  if (path_.size() > 4096) {
    path_.erase(path_.begin(), path_.begin() + (path_.size() - 4096));
  }

  lastKFInliers_ = inliers.rows;
  return true;
}

// KF insertion heuristic: parallax/time/inliers based
bool System::shouldInsertKF(int pnpInliers, double nowTs) const
{
  if (!mapInitialized_) return false;
  if (kfs_.empty()) return true;
  if (pnpInliers < 80) return true;                    // tracking thinning
  if ((nowTs - lastKFTs_) > 1.0) return true;          // time-based
  if (pnpInliers < (lastKFInliers_ * 7) / 10) return true; // drop vs last KF
  return false;
}

// Insert KF for current pose and triangulate new points vs last KF
void System::insertKeyframeAndTriangulate()
{
  if (!mapInitialized_) return;

  // Build KF from current frame; compute dense ORB over full image (cheap at proc scale)
  Keyframe KF; KF.id = nextKFId_++; KF.Rwc = Rwc_; KF.twc = twc_;
  // We can reuse orb_ detector (already configured).
  std::vector<cv::KeyPoint> kps; cv::Mat desc;
  orb_->detectAndCompute(curProc_, cv::noArray(), kps, desc);
  KF.kps.swap(kps); KF.desc = desc;

  // Triangulate vs previous KF (simple 2-KF baseline)
  if (!kfs_.empty()) {
    const Keyframe& Kprev = kfs_.back();

    // match KFprev.desc ↔ KF.desc (ratio + mutual)
    cv::BFMatcher bf(cv::NORM_HAMMING, false);
    std::vector<std::vector<cv::DMatch>> knn01, knn10;
    bf.knnMatch(Kprev.desc, KF.desc, knn01, 2);
    bf.knnMatch(KF.desc,   Kprev.desc, knn10, 2);
    const float ratio=0.75f;
    std::vector<cv::DMatch> cands;
    for (auto& ks:knn01) if (ks.size()>=2 && ks[0].distance < ratio*ks[1].distance) cands.push_back(ks[0]);
    std::vector<char> ok(cands.size(),0);
    for (size_t i=0;i<cands.size();++i){
      auto m=cands[i];
      const auto& rv=knn10[m.trainIdx];
      if (rv.size()<2) continue;
      if (rv[0].distance >= ratio*rv[1].distance) continue;
      if (rv[0].trainIdx == m.queryIdx) ok[i]=1;
    }

    // normalized points for triangulation
    std::vector<cv::Point2f> n0, n1; n0.reserve(ok.size()); n1.reserve(ok.size());
    for (size_t i=0;i<cands.size();++i) if (ok[i]) {
      auto a = Kprev.kps[cands[i].queryIdx].pt;
      auto b = KF.kps   [cands[i].trainIdx].pt;
      // back to full-res pixels
      cv::Vec3d x0(a.x*procScale_, a.y*procScale_, 1.0); x0 = Ki()*x0;
      cv::Vec3d x1(b.x*procScale_, b.y*procScale_, 1.0); x1 = Ki()*x1;
      n0.emplace_back((float)x0[0], (float)x0[1]);
      n1.emplace_back((float)x1[0], (float)x1[1]);
    }

    if (n0.size() >= 20) {
      // P0=[Rcw0|tcw0], P1=[Rcw1|tcw1] but triangulation expects cam1 in cam0:
      cv::Matx33d Rcw0 = Kprev.Rwc.t(), Rcw1 = KF.Rwc.t();
      cv::Vec3d   tcw0 = -(Kprev.Rwc.t()*Kprev.twc);
      cv::Vec3d   tcw1 = -(KF.Rwc.t()*KF.twc);
      cv::Matx34d P0, P1;
      for (int r=0;r<3;++r) for (int c=0;c<3;++c){ P0(r,c)=Rcw0(r,c); P1(r,c)=Rcw1(r,c); }
      P0(0,3)=tcw0[0]; P0(1,3)=tcw0[1]; P0(2,3)=tcw0[2];
      P1(0,3)=tcw1[0]; P1(1,3)=tcw1[1]; P1(2,3)=tcw1[2];

      cv::Mat X4; cv::triangulatePoints(P0,P1,n0,n1,X4);
      const double cosMax = std::cos(70.0 * M_PI/180.0); // viewing angle gate

      for (int i=0;i<X4.cols;++i){
        double X=X4.at<double>(0,i), Y=X4.at<double>(1,i), Z=X4.at<double>(2,i), W=X4.at<double>(3,i);
        if (W<=1e-9) continue;
        cv::Vec3d Xw(X/W, Y/W, Z/W);

        // Cheirality
        cv::Vec3d Xc0 = Rcw0*Xw + tcw0;
        cv::Vec3d Xc1 = Rcw1*Xw + tcw1;
        if (Xc0[2]<=1e-6 || Xc1[2]<=1e-6) continue;

        // Baseline angle (cosine of angle between rays)
        cv::Vec3d v0 = Xc0 / cv::norm(Xc0), v1 = Xc1 / cv::norm(Xc1);
        double cosang = v0.dot(v1);
        if (cosang > cosMax) continue; // too small angle

        MapPoint M; M.Xw = Xw; M.hostKF = KF.id;
        if (!KF.desc.empty() && i < KF.desc.rows) M.desc = KF.desc.row(i).clone();
        mps_.push_back(std::move(M));
      }
    }
  }

  kfs_.push_back(std::move(KF));
  lastKFTs_ = lastTS_;
}

// Convert processing-scale points -> full-res pixel coordinates
void System::toFullResPixels(const std::vector<cv::Point2f>& procPts,
                             std::vector<cv::Point2f>& fullResPx) const {
  fullResPx.resize(procPts.size());
  const float s = static_cast<float>(procScale_);
  for (size_t i = 0; i < procPts.size(); ++i) {
    fullResPx[i].x = procPts[i].x * s;
    fullResPx[i].y = procPts[i].y * s;
  }
}

// Run E vs H RANSAC on matched pairs; store result in eh* fields
void System::runEvsHGate(const std::vector<cv::Point2f>& prevProcPts,
                         const std::vector<cv::Point2f>& curProcPts) {
  ehModel_ = 0; ehInliersE_ = ehInliersH_ = 0; ehParallaxDeg_ = 0.0;
  if (prevProcPts.size() < 8 || curProcPts.size() < 8) return;

  // 1) Promote to full-res pixel coordinates (your intrinsics are full-res)
  std::vector<cv::Point2f> p0, p1;
  toFullResPixels(prevProcPts, p0);
  toFullResPixels(curProcPts,  p1);

  // 2) Median pixel displacement -> rough parallax (deg)
  {
    std::vector<double> disp; disp.reserve(p0.size());
    for (size_t i = 0; i < p0.size(); ++i) {
      disp.push_back(cv::norm(p1[i] - p0[i]));
    }
    if (!disp.empty()) {
      std::nth_element(disp.begin(), disp.begin()+disp.size()/2, disp.end());
      const double medPx = disp[disp.size()/2];
      // small-angle approx: angle ≈ atan(medPx / fx_)
      const double ang = std::atan2(medPx, std::max(1e-6, fx_)) * 180.0 / M_PI;
      ehParallaxDeg_ = ang;
    }
  }

  // 3) RANSAC for Essential (use pixel points + focal,pp)
  int inlE = 0, inlH = 0;
  {
    cv::Mat maskE;
    // Keep E so we can decompose it
    cv::Mat E = cv::findEssentialMat(p0, p1, fx_, cv::Point2d(cx_, cy_),
                                     cv::RANSAC, 0.999, 1.5, maskE);
    if (!E.empty()) {
      for (int i = 0; i < maskE.rows; ++i) inlE += (maskE.at<uchar>(i) ? 1 : 0);
  
      // === NEW: cache rotation prior when E is selected/viable ===
      if (inlE >= 8) {
        cv::Mat R, t;
        // We can pass E to recoverPose to get a consistent R
        int ninl = cv::recoverPose(E, p0, p1, R, t, fx_, cv::Point2d(cx_, cy_), maskE);
        if (ninl >= 8) {
          cv::Matx33d R10;
          for (int r=0; r<3; ++r) for (int c=0; c<3; ++c) R10(r,c) = R.at<double>(r,c);
          // World-from-camera delta for Twc update is R10^T
          R_delta_prior_ = R10.t();
        }
      }
    }
  }
  
  // 4) RANSAC for Homography (pixel domain)
  {
    cv::Mat maskH;
    cv::findHomography(p0, p1, cv::RANSAC, 1.5, maskH, 2000, 0.999);
    if (!maskH.empty()) {
      for (int i = 0; i < maskH.rows; ++i) inlH += maskH.at<uchar>(i) ? 1 : 0;
    }
  }

  ehInliersE_ = inlE;
  ehInliersH_ = inlH;

  // 5) Simple model selection heuristic
  // Prefer E when parallax is present and inliers are comparable; else H.
  const bool hasParallax = (ehParallaxDeg_ >= 1.5);
  bool preferE = false;
  if (inlE >= inlH + 15) preferE = true;
  else if (inlE >= (int)std::round(0.7 * inlH) && hasParallax) preferE = true;

  ehModel_ = preferE ? 1 : 2; // 1=E, 2=H
}

// Integrate VO using E decomposition (monocular; scale arbitrary)
void System::integrateVO_E(const std::vector<cv::Point2f>& prevProcPts,
                           const std::vector<cv::Point2f>& curProcPts)
{
  if (prevProcPts.size() < 8 || curProcPts.size() < 8) return;

  // Promote to full-res pixel coords (intrinsics are full-res)
  std::vector<cv::Point2f> p0, p1;
  toFullResPixels(prevProcPts, p0);
  toFullResPixels(curProcPts,  p1);

  // Find E and recover relative pose (R_10, t_10) from cam0->cam1
  cv::Mat inlierMaskE;
  cv::Mat E = cv::findEssentialMat(p0, p1, fx_, cv::Point2d(cx_, cy_),
                                   cv::RANSAC, 0.999, 1.5, inlierMaskE);
  if (E.empty()) return;

  cv::Mat R, t;
  int ninl = cv::recoverPose(E, p0, p1, R, t, fx_, cv::Point2d(cx_, cy_), inlierMaskE);
  if (ninl < 8) return;

  // Convert to double-friendly formats
  cv::Matx33d R10; cv::Vec3d t10;
  for (int r=0;r<3;++r) for (int c=0;c<3;++c) R10(r,c) = R.at<double>(r,c);
  t10[0] = t.at<double>(0); t10[1] = t.at<double>(1); t10[2] = t.at<double>(2);

  // Monocular: t10 is up to scale. Use a soft scale heuristic from pixel motion
  // to keep the trail legible (not physically accurate, just for visualization).
  // Median pixel displacement (already similar to your parallax calc):
  std::vector<double> disp; disp.reserve(p0.size());
  for (size_t i=0;i<p0.size();++i) disp.push_back(cv::norm(p1[i]-p0[i]));
  std::nth_element(disp.begin(), disp.begin()+disp.size()/2, disp.end());
  const double medPx = disp[disp.size()/2];
  // Scale factor: s ~ medPx / fx_ * k ; choose k≈1.5 to look nice on-screen
  const double s = std::clamp(1.5 * (medPx / std::max(1e-6, fx_)), 0.0, 0.2);
  t10 *= s;

  // Compose world pose. If Twc0 = [Rwc|twc], and cam1 = R10,t10 in cam0 frame:
  // Twc1 = Twc0 * inv(Tc1c0) = Twc0 * [R10^T | -R10^T t10]
  cv::Matx33d R_next = Rwc_ * R10.t();
  cv::Vec3d   t_next = twc_ - Rwc_ * (R10.t() * t10);

  Rwc_ = R_next;
  twc_ = t_next;
  orthoMat(Rwc_);  

  // Record position
  path_.emplace_back((float)twc_[0], (float)twc_[1], (float)twc_[2]);
  if (path_.size() > 4096) {
    path_.erase(path_.begin(), path_.begin() + (path_.size() - 4096));
  }
}

// Integrate rotation from Homography when E is not selected (pure rotation / planar).
void System::integrateVO_H(const std::vector<cv::Point2f>& prevProcPts,
                           const std::vector<cv::Point2f>& curProcPts)
{
  if (prevProcPts.size() < 8 || curProcPts.size() < 8) return;

  // Work in full-res pixel coords (your intrinsics are full-res)
  std::vector<cv::Point2f> p0, p1;
  toFullResPixels(prevProcPts, p0);
  toFullResPixels(curProcPts,  p1);

  // Robust pixel-domain homography
  cv::Mat inl;
  cv::Mat Hpix = cv::findHomography(p0, p1, cv::RANSAC, 1.5, inl, 2000, 0.999);
  if (Hpix.empty()) return;

  // Decompose H to get rotation(s)
  cv::Mat K = (cv::Mat_<double>(3,3) << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1);
  std::vector<cv::Mat> Rs, Ts, Ns;
  int n = cv::decomposeHomographyMat(Hpix, K, Rs, Ts, Ns);
  if (n <= 0) return;

  // Data-driven pick: minimize average angular error between R*a and b over normalized rays
  std::vector<cv::Point3d> b0, b1;
  b0.reserve(p0.size()); b1.reserve(p1.size());
  cv::Matx33d Kinv = Ki();
  for (size_t i = 0; i < p0.size(); ++i) {
    cv::Vec3d a = Kinv * cv::Vec3d(p0[i].x, p0[i].y, 1.0);
    cv::Vec3d b = Kinv * cv::Vec3d(p1[i].x, p1[i].y, 1.0);
    a /= cv::norm(a); b /= cv::norm(b);
    b0.emplace_back(a[0], a[1], a[2]);
    b1.emplace_back(b[0], b[1], b[2]);
  }

  int best = -1;
  double bestErr = 1e18;
  for (int i = 0; i < n; ++i) {
    if (cv::determinant(Rs[i]) <= 0) continue;
    cv::Matx33d Ri;
    for (int r=0;r<3;++r) for (int c=0;c<3;++c) Ri(r,c) = Rs[i].at<double>(r,c);

    double sumAng = 0.0; int cnt = 0;
    for (size_t k = 0; k < b0.size(); ++k) {
      cv::Vec3d ra = Ri * cv::Vec3d(b0[k].x, b0[k].y, b0[k].z);
      double dot = std::max(-1.0, std::min(1.0, ra.dot(cv::Vec3d(b1[k].x, b1[k].y, b1[k].z))));
      sumAng += std::acos(dot);
      ++cnt;
    }
    const double avg = (cnt ? sumAng / cnt : 1e9);
    if (avg < bestErr) { bestErr = avg; best = i; }
  }
  if (best < 0) return;

  // Update world pose: Twc1 = Twc0 * inv(Rc1c0) = Twc0 * R^T (ignore t for VO_H)
  cv::Matx33d R;
  for (int r=0;r<3;++r) for (int c=0;c<3;++c) R(r,c) = Rs[best].at<double>(r,c);
  Rwc_ = Rwc_ * R.t();
  orthoMat(Rwc_);


  // Keep trail coherent (no translation added)
  path_.emplace_back((float)twc_[0], (float)twc_[1], (float)twc_[2]);
  if (path_.size() > 4096) {
    path_.erase(path_.begin(), path_.begin() + (path_.size() - 4096));
  }
}

// Return flattened [x0,z0, x1,z1, ...]
std::vector<float> System::getPathXZ() const {
  std::vector<float> flat;
  flat.reserve(path_.size() * 2);
  for (const auto& P : path_) {
    flat.push_back(P.x);
    flat.push_back(P.z);
  }
  return flat;
}

std::array<double,3> System::getYPR() const {
  // Columns of Rwc_ are camera axes expressed in world coords
  const double r02 = Rwc_(0,2), r12 = Rwc_(1,2), r22 = Rwc_(2,2); // forward (col 2)
  const double r01 = Rwc_(0,1), r11 = Rwc_(1,1);                  // up (col 1)

  // Yaw: heading of forward projected on XZ (OpenCV y is "down")
  double yaw   = std::atan2(r02, r22);

  // Pitch: elevation of forward (negate y component)
  double pitch = std::atan2(-r12, std::sqrt(r02*r02 + r22*r22));

  // Roll: rotation around forward; use up vs right around vertical-ish axis
  double roll  = std::atan2(r01, r11);

  return { yaw, pitch, roll };
}


System::System() {}

void System::init(int width, int height, double fx, double fy, double cx, double cy) {
  imgW_ = width; imgH_ = height; fx_ = fx; fy_ = fy; cx_ = cx; cy_ = cy;

  const int Wp = std::max(1, imgW_ / procScale_);
  const int Hp = std::max(1, imgH_ / procScale_);
  prevGray_.release(); curGray_.release();
  prevProc_.create(Hp, Wp, CV_8UC1);
  curProc_.create(Hp, Wp, CV_8UC1);
  pyrPrev_.reserve(kltLevels_ + 1);
  pyrCur_.reserve(kltLevels_ + 1);
  ptsPrev_.reserve(targetKps_ * 2);
  ptsCur_.reserve(targetKps_ * 2);
  pyrPrev_.clear(); pyrCur_.clear();
  ptsPrev_.clear(); ptsCur_.clear();

  // Initialize ORB extractor (used by ORB tracker and optional descriptors)
  if (!orb_) {
    orb_ = cv::ORB::create(
      orbNFeatures_,      // e.g., 600
      orbScaleFactor_,    // 1.2f
      orbNLevels_,        // 4
      orbEdgeThreshold_,  // 31
      orbFirstLevel_,     // 0
      orbWtaK_,           // 2
      orbScore_,          // cv::ORB::HARRIS_SCORE
      orbPatchSize_,      // 31
      orbFastThreshold_   // 20
    );
  }  

  trackingState_ = 0;
  frameCount_ = 0;
  lastTS_ = 0.0;
  hybFrameIdx_ = 0;
}

void System::feedFrame(const uint8_t* img, double ts, int width, int height, bool isRGBA) {
  auto t0 = std::chrono::high_resolution_clock::now();
  lastTS_ = ts;

  // 1) to gray (full-res)
  if (isRGBA) {
    cv::Mat rgba(imgH_, imgW_, CV_8UC4, const_cast<uint8_t*>(img));
    cv::cvtColor(rgba, curGray_, cv::COLOR_RGBA2GRAY);
  } else {
    curGray_ = cv::Mat(imgH_, imgW_, CV_8UC1, const_cast<uint8_t*>(img));
  }

  // 2) downscale to processing size
  const int Wp = std::max(1, imgW_ / procScale_);
  const int Hp = std::max(1, imgH_ / procScale_);
  cv::resize(curGray_, curProc_, cv::Size(Wp, Hp), 0, 0, cv::INTER_AREA);
  lastMeanY_ = cv::mean(curProc_)[0];
  ranOrbThisFrame_ = false;
  hybFrameIdx_++;
    const bool isKeyframe = (hybFrameIdx_ % std::max(1, hybridEveryN_)) == 0;
    if (isKeyframe) {
      const auto t_orb0 = std::chrono::high_resolution_clock::now();
      // 1) Detect+compute on current processing-scale frame
      orbCurKps_.clear();
      orbCurDesc_.release();
      orb_->detectAndCompute(curProc_, cv::noArray(), orbCurKps_, orbCurDesc_);
  
      ptsPrev_.clear();
      ptsCur_.clear();
  
      std::vector<cv::Point2f> curPtsAll; curPtsAll.reserve(orbCurKps_.size());
      for (auto& k : orbCurKps_) curPtsAll.push_back(k.pt);
  
      std::vector<cv::Point2f> keepPrev, keepCur;
  
      if (!orbPrevDesc_.empty() && !orbCurDesc_.empty()) {
        // 2) Ratio + mutual (symmetric) matching to stabilize correspondences
        cv::BFMatcher bf(cv::NORM_HAMMING, /*crossCheck=*/false);
  
        std::vector<std::vector<cv::DMatch>> knnPC, knnCP;
        bf.knnMatch(orbPrevDesc_, orbCurDesc_, knnPC, 2);
        bf.knnMatch(orbCurDesc_, orbPrevDesc_, knnCP, 2);
  
        const float ratio = 0.7f;
        std::vector<cv::DMatch> candPC;
        candPC.reserve(knnPC.size());
        for (const auto& ks : knnPC) {
          if (ks.size() < 2) continue;
          if (ks[0].distance < ratio * ks[1].distance) candPC.push_back(ks[0]);
        }
  
        // mutual check
        std::vector<char> ok(candPC.size(), 0);
        for (size_t i = 0; i < candPC.size(); ++i) {
          const auto& m = candPC[i];
          // find best in CP for m.trainIdx
          const auto& rev = knnCP[m.trainIdx];
          if (rev.size() < 2) continue;
          if (rev[0].distance >= ratio * rev[1].distance) continue;
          if (rev[0].trainIdx == m.queryIdx) ok[i] = 1;
        }
  
        std::vector<cv::Point2f> pPrev, pCur;
        std::vector<int> idxPrev, idxCur;
        for (size_t i = 0; i < candPC.size(); ++i) if (ok[i]) {
          const auto& m = candPC[i];
          pPrev.push_back(orbPrevKps_[m.queryIdx].pt);
          pCur .push_back(orbCurKps_[m.trainIdx].pt);
          idxPrev.push_back(m.queryIdx);
          idxCur .push_back(m.trainIdx);
        }
  
        // 3) Geometric gating (RANSAC). Use Fundamental matrix (no intrinsics needed).
        cv::Mat inlierMask;
        if (pPrev.size() >= 8) {
          // ransacReprojThreshold = 1.5 px, confidence = 0.99
          (void)cv::findFundamentalMat(pPrev, pCur, cv::FM_RANSAC, 1.5, 0.99, inlierMask);
        } else {
          inlierMask = cv::Mat::ones((int)pPrev.size(), 1, CV_8U);
        }
        
        for (int i = 0; i < inlierMask.rows; ++i) {
          if (inlierMask.at<uchar>(i)) {
            keepPrev.push_back(pPrev[(size_t)i]);
            keepCur .push_back(pCur [(size_t)i]);
          }
        }      
      }
      // --- E/H gate on ORB correspondences (post-F check) ---
      if (keepPrev.size() >= 8 && keepCur.size() >= 8) {
        runEvsHGate(keepPrev, keepCur);
        if (!mapInitialized_ && ehModel_ == 1) {
          integrateVO_E(keepPrev, keepCur);
        } else if (ehModel_ == 2) {
          integrateVO_H(keepPrev, keepCur);   // in the ORB path
        }
      }  

      // Two-view init trigger (only once)
      if (!mapInitialized_ && ehModel_ == 1) {
        (void)tryTwoViewInit(keepPrev, keepCur);
      }

  
      // 4) Top-up: if too few tracked points, detect new ones away from current tracks
      const int target = std::min(maxTracks_, 800);   // cap
      const int minTracked = std::min(target, std::max(10, target * 6 / 10)); // keep ~60% tracked
      if ((int)keepCur.size() < minTracked) {
        cv::Mat mask;
        buildSuppressionMask(curProc_.size(), keepCur, /*radius px=*/12, mask);
  
        // Detect more where we have no points; compute descriptors for those
        std::vector<cv::KeyPoint> addKps;
        cv::Mat addDesc;
        orb_->detectAndCompute(curProc_, mask, addKps, addDesc);
  
        // Append until we reach target
        for (int i = 0; i < (int)addKps.size() && (int)keepCur.size() < target; ++i) {
          keepCur.push_back(addKps[i].pt);
          // no need to maintain keepPrev for new points (they are new births)
        }
      }
  
      // 5) Enforce spatial spread (bucket/grid). Works on keepCur only.
      std::vector<int> uniformIdx;
      bucketUniform(keepCur, curProc_.cols, curProc_.rows, /*cell=*/cellSize_, /*maxKeep=*/target, uniformIdx);
  
      ptsCur_.reserve(uniformIdx.size());
      for (int j : uniformIdx) ptsCur_.push_back(keepCur[j]);
  
      // Tracking state
      trackingState_ = (ptsCur_.size() >= 10) ? 1 : 0;
  
      // 6) Prepare "prev" for next frame: set prev = current (re-compute descriptors to align sets)
      //    Recompute descriptors at the positions we actually kept, so next matching is clean.
      {
        std::vector<cv::KeyPoint> kpForPrev; kpForPrev.reserve(ptsCur_.size());
        for (auto& p : ptsCur_) kpForPrev.emplace_back(cv::Point2f(p.x, p.y), /*size=*/orbPatchSize_);
        orbPrevKps_.swap(kpForPrev);
        orb_->compute(curProc_, orbPrevKps_, orbPrevDesc_); // descriptors for next frame
      }
  
      // Also keep a copy of the current image for any downstream assumptions
      if (prevProc_.size() != curProc_.size()) prevProc_.create(curProc_.rows, curProc_.cols, CV_8UC1);
      curProc_.copyTo(prevProc_);
      ptsPrev_ = ptsCur_;  // hand off these ORB points to KLT for the next frame

      pyrPrev_.clear();
      int maxLevel = std::max(0, kltLevels_);
      cv::buildOpticalFlowPyramid(prevProc_, pyrPrev_, cv::Size(kltWin_, kltWin_), maxLevel);

      const auto t_orb1 = std::chrono::high_resolution_clock::now();
      t_last_orb_ms_ = std::chrono::duration<double,std::milli>(t_orb1 - t_orb0).count();

      ranOrbThisFrame_ = true;
      lastOrbKF_ = hybFrameIdx_;
      orbKFCount_++;
  
      // We ran ORB this frame; skip KLT below.
      auto t_all1 = std::chrono::high_resolution_clock::now();
      t_last_total_ms_ = std::chrono::duration<double, std::milli>(t_all1 - t0).count();
      return;
    }
  

  // 3) build pyramids (processing scale)
  pyrCur_.clear();
  int maxLevel = std::max(0, kltLevels_);
  cv::buildOpticalFlowPyramid(curProc_, pyrCur_, cv::Size(kltWin_, kltWin_), maxLevel);

  // First frame => seed
  if (trackingState_ == 0 || prevProc_.empty() || ptsPrev_.empty()) {
    auto ts0 = std::chrono::high_resolution_clock::now();
    ptsPrev_.clear();
    detectGridShiTomasi(curProc_, /*occupied=*/{}, cellSize_, targetKps_, ptsPrev_);
    auto ts1 = std::chrono::high_resolution_clock::now();
    t_last_seed_ms_ = std::chrono::duration<double, std::milli>(ts1 - ts0).count();

    // Remove hidden per-frame allocations and reuse old
    if (prevProc_.size() != curProc_.size()) prevProc_.create(curProc_.rows, curProc_.cols, CV_8UC1);
    curProc_.copyTo(prevProc_);
    pyrPrev_  = pyrCur_;
    trackingState_ = 1;
    ptsCur_ = ptsPrev_;

    auto t1 = std::chrono::high_resolution_clock::now();
    t_last_klt_ms_ = 0.0;
    t_last_total_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return;
  }

  // 4) KLT track with forward-backward check
  auto tk0 = std::chrono::high_resolution_clock::now();
  std::vector<char> alive;
  ptsCur_.resize(ptsPrev_.size());
  trackFbKLT(pyrPrev_, pyrCur_, ptsPrev_, ptsCur_, alive, kltWin_, kltLevels_,
            kltErrMax_, fbMax_, termcrit_);
  auto tk1 = std::chrono::high_resolution_clock::now();
  t_last_klt_ms_ = std::chrono::duration<double, std::milli>(tk1 - tk0).count();

  // --- E/H gate on alive KLT tracks (use pairs before compaction) ---
  {
    std::vector<cv::Point2f> p0, p1;
    p0.reserve(ptsPrev_.size()); p1.reserve(ptsPrev_.size());
    for (size_t i = 0; i < ptsPrev_.size(); ++i) {
      if (i < (size_t)alive.size() && alive[i]) { p0.push_back(ptsPrev_[i]); p1.push_back(ptsCur_[i]); }
    }
    if (p0.size() >= 8) {
      runEvsHGate(p0, p1);
      if (!mapInitialized_ && ehModel_ == 1) {
        integrateVO_E(p0, p1);
      } else if (ehModel_ == 2) {
        integrateVO_H(p0, p1);              // in the KLT path
      }      
    }

    if (!mapInitialized_ && ehModel_ == 1) {
      (void)tryTwoViewInit(p0, p1);
    }  
  }

  // compact surviving tracks
  auto ts0 = std::chrono::high_resolution_clock::now();
  size_t m = 0;
  for (size_t i=0;i<ptsCur_.size();++i) if (alive[i]) ptsCur_[m++] = ptsCur_[i];
  ptsCur_.resize(m);

  // 5) reseed to maintain budget
  if ((int)ptsCur_.size() < targetKps_) {
    std::vector<cv::Point2f> newPts;
    detectGridShiTomasi(curProc_, ptsCur_, cellSize_, targetKps_ - (int)ptsCur_.size(), newPts);
    ptsCur_.insert(ptsCur_.end(), newPts.begin(), newPts.end());
  }

  thinTracksUniform(ptsCur_, curProc_.cols, curProc_.rows, cellSize_, maxTracks_);
  auto ts1 = std::chrono::high_resolution_clock::now();
  t_last_seed_ms_ = std::chrono::duration<double, std::milli>(ts1 - ts0).count();

  // OPT-> sparse ORB descriptors (for future map assoc); disabled if descEveryN_==0
  if (descEveryN_ > 0 && orb_ && (++frameCount_ % descEveryN_) == 0) {
    std::vector<cv::KeyPoint> kps; cv::KeyPoint::convert(ptsCur_, kps);
    cv::Mat descs; orb_->compute(curProc_, kps, descs);
    // store or expose descs if/when mapping is added
  }

  // ===== Mapping track (PnP) + KF insertion =====
  if (mapInitialized_) {
    const bool pnpOk = trackWithPnP();
    if (pnpOk && shouldInsertKF(lastKFInliers_, lastTS_)) {
      insertKeyframeAndTriangulate();
    }
  }  

  // 6) roll to next frame
  // Remove hidden per-frame allocations and reuse old
  if (prevProc_.size() != curProc_.size()) prevProc_.create(curProc_.rows, curProc_.cols, CV_8UC1);
  curProc_.copyTo(prevProc_);
  pyrPrev_  = pyrCur_;
  ptsPrev_  = ptsCur_;
  trackingState_ = 1;

  auto t1 = std::chrono::high_resolution_clock::now();
  t_last_total_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

std::vector<double> System::getPoints2D() const {
  // return FULL-RES coordinates (scale back from processing scale)
  const float s = static_cast<float>(procScale_);
  const size_t cap = std::min(ptsCur_.size(), static_cast<size_t>(maxReturnPts_));

  std::vector<double> flat;
  flat.reserve(cap * 2);

  // If we have more points than we want to return, just take the first 'cap'.
  for (size_t i = 0; i < cap; ++i) {
    const auto& p = ptsCur_[i];
    flat.push_back(p.x * s);
    flat.push_back(p.y * s);
  }
  return flat;
}

