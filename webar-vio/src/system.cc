
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

  // === ORB path (early exit if selected) =====================================
  if (trackerType_ == TrackerType::ORB) {
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

      const float ratio = 0.75f;
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

    // Timing + exit
    const auto t_orb1 = std::chrono::high_resolution_clock::now();
    t_last_orb_ms_ = std::chrono::duration<double, std::milli>(t_orb1 - t_orb0).count();
    auto t_all1 = std::chrono::high_resolution_clock::now();
    t_last_total_ms_ = std::chrono::duration<double, std::milli>(t_all1 - t0).count();
    return; // skip KLT branch
  }
  // ==========================================================================

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

