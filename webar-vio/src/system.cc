#include "system.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <algorithm>

System::System() {}

// --- helpers ---
static inline cv::Rect cellRect(int r,int c,int W,int H,int R,int C){
  int x0 = (c*W)/C, x1 = ((c+1)*W)/C;
  int y0 = (r*H)/R, y1 = ((r+1)*H)/R;
  return cv::Rect(x0,y0, x1-x0, y1-y0);
}

static void detectGridGFTT(const cv::Mat& img, std::vector<cv::Point2f>& out,
                           int gridRows,int gridCols,int maxPerCell)
{
  out.clear(); out.reserve(gridRows*gridCols*maxPerCell);
  const double quality=0.01; const double minDist=5.0;
  for (int r=0;r<gridRows;++r){
    for (int c=0;c<gridCols;++c){
      cv::Rect roi = cellRect(r,c,img.cols,img.rows,gridRows,gridCols);
      std::vector<cv::Point2f> tmp;
      cv::goodFeaturesToTrack(img(roi), tmp, maxPerCell, quality, minDist);
      for (auto& p: tmp){ p.x += roi.x; p.y += roi.y; out.push_back(p); }
    }
  }
}

// simple circular mask around points (avoid re-detecting on top of tracks)
static void maskFromPoints(const cv::Size& sz, const std::vector<cv::Point2f>& pts,
                           cv::Mat& mask, int radius=8)
{
  mask = cv::Mat(sz, CV_8UC1, cv::Scalar(255));
  for (auto& p: pts) cv::circle(mask, p, radius, cv::Scalar(0), -1, cv::LINE_AA);
}

static void topupDetection(const cv::Mat& img, std::vector<cv::Point2f>& pts,
                           int wantTotal, int gridRows,int gridCols,int perCell)
{
  if ((int)pts.size() >= wantTotal) return;
  cv::Mat m; maskFromPoints(img.size(), pts, m, 10);
  // detect with mask per cell
  std::vector<cv::Point2f> cand;
  for (int r=0;r<gridRows;++r){
    for (int c=0;c<gridCols;++c){
      cv::Rect roi = cellRect(r,c,img.cols,img.rows,gridRows,gridCols);
      // skip cell if it's heavily masked
      if (cv::countNonZero(m(roi)) < roi.area()/10) continue;
      std::vector<cv::Point2f> tmp;
      cv::goodFeaturesToTrack(img(roi), tmp, perCell, 0.01, 5.0, m(roi));
      for (auto& p: tmp){ p.x += roi.x; p.y += roi.y; cand.push_back(p); }
    }
  }
  // greedily add until we hit wantTotal
  for (auto& p: cand){ pts.push_back(p); if ((int)pts.size()>=wantTotal) break; }
}

static void pruneByForwardBackwardLK(const cv::Mat& prev, const cv::Mat& cur,
                                     std::vector<cv::Point2f>& p0,
                                     std::vector<cv::Point2f>& p1)
{
  if (p0.empty()) return;
  std::vector<uchar> st01, st10; std::vector<float> err01, err10;
  std::vector<cv::Point2f> p1fw, p0bw;

  cv::calcOpticalFlowPyrLK(prev, cur, p0, p1fw, st01, err01,
    cv::Size(15,15), 3, cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS, 20, 0.03),
    0, 1e-4);

  // back
  cv::calcOpticalFlowPyrLK(cur, prev, p1fw, p0bw, st10, err10,
    cv::Size(15,15), 3, cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS, 20, 0.03),
    0, 1e-4);

  std::vector<cv::Point2f> p0_out, p1_out; p0_out.reserve(p0.size()); p1_out.reserve(p0.size());
  for (size_t i=0;i<p0.size();++i){
    if (!st01[i] || !st10[i]) continue;
    if (cv::norm(p0[i] - p0bw[i]) < 0.8) { // FB threshold
      p0_out.push_back(p0[i]);
      p1_out.push_back(p1fw[i]);
    }
  }
  p0.swap(p0_out); p1.swap(p1_out);
}

static void pruneByGeometry(const std::vector<cv::Point2f>& p0,
                            const std::vector<cv::Point2f>& p1,
                            std::vector<uchar>& inlierMask,
                            double ransacThresh)
{
  if (p0.size()<8){ inlierMask.assign(p0.size(), 1); return; }
  // Fundamental (no intrinsics used in tracker)
  cv::findFundamentalMat(p0, p1, cv::USAC_MAGSAC, ransacThresh, 0.999, inlierMask);
}

void System::init(int width, int height, double fx, double fy, double cx, double cy) {
  w_ = width;  h_ = height;
  fx_ = fx; fy_ = fy; cx_ = cx; cy_ = cy;
  hasPrev_ = false;
  trackingState_ = 0;
  prevGray_.release();
  curGray_.release();
  ptsPrev_.clear();
  ptsCur_.clear();
}

void System::ensureGray(const uint8_t* img, int width, int height, bool isRGBA) {
  if (isRGBA) {
    cv::Mat rgba(h_, w_, CV_8UC4, const_cast<uint8_t*>(img));
    curGray_.create(h_, w_, CV_8UC1);
    cv::cvtColor(rgba, curGray_, cv::COLOR_RGBA2GRAY);
  } else {
    curGray_ = cv::Mat(h_, w_, CV_8UC1, const_cast<uint8_t*>(img)).clone();
  }
}


void System::detectNewPoints(const cv::Mat& gray) {
  std::vector<cv::Point2f> fresh;
  cv::goodFeaturesToTrack(
    gray, fresh, maxCorners_, qualityLevel_, minDistance_, cv::noArray(),
    blockSize_, useHarris_, 0.04
  );
  // Replace current set (simple policy)
  ptsCur_ = std::move(fresh);
}

void System::trackWithLK(const cv::Mat& prev, const cv::Mat& cur) {
  if (ptsPrev_.empty()) { ptsCur_.clear(); return; }

  std::vector<cv::Point2f> nextPts;
  std::vector<unsigned char> status;
  std::vector<float> err;

  cv::calcOpticalFlowPyrLK(
    prev, cur, ptsPrev_, nextPts, status, err,
    cv::Size(21,21), 3,
    cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS, 30, 0.01),
    0, 1e-4
  );

  // Keep only successfully tracked points within the frame
  ptsCur_.clear();
  ptsCur_.reserve(nextPts.size());
  for (size_t i=0; i<nextPts.size(); ++i) {
    if (!status[i]) continue;
    const cv::Point2f& p = nextPts[i];
    if (p.x >= 0 && p.y >= 0 && p.x < w_ && p.y < h_) {
      ptsCur_.push_back(p);
    }
  }
}

void System::feedFrame(const uint8_t* img, double ts, int width, int height, bool isRGBA) {
  lastTS_ = ts;

  // Re-init if size changed
  if (width!=w_ || height!=h_) {
    init(width, height, fx_, fy_, cx_, cy_);
  }

  // 1) Gray + downscale (reused Mats)
  ensureGray(img, width, height, isRGBA);        // fills curGray_
  if (procW_==0){ procW_ = int(w_*procScale_); procH_ = int(h_*procScale_); }
  curProc_.create(procH_, procW_, CV_8UC1);
  cv::resize(curGray_, curProc_, curProc_.size(), 0,0, cv::INTER_AREA);

  // 2) First frame -> detect
  static int frameCount = 0;
  if (!hasPrev_) {
    detectGridGFTT(curProc_, ptsCur_, gridRows_, gridCols_, maxPerCell_);
    prevProc_ = curProc_.clone();
    ptsPrev_  = ptsCur_;
    hasPrev_ = true;
    trackingState_ = 1;
    return;
  }

  // 3) Track prev -> cur with forward-backward check
  std::vector<cv::Point2f> p0 = ptsPrev_;
  std::vector<cv::Point2f> p1;
  p1.reserve(p0.size());
  pruneByForwardBackwardLK(prevProc_, curProc_, p0, p1);

  // 4) Geometric RANSAC pruning (threshold ~ 1.0 px at proc scale)
  std::vector<uchar> inl;
  pruneByGeometry(p0, p1, inl, 1.0);
  ptsCur_.clear(); ptsCur_.reserve(p1.size());
  for (size_t i=0;i<p1.size();++i) if (inl[i]) ptsCur_.push_back(p1[i]);

  // 5) Periodic redetect/top-up
  frameCount++;
  if (frameCount % redetectEveryN_ == 0 || (int)ptsCur_.size() < minKeep_) {
    topupDetection(curProc_, ptsCur_, gridRows_*gridCols_*maxPerCell_, gridRows_, gridCols_, maxPerCell_);
  }

  // 6) Roll
  prevProc_ = curProc_.clone();
  ptsPrev_  = ptsCur_;
  trackingState_ = 1;
}

std::vector<double> System::getPoints2D() const {
  std::vector<double> out; out.reserve(ptsCur_.size()*2);
  const double s = 1.0 / std::max(1e-6, (double)procScale_); // up-scale
  for (const auto& p : ptsCur_) { out.push_back(p.x * s); out.push_back(p.y * s); }
  return out;
}

