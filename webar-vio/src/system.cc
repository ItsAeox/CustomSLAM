#include "system.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>

System::System() {}

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

void System::ensureGray(const uint8_t* img, int width, int height, bool isRGBA, cv::Mat& outGray) {
  if (isRGBA) {
    cv::Mat rgba(height, width, CV_8UC4, const_cast<uint8_t*>(img));
    cv::cvtColor(rgba, outGray, cv::COLOR_RGBA2GRAY);
  } else {
    // Copy to keep lifetime local to this call
    cv::Mat src(height, width, CV_8UC1, const_cast<uint8_t*>(img));
    outGray = src.clone();
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

  // If size changed, re-init
  if (width != w_ || height != h_) {
    init(width, height, fx_, fy_, cx_, cy_);
  }

  ensureGray(img, width, height, isRGBA, curGray_);

  if (!hasPrev_) {
    detectNewPoints(curGray_);
    prevGray_ = curGray_.clone();
    ptsPrev_  = ptsCur_;
    hasPrev_ = true;
    trackingState_ = 1;
    return;
  }

  // Track
  trackWithLK(prevGray_, curGray_);

  // If too few, top up with fresh detections (on current frame)
  if (static_cast<int>(ptsCur_.size()) < minKeep_) {
    // Merge: keep what we have, detect more, then append non-duplicates by distance
    std::vector<cv::Point2f> fresh;
    cv::goodFeaturesToTrack(curGray_, fresh, maxCorners_, qualityLevel_, minDistance_, cv::noArray(),
                            blockSize_, useHarris_, 0.04);

    // Simple spatial filter to avoid near-duplicates
    const float minDist2 = static_cast<float>(minDistance_ * minDistance_);
    for (const auto& q : fresh) {
      bool close = false;
      for (const auto& p : ptsCur_) {
        const float dx = p.x - q.x, dy = p.y - q.y;
        if (dx*dx + dy*dy < minDist2) { close = true; break; }
      }
      if (!close) ptsCur_.push_back(q);
      if ((int)ptsCur_.size() >= maxCorners_) break;
    }
  }

  // Housekeeping for next iteration
  prevGray_ = curGray_.clone();
  ptsPrev_  = ptsCur_;
  trackingState_ = 1;
}

std::vector<double> System::getPoints2D() const {
  std::vector<double> out;
  out.reserve(ptsCur_.size() * 2);
  for (const auto& p : ptsCur_) {
    out.push_back(static_cast<double>(p.x));
    out.push_back(static_cast<double>(p.y));
  }
  return out;
}

