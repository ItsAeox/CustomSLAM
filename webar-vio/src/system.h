#pragma once
#include <vector>
#include <cstdint>
#include <opencv2/core.hpp>

class System {
public:
  System();

  // (Intrinsics not strictly needed for 2D tracking, but kept for parity/future)
  void init(int width, int height, double fx, double fy, double cx, double cy);

  // img: RGBA or Gray buffer straight from JS; ts in seconds
  void feedFrame(const uint8_t* img, double ts, int width, int height, bool isRGBA);

  // Flattened [x0,y0, x1,y1, ...] in pixel coords (image space, origin top-left)
  std::vector<double> getPoints2D() const;

  int getNumKeypoints() const { return static_cast<int>(ptsCur_.size()); }
  int getTrackState()   const { return trackingState_; } // 0=uninit,1=tracking
  double getLastTS()    const { return lastTS_; }

private:
  void ensureGray(const uint8_t* img, int width, int height, bool isRGBA, cv::Mat& outGray);
  void detectNewPoints(const cv::Mat& gray);
  void trackWithLK(const cv::Mat& prev, const cv::Mat& cur);

  // Tuning
  int   maxCorners_     = 1000;
  double qualityLevel_  = 0.01;
  double minDistance_   = 7.0;
  int   blockSize_      = 3;
  bool  useHarris_      = false;

  int   redetectEveryN_ = 12;   // refresh features periodically
  int   minKeep_        = 200;  // if we fall below this, re-detect

  // State
  bool  hasPrev_   = false;
  int   trackingState_ = 0;
  int   w_ = 0, h_ = 0;
  double fx_=0, fy_=0, cx_=0, cy_=0;
  double lastTS_ = 0.0;

  cv::Mat prevGray_, curGray_;
  std::vector<cv::Point2f> ptsPrev_, ptsCur_;
};
