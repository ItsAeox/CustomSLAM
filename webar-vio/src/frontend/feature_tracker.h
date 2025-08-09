#pragma once
#include <opencv2/opencv.hpp>

class FeatureTracker {
public:
  explicit FeatureTracker(int nfeatures = 1200) {
    // ORB tuned a bit higher for stability on webcams
    orb_ = cv::ORB::create(nfeatures, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
    matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, /*crossCheck=*/false);
  }

  void detectAndDescribe(const cv::Mat& gray,
                         std::vector<cv::KeyPoint>& kps, cv::Mat& desc) {
    CV_Assert(!gray.empty());
    cv::Mat g;
    if (gray.channels() == 1) g = gray;
    else cv::cvtColor(gray, g, cv::COLOR_BGR2GRAY);
    orb_->detectAndCompute(g, cv::noArray(), kps, desc);
  }

  // ratio-test + symmetry, with safety checks
  std::vector<cv::DMatch> match(const cv::Mat& d1, const cv::Mat& d2) {
    std::vector<cv::DMatch> empty;
    if (d1.empty() || d2.empty()) return empty;
    if (d1.type() != CV_8U || d2.type() != CV_8U) return empty;
    if (d1.cols != d2.cols) return empty; // ORB: 32 bytes per descriptor

    std::vector<std::vector<cv::DMatch>> knn12, knn21;
    matcher_->knnMatch(d1, d2, knn12, 2);
    matcher_->knnMatch(d2, d1, knn21, 2);

    auto good12 = ratioFilter(knn12);
    auto good21 = ratioFilter(knn21);

    std::vector<cv::DMatch> sym;
    sym.reserve(std::min(good12.size(), good21.size()));
    for (auto& m12 : good12) {
      for (auto& m21 : good21) {
        if (m12.queryIdx == m21.trainIdx && m12.trainIdx == m21.queryIdx) {
          sym.push_back(m12);
          break;
        }
      }
    }
    return sym;
  }

private:
  static std::vector<cv::DMatch> ratioFilter(
      const std::vector<std::vector<cv::DMatch>>& knn, float r=0.8f) {
    std::vector<cv::DMatch> out;
    out.reserve(knn.size());
    for (auto& v : knn)
      if (v.size()==2 && v[0].distance < r * v[1].distance) out.push_back(v[0]);
    return out;
  }

  cv::Ptr<cv::ORB> orb_;
  cv::Ptr<cv::BFMatcher> matcher_;
};
