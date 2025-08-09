#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "common/camera.h"
#include "map/frame.h"
#include "frontend/feature_tracker.h"

class System {
    public:
    explicit System(const Camera& cam) : cam_(cam) {
        T_wc_ = Sophus::SE3d(); // identity
        Kcv_ = (cv::Mat_<double>(3,3) << cam_.fx,0,cam_.cx, 0,cam_.fy,cam_.cy, 0,0,1);
    }

    // RGBA or grayscale accepted; we convert to gray
    void ProcessFrame(const uint8_t* img, double ts, int strideRGBAorGray, bool isRGBA=true) {
        cv::Mat gray;
        if (isRGBA) {
            cv::Mat rgba(cam_.height, cam_.width, CV_8UC4, const_cast<uint8_t*>(img), strideRGBAorGray);
            cv::cvtColor(rgba, gray, cv::COLOR_RGBA2GRAY);
        } else {
            gray = cv::Mat(cam_.height, cam_.width, CV_8UC1, const_cast<uint8_t*>(img), strideRGBAorGray).clone();
        }

        Frame cur; cur.timestamp=ts; cur.gray=gray;
        tracker_.detectAndDescribe(cur.gray, cur.kps, cur.desc);

        if (has_prev_) {
            // match to previous
            auto matches = tracker_.match(prev_.desc, cur.desc);
            if (matches.size() >= 20) {
                std::vector<cv::Point2f> p1, p2;
                p1.reserve(matches.size()); p2.reserve(matches.size());
                for (auto&m : matches) {
                    p1.push_back(prev_.kps[m.queryIdx].pt);
                    p2.push_back(cur.kps[m.trainIdx].pt);
                }

                // Essential matrix with RANSAC
                cv::Mat inlierMask;
                cv::Mat E = cv::findEssentialMat(p2, p1, Kcv_, cv::RANSAC, 0.999, 1.5, inlierMask);
                cv::Mat R, t;
                int inl = cv::recoverPose(E, p2, p1, Kcv_, R, t, inlierMask);

                if (inl >= 15) {
                    // Build T_c2_c1 (from cur to prev image coordinates)
                    Eigen::Matrix3d Re; cv::cv2eigen(R, Re);
                    Eigen::Vector3d te; cv::cv2eigen(t, te);
                    Sophus::SE3d T_c2c1(Re, te);

                    // world poses: T_wc(new) = T_wc(prev) * T_cprev_cnew^{-1}
                    // Note: recoverPose returns T_c2_c1, we want T_c1_c2
                    Sophus::SE3d T_c1c2 = T_c2c1.inverse();
                    T_wc_ = T_wc_ * T_c1c2; // accumulate (scale=1)

                    last_inlier_count_ = inl;
                }
            }
        }
        prev_ = std::move(cur);
        has_prev_ = true;
    }

    // Stub for now (M1 will use this)
    void ProcessIMU(double /*ts*/, double /*ax*/, double /*ay*/, double /*az*/,
                    double /*gx*/, double /*gy*/, double /*gz*/) {}

    // Column-major 4x4 for WebGL
    std::array<double,16> CurrentPoseGL() const {
        // We return T_wc (world-from-camera). Three.js wants object->world.
        // If you place the cactus at world origin, apply T_wc as the camera pose inverse, or
        // use T_cw depending on your renderer logic. For now we expose T_wc.
        Eigen::Matrix4d T = T_wc_.matrix();
        std::array<double,16> a{};
        int k=0;
        for (int c=0;c<4;++c) for (int r=0;r<4;++r) a[k++] = T(r,c);
            return a;
    }

    int lastInliers() const { return last_inlier_count_; }

    private:
        Camera cam_;
        FeatureTracker tracker_;
        bool has_prev_{false};
        Frame prev_;
        Sophus::SE3d T_wc_;
        cv::Mat Kcv_;
        int last_inlier_count_{0};
};
