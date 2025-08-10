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
    // HUD counters (read via bindings)
    int num_keypoints_   = 0;   // detected features this frame
    int num_inliers_     = 0;   // E-matrix inliers from recoverPose
    int tracking_state_  = 0;   // 0=idle/no prev, 1=initializing, 2=tracking

    explicit System(const Camera& cam) : cam_(cam) {
        T_wc_ = Sophus::SE3d(); // identity
        Kcv_  = (cv::Mat_<double>(3,3) << cam_.fx,0,cam_.cx, 0,cam_.fy,cam_.cy, 0,0,1);
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

        Frame cur; cur.timestamp = ts; cur.gray = gray;
        tracker_.detectAndDescribe(cur.gray, cur.kps, cur.desc);
        num_keypoints_ = static_cast<int>(cur.kps.size());

        // no previous frame -> stash and wait
        if (!has_prev_) {
            prev_ = std::move(cur);
            has_prev_ = true;
            tracking_state_ = 1;        // initializing (need a baseline)
            num_inliers_ = 0;
            return;
        }

        // match to previous
        auto matches = tracker_.match(prev_.desc, cur.desc);
        if (matches.size() < 20) {
            // not enough for a robust E
            num_inliers_ = 0;
            tracking_state_ = 1;
            prev_ = std::move(cur);
            return;
        }

        // --- build matched points from prev -> cur (you already have matches) ---
        std::vector<cv::Point2f> p_prev, p_cur;
        p_prev.reserve(matches.size());
        p_cur.reserve(matches.size());
        for (const auto& m : matches) {
            p_prev.push_back(prev_.kps[m.queryIdx].pt); // prev
            p_cur .push_back(cur .kps[m.trainIdx].pt);  // cur
        }

        // --- estimate Essential (looser threshold on mobile) ---
        cv::Mat inlierMask;
        const double prob = 0.999;
        const double ransac_thresh = 2.0; // px
        cv::Mat E = cv::findEssentialMat(p_prev, p_cur, Kcv_, cv::RANSAC, prob, ransac_thresh, inlierMask);

        if (E.empty()) {
            num_inliers_ = 0;
            tracking_state_ = 1;
            prev_ = std::move(cur);
            return;
        }

        // --- recoverPose RETURNS motion that maps prev -> cur ---
        cv::Mat R, t;
        int inl = cv::recoverPose(E, p_prev, p_cur, Kcv_, R, t, inlierMask);
        num_inliers_ = inl;

        if (inl >= 25) {
            Eigen::Matrix3d Re; cv::cv2eigen(R, Re);
            Eigen::Vector3d te; cv::cv2eigen(t, te);

            // Motion from prev to cur in CAMERA frame
            Sophus::SE3d T_c1c2(Re, te);

            // Accumulate world pose: T_wc(new) = T_wc(prev) * T_cprev_cnew
            T_wc_ = T_wc_ * T_c1c2;

            tracking_state_ = 2;
        } else {
            tracking_state_ = 1;
        }

        prev_ = std::move(cur);
    }

    // IMU stub (we'll fuse at M1)
    void ProcessIMU(double /*ts*/, double /*ax*/, double /*ay*/, double /*az*/,
                    double /*gx*/, double /*gy*/, double /*gz*/) {}

    // Column-major 4x4 for WebGL (always returns something: current T_wc_)
    std::array<double,16> CurrentPoseGL() const {
        Eigen::Matrix4d T = T_wc_.matrix();
        std::array<double,16> a{};
        int k=0;
        for (int c=0;c<4;++c)
            for (int r=0;r<4;++r)
                a[k++] = T(r,c);
        return a;
    }

    std::array<double,16> CurrentPoseGLThree() const {
        Eigen::Matrix4d S = Eigen::Matrix4d::Identity();
        S(1,1) = -1.0; // flip Y
        S(2,2) = -1.0; // flip Z
        Eigen::Matrix4d Tthree = S * T_wc_.matrix() * S;

        std::array<double,16> a{};
        int k=0;
        for (int c=0;c<4;++c)
        for (int r=0;r<4;++r)
            a[k++] = Tthree(r,c);
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
