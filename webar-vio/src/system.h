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

        std::vector<cv::Point2f> p_prev, p_cur;
        p_prev.reserve(matches.size());
        p_cur.reserve(matches.size());
        for (const auto& m : matches) {
            p_prev.push_back(prev_.kps[m.queryIdx].pt); // x_{k-1}
            p_cur .push_back(cur .kps[m.trainIdx].pt); // x_k
        }

        // Estimate Essential matrix with RANSAC (pixel threshold ~1.0–1.5 is OK for 720p)
        cv::Mat inlierMask;
        const double prob = 0.999;
        const double ransac_thresh = 1.2; // pixels
        cv::Mat E = cv::findEssentialMat(p_cur, p_prev, Kcv_, cv::RANSAC, prob, ransac_thresh, inlierMask);

        if (E.empty()) {
            num_inliers_ = 0;
            tracking_state_ = 1;
            prev_ = std::move(cur);
            return;
        }

        cv::Mat R, t;
        int inl = cv::recoverPose(E, p_cur, p_prev, Kcv_, R, t, inlierMask);
        num_inliers_ = inl;

        // update state if robust enough
        if (inl >= 25) {
            // Build T_c2_c1 (cur->prev), then invert to get motion prev->cur
            Eigen::Matrix3d Re; cv::cv2eigen(R, Re);
            Eigen::Vector3d te; cv::cv2eigen(t, te);
            Sophus::SE3d T_c2c1(Re, te);
            Sophus::SE3d T_c1c2 = T_c2c1.inverse();

            // Accumulate world pose: T_wc(new) = T_wc(prev) * T_cprev_cnew
            T_wc_ = T_wc_ * T_c1c2;

            tracking_state_ = 2; // tracking
            last_inlier_count_ = inl;
        } else {
            tracking_state_ = 1; // still initializing/weak
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

    // Column-major 4x4 for three.js (CV -> three basis fix)
    // Returns current T_wc (world-from-camera) already converted for three.js.
    std::array<double,16> CurrentPoseGLThree() const {
        // S changes basis from OpenCV (camera +Z forward, X right, Y down)
        // to three.js (camera -Z forward, X right, Y up).
        Eigen::Matrix4d S = Eigen::Matrix4d::Identity();
        S(1,1) = -1.0; // flip Y
        S(2,2) = -1.0; // flip Z

        Eigen::Matrix4d Tcv = T_wc_.matrix();        // world-from-camera in CV basis
        Eigen::Matrix4d Tthree = S * Tcv * S;        // convert to three.js basis

        std::array<double,16> a{};
        int k=0;
        for (int c=0;c<4;++c)
            for (int r=0;r<4;++r)
                a[k++] = Tthree(r,c); // column-major
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
