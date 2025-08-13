#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "common/camera.h"
#include "map/frame.h"
#include "map/mappoint.h"
#include "frontend/feature_tracker.h"
#include <deque>
#include <algorithm>
#include <vector>
#include <cstdlib>   // for rand()


struct IMUSample {
    double t;                 // seconds, same clock as ProcessFrame ts
    Eigen::Vector3d acc;      // m/s^2 (device frame, may include gravity)
    Eigen::Vector3d gyro;     // rad/s  (device frame)
};

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
            // --- build matched points from prev -> cur ---
            std::vector<cv::Point2f> p_prev, p_cur;
            std::vector<int> prevIdxForMatch;             // 👈 NEW
            p_prev.reserve(matches.size());
            p_cur.reserve(matches.size());
            prevIdxForMatch.reserve(matches.size());
            for (const auto& m : matches) {
                p_prev.push_back(prev_.kps[m.queryIdx].pt);   // prev
                p_cur .push_back(cur .kps[m.trainIdx].pt);    // cur
                prevIdxForMatch.push_back(m.queryIdx);        // 👈 remember original row in prev_.desc
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
            cv::Mat Rcv, tcv;
            int inl = cv::recoverPose(E, p_prev, p_cur, Kcv_, Rcv, tcv, inlierMask);
            num_inliers_ = inl;

            if (inl >= 25) {
                Eigen::Matrix3d R_vo; cv::cv2eigen(Rcv, R_vo);
                Eigen::Vector3d t_vo; cv::cv2eigen(tcv, t_vo);

                if (!initialized_) {
                    // --- Two-view initialize ---
                    // 2) Triangulate inliers between prev (cam1) and cur (cam2)
                    // Build a char mask from inlierMask (CV_8U) for our helper
                    std::vector<char> inlVec(inlierMask.rows ? inlierMask.rows : inlierMask.cols, 0);
                    for (int i = 0; i < (int)inlVec.size(); ++i) {
                        unsigned char v = inlierMask.rows ? inlierMask.at<unsigned char>(i,0)
                                                        : inlierMask.at<unsigned char>(0,i);
                        inlVec[i] = (v != 0);
                    }
                    triangulateInliers(p_prev, p_cur, inlVec, prevIdxForMatch, R_vo, t_vo, prev_, cur);
                    fitPlaneViaHomography(p_prev, p_cur, R_vo, t_vo);

                    // 3) Convert VO relative [R,t] (cam1->cam2) to absolute world-from-camera T_wc
                    // P2 = K[R|t] => R_cw = R^T, t_cw = -R^T t
                    Eigen::Matrix3d R_wc = R_vo.transpose();
                    Eigen::Vector3d t_wc = -R_wc * t_vo;

                    // Blend IMU on WORLD rotation
                    Eigen::Quaterniond q_vo_world(R_wc);
                    Eigen::Quaterniond q_pred = R_imu_pred_.unit_quaternion();
                    double w_pred = imu_started_ ? 0.25 : 0.0;
                    Eigen::Quaterniond q_blend = q_pred.slerp(w_pred, q_vo_world).normalized();
                    T_wc_ = Sophus::SE3d(Sophus::SO3d(q_blend), t_wc);
                    
                    if (have_gravity_ && !did_world_align_) {
                        Eigen::Vector3d up_cam = -g_est_.normalized();
                        Eigen::Vector3d up_world(0, 1, 0);
                        Eigen::Quaterniond q_align = Eigen::Quaterniond::FromTwoVectors(up_cam, up_world);
                        Sophus::SO3d R_align(q_align.normalized());
                        // rotate camera pose
                        T_wc_ = Sophus::SE3d(R_align, Eigen::Vector3d::Zero()) * T_wc_;
                        // rotate existing map points into the aligned world
                        for (auto& mp : map_points_) mp.Xw = R_align * mp.Xw;
                        did_world_align_ = true;
                    }

                    // Optional: re-anchor IMU prediction
                    R_imu_pred_ = Sophus::SO3d(q_blend);

                    initialized_ = (map_points_.size() >= 40); // need some points
                    tracking_state_ = 2;
                } else {
                    // --- Tracking via PnP on map points ---
                    if (!trackWithPnP(cur)) {
                        // Use world-from-camera increment (inverse of [R|t])
                        bool planeOK = fitPlaneViaHomography(p_prev, p_cur, R_vo, t_vo);

                        Eigen::Matrix3d R_wc_inc = R_vo.transpose();
                        Eigen::Vector3d t_wc_inc = -R_wc_inc * t_vo;

                        // Blend IMU on WORLD rotation
                        Eigen::Quaterniond q_vo_world(R_wc_inc);
                        Eigen::Quaterniond q_pred = R_imu_pred_.unit_quaternion();
                        double w_pred = imu_started_ ? 0.25 : 0.0;
                        Eigen::Quaterniond q_blend = q_pred.slerp(w_pred, q_vo_world).normalized();

                        Sophus::SE3d T_inc(Sophus::SO3d(q_blend), t_wc_inc);
                        T_wc_ = T_wc_ * T_inc;

                        // Optional: re-anchor IMU prediction after successful visual step
                        R_imu_pred_ = Sophus::SO3d(q_blend);

                        tracking_state_ = 2;
                        num_inliers_ = inl;
                    } else {
                        // re‑anchor IMU prediction around successful update
                        R_imu_pred_ = Sophus::SO3d(T_wc_.so3().unit_quaternion());
                    }
                }
            } else {
                tracking_state_ = 1;
            }
            // frame_counter_++;
            // if ((frame_counter_ % 10) == 0 && (int)map_points_.size() >= 30) {
            //     fitDominantPlaneRansac();
            // }
            T_w_c_prev_ = T_wc_;   // cam2 becomes cam1 for the next iteration
            prev_ = std::move(cur);
        }


        void ProcessIMU(double ts, double ax, double ay, double az,
                        double gx, double gy, double gz) {
            // 1) push into buffer (unchanged from before)
            IMUSample s;
            s.t    = ts;
            s.acc  = Eigen::Vector3d(ax, ay, az);
            s.gyro = Eigen::Vector3d(gx, gy, gz);
            imu_buf_.push_back(s);
            const double tmin = ts - imu_keep_seconds_;
            while (!imu_buf_.empty() && imu_buf_.front().t < tmin) imu_buf_.pop_front();

            // 2) gravity LPF (when magnitude is plausible)
            const double anorm = s.acc.norm();
            if (anorm > 4.0 && anorm < 15.0) { // ~[0.4g,1.5g]
                const double alpha = 0.02;     // smooth; ~0.02 @100Hz ~ 0.5s time constant
                if (!have_gravity_) {
                    g_est_ = s.acc;
                    have_gravity_ = true;
                } else {
                    g_est_ = (1.0 - alpha) * g_est_ + alpha * s.acc;
                }
            }

            // 3) gyro integration for short‑term orientation prediction
            if (!imu_started_) {
                imu_last_t_ = ts;
                imu_started_ = true;
                return;
            }
            double dt = ts - imu_last_t_;
            imu_last_t_ = ts;
            if (dt <= 0.0 || dt > 0.1) return; // guard

            // NOTE: assume device frame ≈ camera frame for now; we’ll refine mapping later
            Eigen::Vector3d w = s.gyro; // rad/s
            Sophus::SO3d dR = Sophus::SO3d::exp(w * dt);
            R_imu_pred_ = R_imu_pred_ * dR;
        }

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

        // fetch IMU samples in [t0, t1]; inclusive at the ends
        std::vector<IMUSample> getIMUSpan(double t0, double t1) const {
            std::vector<IMUSample> out;
            if (imu_buf_.empty() || t1 <= t0) return out;
            for (const auto& s : imu_buf_) {
                if (s.t + 1e-6 >= t0 && s.t - 1e-6 <= t1) out.push_back(s);
            }
            return out;
        }

        // Returns current device "up" in camera/device frame (unit vector).
        Eigen::Vector3d UpHintCam() const {
            if (!have_gravity_) return Eigen::Vector3d(0,1,0);
            return (-g_est_).normalized(); // accel measures +g down; -g ≈ up
        }

        // Export up to maxN map points as [x0,y0,z0, x1,y1,z1, ...]
        std::vector<double> ExportPointsXYZ(int maxN = 500) const {
            std::vector<double> out;
            if (map_points_.empty()) return out;
            const int N = std::min<int>(maxN, (int)map_points_.size());
            out.reserve(N * 3);
            for (int i = 0; i < N; ++i) {
                const auto& X = map_points_[i].Xw;
                out.push_back(X.x());
                out.push_back(X.y());
                out.push_back(X.z());
            }
            return out;
        }

        // Export plane as [nx,ny,nz, d, cx,cy,cz, inliers] in THREE basis (x,-y,-z)
        // Returns empty if no plane
        std::vector<double> ExportPlaneThree() const {
            std::vector<double> out;
            if (!has_plane_ || !plane_.valid) return out;

            // Flip to Three basis: x' =  x, y' = -y, z' = -z
            auto flip = [](const Eigen::Vector3d& v){
                return Eigen::Vector3d(v.x(), -v.y(), -v.z());
            };
            Eigen::Vector3d n3 = flip(plane_.n).normalized();
            double d3 = plane_.d; // linear reflection about origin keeps d the same
            Eigen::Vector3d c3 = flip(plane_.centroid);

            out.reserve(8);
            out.push_back(n3.x()); out.push_back(n3.y()); out.push_back(n3.z());
            out.push_back(d3);
            out.push_back(c3.x()); out.push_back(c3.y()); out.push_back(c3.z());
            out.push_back((double)plane_.inliers);
            return out;
        }

        int lastInliers() const { return last_inlier_count_; }
        int NumMapPoints() const { return (int)map_points_.size(); }

    private:
        Camera cam_;
        FeatureTracker tracker_;
        bool has_prev_{false};
        Frame prev_;
        Sophus::SE3d T_wc_;
        cv::Mat Kcv_;
        int last_inlier_count_{0};
        std::deque<IMUSample> imu_buf_;
        double imu_keep_seconds_ = 5.0; // keep last 5s of IMU for preintegration
        // --- IMU fusion (step 2A) ---
        Eigen::Vector3d g_est_{0, -9.81, 0};   // running accel low‑pass (m/s^2), device frame
        bool have_gravity_{false};

        Sophus::SO3d R_imu_pred_{Sophus::SO3d()}; // integrated gyro orientation since last reset
        double imu_last_t_{0.0};
        bool imu_started_{false};

        // one‑time world "up" alignment
        bool did_world_align_{false};

        // ---- Minimal map / initializer state ----
        bool initialized_{false};
        std::vector<MapPoint> map_points_;   // compact point cloud
        cv::Mat map_desc_;                   // Nx32 CV_8U (stack of descriptors)
        cv::BFMatcher map_matcher_{cv::NORM_HAMMING, /*crossCheck=*/false};

        Sophus::SE3d T_w_c_prev_{Sophus::SE3d()};

        // ---- Plane model (dominant plane from map points) ----
        struct PlaneModel {
            Eigen::Vector3d n = Eigen::Vector3d::UnitY(); // unit normal
            double d = 0.0;                                // plane: n·X + d = 0
            int inliers = 0;
            Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
            bool valid = false;
        };
        PlaneModel plane_;
        bool has_plane_{false};
        int frame_counter_{0};

        // Build projection matrix K*[R|t] from Sophus pose T_wc (world-from-camera)
        inline cv::Matx34d projectionFromPose(const Sophus::SE3d& T_wc) const {
            Eigen::Matrix3d Rw = T_wc.rotationMatrix();
            Eigen::Vector3d tw = T_wc.translation();
            // We need P = K * [R_cw | t_cw] where R_cw,t_cw transform world->cam.
            // If T_wc is world-from-cam, then [R_cw|t_cw] = [R_w c^T | -R_w c^T * t_w c]
            Eigen::Matrix3d Rcw = Rw.transpose();
            Eigen::Vector3d tcw = -Rcw * tw;

            cv::Matx34d Rt;
            Rt(0,0)=Rcw(0,0); Rt(0,1)=Rcw(0,1); Rt(0,2)=Rcw(0,2); Rt(0,3)=tcw(0);
            Rt(1,0)=Rcw(1,0); Rt(1,1)=Rcw(1,1); Rt(1,2)=Rcw(1,2); Rt(1,3)=tcw(1);
            Rt(2,0)=Rcw(2,0); Rt(2,1)=Rcw(2,1); Rt(2,2)=Rcw(2,2); Rt(2,3)=tcw(2);

            cv::Matx33d K;
            K(0,0)=cam_.fx; K(0,1)=0;      K(0,2)=cam_.cx;
            K(1,0)=0;      K(1,1)=cam_.fy; K(1,2)=cam_.cy;
            K(2,0)=0;      K(2,1)=0;      K(2,2)=1;
            return K * Rt;
        }

        // Triangulate inliers between two views with known relative pose
        void triangulateInliers(const std::vector<cv::Point2f>& p1,
                                const std::vector<cv::Point2f>& p2,
                                const std::vector<char>& inlMask,
                                const std::vector<int>& prevIdxForMatch,   // 👈 NEW
                                const Eigen::Matrix3d& R12,
                                const Eigen::Vector3d& t12,
                                const Frame& f1, const Frame& /*f2*/) {
            Sophus::SE3d T_w_c1; // identity (world == cam1)

            // recoverPose gives cam1 -> cam2
            Sophus::SE3d T_c1_c2(R12, t12);

            // We need world-from-cam2 (cam2 -> world == cam1) = inverse of cam1->cam2
            Sophus::SE3d T_w_c2(R12.transpose(), -R12.transpose() * t12);

            // Build projection matrices K [R_cw | t_cw]
            cv::Matx34d P1 = projectionFromPose(T_w_c1);
            cv::Matx34d P2 = projectionFromPose(T_w_c2);

            // Collect inlier correspondences (pixel coords)
            std::vector<cv::Point2f> x1, x2;
            std::vector<int> idxMatch; idxMatch.reserve(inlMask.size());
            x1.reserve(inlMask.size()); x2.reserve(inlMask.size());
            for (size_t i = 0; i < inlMask.size(); ++i) {
                if (!inlMask[i]) continue;
                x1.push_back(p1[i]);
                x2.push_back(p2[i]);
                idxMatch.push_back((int)i);   // index into p1/p2 arrays
            }
            if ((int)x1.size() < 20) return;

            cv::Mat pts4; // 4xN (float)
            cv::triangulatePoints(P1, P2, x1, x2, pts4);

            auto proj = [&](const Eigen::Vector3d& Xc) {
                double u = cam_.fx*Xc.x()/Xc.z() + cam_.cx;
                double v = cam_.fy*Xc.y()/Xc.z() + cam_.cy;
                return cv::Point2f((float)u,(float)v);
            };

            for (int i = 0; i < pts4.cols; ++i) {
                // Robustly read homogeneous point (float or double)
                double X = (pts4.depth()==CV_64F) ? pts4.at<double>(0,i) : pts4.at<float>(0,i);
                double Y = (pts4.depth()==CV_64F) ? pts4.at<double>(1,i) : pts4.at<float>(1,i);
                double Z = (pts4.depth()==CV_64F) ? pts4.at<double>(2,i) : pts4.at<float>(2,i);
                double W = (pts4.depth()==CV_64F) ? pts4.at<double>(3,i) : pts4.at<float>(3,i);
                if (std::abs(W) < 1e-9) continue;
                Eigen::Vector3d Xc1(X/W, Y/W, Z/W);      // in cam1/world frame

                // Cheirality
                if (Xc1.z() <= 0) continue;
                Eigen::Vector3d Xc2 = R12 * Xc1 + t12;
                if (Xc2.z() <= 0) continue;

                // Reprojection filter (px)
                cv::Point2f u1 = x1[i], u2 = x2[i];
                if (cv::norm(u1 - proj(Xc1)) > 3.0 || cv::norm(u2 - proj(Xc2)) > 3.0) continue;

                // ✅ Correct descriptor row from the original prev frame
                const int matchIdx = idxMatch[i];                 // index into p1/p2
                if (matchIdx < 0 || matchIdx >= (int)prevIdxForMatch.size()) continue;
                const int kpIdx = prevIdxForMatch[matchIdx];      // row in f1.desc
                if (kpIdx < 0 || kpIdx >= f1.desc.rows) continue;

                MapPoint mp;
                mp.Xw  = Xc1;                         // world == cam1
                mp.desc = f1.desc.row(kpIdx).clone(); // descriptor from prev frame
                map_points_.push_back(std::move(mp));
            }

            // Rebuild descriptor matrix
            if (!map_points_.empty()) {
                map_desc_.release();
                const int D = f1.desc.cols;              // descriptor length (usually 32 for ORB)
                map_desc_ = cv::Mat((int)map_points_.size(), D, CV_8U);
                for (int r = 0; r < (int)map_points_.size(); ++r) {
                    map_points_[r].desc.copyTo(map_desc_.row(r));
                }
            }
        }  


        // Match map points to current descriptors and run PnP
        bool trackWithPnP(const Frame& cur) {
            if (map_points_.empty() || map_desc_.empty() || cur.desc.empty()) return false;

            // KNN + ratio test
            std::vector<std::vector<cv::DMatch>> knn;
            map_matcher_.knnMatch(map_desc_, cur.desc, knn, 2);
            std::vector<cv::DMatch> good;
            good.reserve(knn.size());
            for (auto& v : knn) {
                if (v.size()==2 && v[0].distance < 0.8f * v[1].distance) good.push_back(v[0]);
            }
            if (good.size() < 12) return false;

            std::vector<cv::Point3f> pts3;
            std::vector<cv::Point2f> pts2;
            pts3.reserve(good.size());
            pts2.reserve(good.size());
            for (const auto& m : good) {
                const MapPoint& mp = map_points_[m.queryIdx]; // queryIdx indexes map_desc_ rows
                pts3.emplace_back((float)mp.Xw.x(), (float)mp.Xw.y(), (float)mp.Xw.z());
                pts2.emplace_back(cur.kps[m.trainIdx].pt);
            }

            cv::Mat rvec, tvec, inliers;
            cv::Mat K = Kcv_;
            bool ok = cv::solvePnPRansac(pts3, pts2, K, cv::noArray(),
                                        rvec, tvec, /*useExtrinsicGuess=*/false,
                                        100, 3.0, 0.99, inliers, cv::SOLVEPNP_EPNP);
            if (!ok || inliers.rows < 12) return false;

            // Convert to Sophus SE3
            cv::Mat Rcv;
            cv::Rodrigues(rvec, Rcv);

            // OpenCV returns R_cw, t_cw (world->camera)
            Eigen::Matrix3d R_cw; cv::cv2eigen(Rcv, R_cw);
            Eigen::Vector3d t_cw; cv::cv2eigen(tvec, t_cw);

            // Convert to world-from-camera: T_wc = [R_cw^T, -R_cw^T * t_cw]
            Eigen::Matrix3d R_wc = R_cw.transpose();
            Eigen::Vector3d t_wc = -R_wc * t_cw;

            // Blend IMU on *world* rotation
            Eigen::Quaterniond q_vo(R_wc);
            Eigen::Quaterniond q_pred = R_imu_pred_.unit_quaternion();
            double w_pred = imu_started_ ? 0.25 : 0.0;
            Eigen::Quaterniond q_blend = q_pred.slerp(w_pred, q_vo).normalized();

            T_wc_ = Sophus::SE3d(Sophus::SO3d(q_blend), t_wc);  // ✅ correct frame
            R_imu_pred_ = Sophus::SO3d(T_wc_.so3().unit_quaternion());
            tracking_state_ = 2;
            num_inliers_ = inliers.rows;
            return true;
        }

        // Fit dominant plane with simple RANSAC over map_points_ (world frame)
        void fitDominantPlaneRansac() {
            const int N = (int)map_points_.size();
            if (N < 50) { has_plane_ = false; plane_ = PlaneModel(); return; }

            // Gather points to a simple vector for cache locality
            std::vector<Eigen::Vector3d> pts; pts.reserve(N);
            for (const auto& mp : map_points_) pts.push_back(mp.Xw);

            // Robust distance scale: use median radius to set threshold (~2%)
            std::vector<double> radii; radii.reserve(N);
            for (auto& p : pts) radii.push_back(p.norm());
            std::nth_element(radii.begin(), radii.begin()+radii.size()/2, radii.end());
            const double med = radii[radii.size()/2];
            const double dist_thresh = std::max(1e-3, 0.06 * med); // ~6% of scene scale

            auto planeFrom3 = [](const Eigen::Vector3d& a,
                                const Eigen::Vector3d& b,
                                const Eigen::Vector3d& c,
                                Eigen::Vector3d& n, double& d)->bool {
                Eigen::Vector3d nraw = (b - a).cross(c - a);
                double nn = nraw.norm();
                if (nn < 1e-9) return false;
                n = nraw / nn;
                d = -n.dot(a);
                return true;
            };

            // Simple RANSAC
            int best_inl = 0;
            Eigen::Vector3d best_n(0,1,0), best_c = Eigen::Vector3d::Zero();
            double best_d = 0;
            const int iters = 200;
            for (int it = 0; it < iters; ++it) {
                // sample 3 distinct indices
                int i = rand() % N, j = rand() % N, k = rand() % N;
                if (i==j || i==k || j==k) { --it; continue; }

                Eigen::Vector3d n; double d;
                if (!planeFrom3(pts[i], pts[j], pts[k], n, d)) continue;

                // count inliers
                int cnt = 0;
                Eigen::Vector3d csum(0,0,0);
                for (int t = 0; t < N; ++t) {
                    double dist = std::abs(n.dot(pts[t]) + d);
                    if (dist < dist_thresh) { ++cnt; csum += pts[t]; }
                }
                if (cnt > best_inl) {
                    best_inl = cnt;
                    best_n = n;
                    best_d = d;
                    // best_c = (cnt>0) ? (csum / (double)cnt) : Eigen::Vector3d::Zero();  // <-- remove this
                    if (cnt > 0) best_c = csum / double(cnt);
                    else         best_c.setZero();
                }
            }

            // Accept if enough inliers
            if (best_inl >= 50) {
                // Re-orient normal to point roughly "up" (optional)
                Eigen::Vector3d world_up(0,1,0);
                if (best_n.dot(world_up) < 0) { best_n = -best_n; best_d = -best_d; }

                plane_.n = best_n.normalized();
                plane_.d = best_d;
                plane_.inliers = best_inl;
                plane_.centroid = best_c;
                plane_.valid = true;
                has_plane_ = true;
            } else {
                has_plane_ = false;
                plane_ = PlaneModel();
            }
        }

        bool fitPlaneViaHomography(const std::vector<cv::Point2f>& p1,
                                const std::vector<cv::Point2f>& p2,
                                const Eigen::Matrix3d& R12,
                                const Eigen::Vector3d& t12) {
            if (p1.size() < 40 || p2.size() < 40) return false;

            // 1) Robust homography in pixel coords
            cv::Mat inlH;
            cv::Mat H = cv::findHomography(p1, p2, cv::RANSAC, 3.0, inlH, 2000, 0.995);
            if (H.empty()) return false;

            int hinl = cv::countNonZero(inlH);
            // Require a meaningful planar consensus
            //if (hinl < 80 || hinl < (int)p1.size() * 0.25) return false;
            int need = std::max(30, int(p1.size()*0.30));   // was 80 / 25%
            if (hinl < need) return false;

            // 2) Normalize by intrinsics
            Eigen::Matrix3d K;  cv::cv2eigen(Kcv_, K);
            Eigen::Matrix3d He; cv::cv2eigen(H, He);
            Eigen::Matrix3d Hn = K.inverse() * He * K;      // ~ R + t n^T / d

            // 3) Compute v = n/d from A = Hn - R = t v^T  (rank-1)
            Eigen::Matrix3d R = R12;
            Eigen::Vector3d t = t12;
            double t2 = t.squaredNorm();
            if (t2 < 1e-10) return false;

            Eigen::Matrix3d A = Hn - R;
            Eigen::Vector3d v = A.transpose() * t / t2;     // least squares
            double vnorm = v.norm();
            if (vnorm < 1e-9 || !std::isfinite(vnorm)) return false;

            Eigen::Vector3d n_cam1 = v / vnorm;            // unit normal in cam1
            double d_cam1 = 1.0 / vnorm;                    // distance in cam1

            // Fix sign so normal roughly points "up" if we have gravity
            if (have_gravity_) {
                Eigen::Vector3d up_cam = -g_est_.normalized();
                if (n_cam1.dot(up_cam) < 0) { n_cam1 = -n_cam1; d_cam1 = -d_cam1; }
            }

            // 4) Convert plane from cam1 to world using stored T_w_c_prev_
            Eigen::Matrix3d Rwc = T_w_c_prev_.rotationMatrix();
            Eigen::Vector3d twc = T_w_c_prev_.translation();

            Eigen::Vector3d n_w = Rwc * n_cam1;
            Eigen::Vector3d c_cam1 = -d_cam1 * n_cam1;          // centroid point on plane in cam1
            Eigen::Vector3d c_w = Rwc * c_cam1 + twc;

            // Optional: EMA smoothing to reduce jitter
            const double alpha = 0.2;
            if (has_plane_ && plane_.valid) {
                n_w = (1.0 - alpha) * plane_.n + alpha * n_w; n_w.normalize();
                c_w = (1.0 - alpha) * plane_.centroid + alpha * c_w;
            }

            // Pack model
            // Recompute d from n·X + d = 0 using centroid
            double d_w = -n_w.dot(c_w);

            // Ensure normal upright if we know gravity (flip if needed)
            if (have_gravity_ && n_w.dot(Eigen::Vector3d(0,1,0)) < 0) { n_w = -n_w; d_w = -d_w; }

            plane_.n = n_w.normalized();
            plane_.d = d_w;
            plane_.centroid = c_w;
            plane_.inliers = hinl;
            plane_.valid = true;
            has_plane_ = true;
            return true;
        }

};
