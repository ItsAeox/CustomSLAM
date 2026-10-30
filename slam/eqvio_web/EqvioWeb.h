// WebVIO's EqVIO mode (claude/PLAN.md 4d): EqVIO (equivariant filter VIO, van Goor & Mahony, T-RO 2023;
// github.com/pvangoor/eqvio, GPL-3.0) with its GIFT feature tracker, driven frame by frame for the browser (bindings.cc)
// and for the native / Node replay (eqvio_run.cc). Fed as eqvio_opt (src/main_opt.cpp): every IMU sample up to a frame's
// time, then the frame; frame stamps moved to the IMU clock by main:cameraLag (camera stamp - IMU-clock time).
#pragma once

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include "GIFT/PointFeatureTracker.h"
#include "eqvio/VIOFilter.h"
#include "eqvio/VIOFilterSettings.h"

class EqvioWeb {
  public:
    // config: EqVIO YAML text (eqf, GIFT, main:cameraLag); intrinsics fx fy cx cy (pinhole, no distortion); Tbc: the
    // camera pose in the IMU frame, row-major 4x4 (sensor.yaml T_BS; settings.yaml IMU.T_b_c1)
    EqvioWeb(const std::string& configYaml, const std::vector<double>& intrinsics, int width, int height,
             const std::vector<double>& Tbc);
    void addImu(double t, const Eigen::Vector3d& gyr, const Eigen::Vector3d& acc);  // IMU clock (s)
    void addFrame(double t, const cv::Mat& image);  // camera clock (s); grey (as the browser) or BGR (as eqvio_opt)
    void reset();  // a new filter and feature tracker, same settings (queued IMU samples are kept)

    bool initialised() const { return filter_.isInitialised(); }
    double time() const { return filter_.getTime(); }  // IMU clock
    Eigen::Matrix4d Twc() const;                        // camera pose in the gravity-aligned world (m)
    double speed() const { return state_.sensor.velocity.norm(); }  // m/s (divergence check)
    // the filter's landmarks in the world (m), plus those it dropped within the last keepSeconds (their last estimate);
    // only landmarks tracked for at least minAge frames (a new landmark starts at the configured scene depth)
    std::vector<Eigen::Vector3d> points(double keepSeconds = 0) const;
    int minAge = 15;

    double cameraLag = 0;
    // main:initAverage (s): EqVIO takes its attitude (gravity) from the first IMU sample; > 0: from the mean specific force
    // over this span instead (frames before it are skipped). 0 = upstream.
    double initAverage = 0;
    int nFeatures = 0;                     // tracked features in the last frame
    double msFeatures = 0, msFilter = 0;   // last frame: GIFT tracking, filter (IMU propagation + update)

  private:
    YAML::Node config_;
    VIOFilter::Settings settings_;
    std::shared_ptr<GIFT::StandardCamera> camera_;
    VIOFilter filter_;
    std::unique_ptr<GIFT::PointFeatureTracker> tracker_;
    std::deque<IMUVelocity> imu_;
    VIOState state_;
    double lastStamp_ = 0;
    std::map<int, std::pair<int, Eigen::Vector3d>> current_;  // landmark id -> (frames tracked, world position)
    std::deque<std::pair<double, Eigen::Vector3d>> retired_;  // (frame time, last world position) of dropped landmarks
};
