#include "EqvioWeb.h"

#include <algorithm>
#include <chrono>
#include <iterator>

#include "eqvio/LoopTimer.h"
#include "eqvio/common/safeConfig.h"

namespace {
double msSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// eqvio_opt's convertGIFTFeatures (src/main_opt.cpp)
VisionMeasurement toMeasurement(const std::vector<GIFT::Feature>& features, double stamp) {
    VisionMeasurement m;
    m.stamp = stamp;
    std::transform(features.begin(), features.end(), std::inserter(m.camCoordinates, m.camCoordinates.begin()),
                   [](const GIFT::Feature& f) { return std::make_pair(f.idNumber, f.camCoordinatesEigen()); });
    return m;
}

constexpr double kRetiredMaxAge = 60;     // s
constexpr size_t kRetiredMax = 2000;
}  // namespace

EqvioWeb::EqvioWeb(const std::string& configYaml, const std::vector<double>& intrinsics, int width, int height,
                   const std::vector<double>& Tbc) {
    // VIOFilter times its stages with EqVIO's global loopTimer (LoopTimer.h), whose labels must exist (std::map::at)
    loopTimer.initialise({"correction", "preprocessing", "propagation"});
    config_ = YAML::Load(configYaml);
    cameraLag = configOrDefault(config_, "main:cameraLag", 0.0);
    initAverage = configOrDefault(config_, "main:initAverage", 0.0);
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = intrinsics[0];
    K.at<double>(1, 1) = intrinsics[1];
    K.at<double>(0, 2) = intrinsics[2];
    K.at<double>(1, 2) = intrinsics[3];
    camera_ = std::make_shared<GIFT::StandardCamera>(cv::Size(width, height), K, std::vector<ftype>{0, 0, 0, 0});
    settings_ = VIOFilter::Settings(config_["eqf"]);
    Eigen::Matrix4d T;
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) T(r, c) = Tbc[4 * r + c];
    settings_.cameraOffset = liepp::SE3d(T);  // as the ASL reader's sensor.yaml T_BS
    reset();
}

void EqvioWeb::reset() {
    filter_ = VIOFilter(settings_);
    tracker_ = std::make_unique<GIFT::PointFeatureTracker>(camera_);
    tracker_->settings.configure(config_["GIFT"]);
    state_ = VIOState();
    current_.clear();
    retired_.clear();
    nFeatures = 0;
}

void EqvioWeb::addImu(double t, const Eigen::Vector3d& gyr, const Eigen::Vector3d& acc) {
    IMUVelocity v;
    v.stamp = t;
    v.gyr = gyr;
    v.acc = acc;
    imu_.push_back(v);
}

void EqvioWeb::addFrame(double t, const cv::Mat& image) {
    const double stamp = t - cameraLag;
    auto t0 = std::chrono::steady_clock::now();
    if (!filter_.isInitialised() && initAverage > 0) {
        size_t n = 0;
        while (n < imu_.size() && imu_[n].stamp < stamp) n++;
        if (n == 0 || imu_[n - 1].stamp - imu_.front().stamp < initAverage) {  // not enough IMU yet: frame skipped
            nFeatures = 0;
            msFeatures = msFilter = 0;
            return;
        }
        IMUVelocity mean = imu_[n - 1];
        mean.acc.setZero();
        mean.gyr.setZero();
        for (size_t i = 0; i < n; i++) {
            mean.acc += imu_[i].acc;
            mean.gyr += imu_[i].gyr;
        }
        mean.acc /= double(n);
        mean.gyr /= double(n);
        filter_.initialiseFromIMUData(mean);
        imu_.erase(imu_.begin(), imu_.begin() + (n - 1));  // the newest one goes into the filter's IMU buffer below
    }
    // eqvio_opt's order: an IMU sample before a frame only if strictly older (image first on equal stamps)
    while (!imu_.empty() && imu_.front().stamp < stamp) {
        filter_.processIMUData(imu_.front());
        imu_.pop_front();
    }
    double ms = msSince(t0);
    t0 = std::chrono::steady_clock::now();
    const VisionMeasurement prediction = filter_.getFeaturePredictions(camera_, stamp);
    tracker_->processImage(image, prediction.ocvCoordinates());
    const auto features = tracker_->outputFeatures();
    VisionMeasurement m = toMeasurement(features, stamp);
    m.cameraPtr = camera_;
    msFeatures = msSince(t0);
    nFeatures = static_cast<int>(features.size());
    t0 = std::chrono::steady_clock::now();
    filter_.processVisionData(m);
    state_ = filter_.stateEstimate();
    msFilter = ms + msSince(t0);
    // landmarks: keep the last world estimate of those the filter dropped (point cloud for the AR page's surfaces)
    lastStamp_ = stamp;
    std::map<int, std::pair<int, Eigen::Vector3d>> now;
    const liepp::SE3d Twc = state_.sensor.pose * state_.sensor.cameraOffset;
    for (const auto& l : state_.cameraLandmarks) {
        const auto it = current_.find(l.id);
        now[l.id] = {it == current_.end() ? 1 : it->second.first + 1, Twc * l.p};
    }
    for (const auto& [id, ap] : current_)
        if (!now.count(id) && ap.first >= minAge) retired_.emplace_back(stamp, ap.second);
    current_ = std::move(now);
    while (!retired_.empty() && (retired_.size() > kRetiredMax || retired_.front().first < stamp - kRetiredMaxAge))
        retired_.pop_front();
}

Eigen::Matrix4d EqvioWeb::Twc() const { return (state_.sensor.pose * state_.sensor.cameraOffset).asMatrix(); }

std::vector<Eigen::Vector3d> EqvioWeb::points(double keepSeconds) const {
    std::vector<Eigen::Vector3d> out;
    out.reserve(current_.size() + retired_.size());
    for (const auto& [id, ap] : current_)
        if (ap.first >= minAge) out.push_back(ap.second);
    for (const auto& [t, p] : retired_)
        if (t >= lastStamp_ - keepSeconds) out.push_back(p);
    return out;
}
