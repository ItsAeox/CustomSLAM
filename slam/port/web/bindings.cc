// JavaScript API for the ORB-SLAM3 web port (Embind). Runs inside a Web Worker (slam.worker.js);
// ORB-SLAM3's LocalMapping / LoopClosing threads become further workers via emscripten pthreads.
//
//   const slam = new Module.WebSlam();
//   slam.init('/vocab/ORBvoc.bin', '/settings.yaml', false, -1); // files in the emscripten FS (MEMFS); clahe; seed
//   const ptr = slam.frameBuffer(w, h);                          // gray8 buffer inside the wasm heap
//   HEAPU8.set(gray, ptr); const state = slam.track(t);          // t in seconds
//   slam.poseTwc()   -> Float32Array(16), column-major camera-to-world
//   slam.trackedPoints() -> Float32Array(3N), world coordinates of the map points seen in the last frame

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <chrono>
#include <memory>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "MapPoint.h"
#include "System.h"
#include "Thirdparty/DBoW2/DUtils/Random.h"

using namespace emscripten;

class WebSlam {
 public:
  // seed < 0 keeps ORB-SLAM3's default (TwoViewReconstruction seeds with 0); otherwise the RANSAC seed.
  bool init(const std::string& vocabFile, const std::string& settingsFile, bool clahe, int seed) {
    if (seed >= 0) DUtils::Random::SeedRandOnce(seed);
    clahe_ = clahe;
    if (clahe_) claheOp_ = cv::createCLAHE(3.0, cv::Size(8, 8));  // as in ORB-SLAM3's TUM-VI examples
    sys_ = std::make_unique<ORB_SLAM3::System>(vocabFile, settingsFile, ORB_SLAM3::System::MONOCULAR,
                                               /*bUseViewer=*/false);
    return true;
  }

  // Returns the address of a w*h gray8 buffer in the wasm heap that JS fills before calling track().
  uintptr_t frameBuffer(int w, int h) {
    if (frame_.cols != w || frame_.rows != h) frame_.create(h, w, CV_8UC1);
    return reinterpret_cast<uintptr_t>(frame_.data);
  }

  // Track the frame currently in frameBuffer(). Returns ORB-SLAM3's tracking state
  // (-1 system not ready, 0 no images yet, 1 not initialized, 2 OK, 3 recently lost, 4 lost, 5 OK klt).
  int track(double timestamp) {
    if (!sys_) return -1;
    auto t0 = std::chrono::steady_clock::now();
    cv::Mat im = frame_;
    if (clahe_) { claheOp_->apply(frame_, work_); im = work_; }
    // As ORB-SLAM3's examples: resize when the settings request it (Camera.newWidth/newHeight, e.g. EuRoC 600x350)
    const float scale = sys_->GetImageScale();
    if (scale != 1.f) {
      cv::resize(im, scaled_, cv::Size((int)(im.cols * scale), (int)(im.rows * scale)));
      im = scaled_;
    }
    Sophus::SE3f Tcw = sys_->TrackMonocular(im, timestamp);
    lastMs_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const int state = sys_->GetTrackingState();
    Eigen::Matrix4f Twc = Tcw.inverse().matrix();
    for (int c = 0; c < 4; ++c)
      for (int r = 0; r < 4; ++r) twc_[c * 4 + r] = Twc(r, c);
    return state;
  }

  val poseTwc() const { return val(typed_memory_view(16, twc_)); }
  double lastTrackMs() const { return lastMs_; }

  val trackedPoints() {
    pts_.clear();
    if (sys_) {
      for (ORB_SLAM3::MapPoint* mp : sys_->GetTrackedMapPoints()) {
        if (!mp || mp->isBad()) continue;
        Eigen::Vector3f p = mp->GetWorldPos();
        pts_.insert(pts_.end(), {p.x(), p.y(), p.z()});
      }
    }
    return val(typed_memory_view(pts_.size(), pts_.data()));
  }

  void reset() { if (sys_) sys_->ResetActiveMap(); }
  void shutdown() { if (sys_) { sys_->Shutdown(); sys_.reset(); } }

  // End of a dataset run: stop the threads and save ORB-SLAM3's final (optimised) trajectory, exactly like
  // the native driver does (EuRoC format, ns timestamps). Files are written to the emscripten FS.
  bool finishAndSave(const std::string& framesPath, const std::string& keyframesPath) {
    if (!sys_) return false;
    sys_->Shutdown();
    sys_->SaveTrajectoryEuRoC(framesPath);
    sys_->SaveKeyFrameTrajectoryEuRoC(keyframesPath);
    return true;
  }

 private:
  std::unique_ptr<ORB_SLAM3::System> sys_;
  cv::Mat frame_, work_, scaled_;
  bool clahe_ = false;
  cv::Ptr<cv::CLAHE> claheOp_;
  float twc_[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  std::vector<float> pts_;
  double lastMs_ = 0;
};

EMSCRIPTEN_BINDINGS(orbslam3_web) {
  class_<WebSlam>("WebSlam")
      .constructor<>()
      .function("init", &WebSlam::init)
      .function("frameBuffer", &WebSlam::frameBuffer)
      .function("track", &WebSlam::track)
      .function("poseTwc", &WebSlam::poseTwc)
      .function("lastTrackMs", &WebSlam::lastTrackMs)
      .function("trackedPoints", &WebSlam::trackedPoints)
      .function("reset", &WebSlam::reset)
      .function("shutdown", &WebSlam::shutdown)
      .function("finishAndSave", &WebSlam::finishAndSave);
}
