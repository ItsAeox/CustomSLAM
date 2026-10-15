// JavaScript API for the ORB-SLAM3 web port (Embind). Runs inside a Web Worker (slam.worker.js);
// ORB-SLAM3's LocalMapping / LoopClosing threads become further workers via emscripten pthreads.
//
//   const slam = new Module.WebSlam();
//   slam.init('/vocab/ORBvoc.bin', '/settings.yaml', false, -1); // files in the emscripten FS (MEMFS); clahe; seed
//   const ptr = slam.frameBuffer(w, h);                          // gray8 buffer inside the wasm heap
//   HEAPU8.set(gray, ptr); const state = slam.track(t);          // t in seconds
//   slam.poseTwc()   -> Float32Array(16), column-major camera-to-world
//   slam.trackedPoints() -> Float32Array(3N), world coordinates of the map points seen in the last frame
//   viewer: frameKeyPoints(), mapPoints(), keyFrames(), graphEdges(), initMatches(), mapStats() (see below)
//   Typed arrays are views into the wasm heap: copy them (slice()) before the next call.
//
//   const cal = new Module.Calibrator();                        // camera calibration (web/calib.html), no SLAM needed
//   cal.frameBuffer(w, h); cal.detect(cols, rows) -> corners; cal.addView(); cal.solve(cols, rows) -> intrinsics

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <chrono>
#include <cmath>
#include <memory>
#include <unordered_set>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "Atlas.h"
#include "LocalMapping.h"
#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"
#include "Tracking.h"
#include "System.h"
#include "Thirdparty/DBoW2/DUtils/Random.h"
#include "WebConfig.h"

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
    int state = sys_->GetTrackingState();
    if (state == ORB_SLAM3::Tracking::OK && sys_->WebTracker()->mbWebKLTFrame) state = 5;  // OK, tracked by KLT
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

  // ---------------------------------------------------------------- viewer data
  // What ORB-SLAM3's FrameDrawer / MapDrawer draw upstream. Call between frames (same thread as track()).

  // Current frame keypoints in input-image pixels: (x, y, flag) per keypoint. flag: 0 = no map point,
  // 1 = matched to a map point (green in ORB-SLAM3's frame view), 2 = visual-odometry match (point with no
  // observations yet), 3 = initialisation match (state "not initialized").
  val frameKeyPoints() {
    kps_.clear();
    nMatches_ = 0;
    if (!sys_) return val(typed_memory_view(0, kps_.data()));
    ORB_SLAM3::Tracking* tr = sys_->WebTracker();
    const ORB_SLAM3::Frame& F = tr->mCurrentFrame;
    const float inv = 1.f / sys_->GetImageScale();
    std::vector<char> ini;
    if (tr->mState == ORB_SLAM3::Tracking::NOT_INITIALIZED) {
      ini.assign(F.mvKeys.size(), 0);
      for (int m : tr->mvIniMatches) if (m >= 0 && m < (int)ini.size()) ini[m] = 1;
    }
    for (size_t i = 0; i < F.mvKeys.size(); ++i) {
      float flag = 0.f;
      if (!ini.empty()) {
        flag = ini[i] ? 3.f : 0.f;
      } else if (i < F.mvpMapPoints.size() && F.mvpMapPoints[i] && !F.mvbOutlier[i]) {
        flag = F.mvpMapPoints[i]->Observations() > 0 ? 1.f : 2.f;
        if (flag == 1.f) ++nMatches_;
      }
      kps_.insert(kps_.end(), {F.mvKeys[i].pt.x * inv, F.mvKeys[i].pt.y * inv, flag});
    }
    return val(typed_memory_view(kps_.size(), kps_.data()));
  }

  // Map points of the active map: (x, y, z, local) per point; local = 1 for the local map around the camera
  // (red in ORB-SLAM3's map view), else 0 (black).
  val mapPoints() {
    mps_.clear();
    if (!sys_) return val(typed_memory_view(0, mps_.data()));
    ORB_SLAM3::Map* map = sys_->WebAtlas()->GetCurrentMap();
    if (!map) return val(typed_memory_view(0, mps_.data()));
    const std::vector<ORB_SLAM3::MapPoint*> all = map->GetAllMapPoints();
    const std::vector<ORB_SLAM3::MapPoint*> ref = map->GetReferenceMapPoints();
    std::unordered_set<ORB_SLAM3::MapPoint*> local(ref.begin(), ref.end());
    mps_.reserve(all.size() * 4);
    for (ORB_SLAM3::MapPoint* mp : all) {
      if (!mp || mp->isBad()) continue;
      const Eigen::Vector3f p = mp->GetWorldPos();
      mps_.insert(mps_.end(), {p.x(), p.y(), p.z(), local.count(mp) ? 1.f : 0.f});
    }
    return val(typed_memory_view(mps_.size(), mps_.data()));
  }

  // Keyframes of the active map: 12 floats each, Twc as [R row-major (9), t (3)].
  val keyFrames() {
    kfs_.clear();
    if (!sys_) return val(typed_memory_view(0, kfs_.data()));
    ORB_SLAM3::Map* map = sys_->WebAtlas()->GetCurrentMap();
    if (!map) return val(typed_memory_view(0, kfs_.data()));
    for (ORB_SLAM3::KeyFrame* kf : map->GetAllKeyFrames()) {
      if (!kf || kf->isBad()) continue;
      const Sophus::SE3f Twc = kf->GetPoseInverse();
      const Eigen::Matrix3f R = Twc.rotationMatrix();
      const Eigen::Vector3f t = Twc.translation();
      for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) kfs_.push_back(R(r, c));
      kfs_.insert(kfs_.end(), {t.x(), t.y(), t.z()});
    }
    return val(typed_memory_view(kfs_.size(), kfs_.data()));
  }

  // Keyframe graph as line segments (6 floats each: two camera centres), as MapDrawer::DrawKeyFrames(bDrawGraph):
  // covisibility (weight >= 100), spanning tree (parent) and loop edges.
  val graphEdges() {
    edges_.clear();
    if (!sys_) return val(typed_memory_view(0, edges_.data()));
    ORB_SLAM3::Map* map = sys_->WebAtlas()->GetCurrentMap();
    if (!map) return val(typed_memory_view(0, edges_.data()));
    auto seg = [this](const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
      edges_.insert(edges_.end(), {a.x(), a.y(), a.z(), b.x(), b.y(), b.z()});
    };
    for (ORB_SLAM3::KeyFrame* kf : map->GetAllKeyFrames()) {
      if (!kf || kf->isBad()) continue;
      const Eigen::Vector3f Ow = kf->GetCameraCenter();
      for (ORB_SLAM3::KeyFrame* c : kf->GetCovisiblesByWeight(100))
        if (c && !c->isBad() && c->mnId < kf->mnId) seg(Ow, c->GetCameraCenter());
      if (ORB_SLAM3::KeyFrame* p = kf->GetParent()) seg(Ow, p->GetCameraCenter());
      for (ORB_SLAM3::KeyFrame* l : kf->GetLoopEdges())
        if (l && l->mnId < kf->mnId) seg(Ow, l->GetCameraCenter());
    }
    return val(typed_memory_view(edges_.size(), edges_.data()));
  }

  // Initialisation matches (state "not initialized"): (x0, y0, x1, y1) from the initial frame to the current one.
  val initMatches() {
    ini_.clear();
    if (!sys_) return val(typed_memory_view(0, ini_.data()));
    ORB_SLAM3::Tracking* tr = sys_->WebTracker();
    if (tr->mState != ORB_SLAM3::Tracking::NOT_INITIALIZED) return val(typed_memory_view(0, ini_.data()));
    const float inv = 1.f / sys_->GetImageScale();
    const auto& k0 = tr->mInitialFrame.mvKeys;
    const auto& k1 = tr->mCurrentFrame.mvKeys;
    for (size_t i = 0; i < tr->mvIniMatches.size() && i < k0.size(); ++i) {
      const int m = tr->mvIniMatches[i];
      if (m < 0 || m >= (int)k1.size()) continue;
      ini_.insert(ini_.end(), {k0[i].pt.x * inv, k0[i].pt.y * inv, k1[m].pt.x * inv, k1[m].pt.y * inv});
    }
    return val(typed_memory_view(ini_.size(), ini_.data()));
  }

  // [keyframes in active map, map points in active map, map-point matches in the last frame (frameKeyPoints()),
  //  number of maps in the atlas]
  val mapStats() {
    stats_[0] = stats_[1] = stats_[3] = 0;
    stats_[2] = nMatches_;
    if (sys_) {
      ORB_SLAM3::Atlas* atlas = sys_->WebAtlas();
      if (ORB_SLAM3::Map* map = atlas->GetCurrentMap()) {
        stats_[0] = (int)map->KeyFramesInMap();
        stats_[1] = (int)map->MapPointsInMap();
      }
      stats_[3] = atlas->CountMaps();
    }
    return val(typed_memory_view(4, stats_));
  }

  // Live per-stage profile (AR page, ?prof=1). Always: [KLT frames, ORB frames, KLT ms mean, LK pyramid ms per frame].
  // Profiling build (WEB_PROFILE = ORB-SLAM3's REGISTER_TIMES) adds the running means (ms) of the stage timers:
  // tracking [ORB extraction, pose prediction, local-map tracking, new keyframe, total, frames] and local mapping
  // [KF insertion, map-point culling, map-point creation, local BA, KF culling, total, keyframes]. Diagnostic: the
  // mapping vectors are read while the mapping thread may append to them.
  val profile() {
    prof_.clear();
    if (!sys_) return val(typed_memory_view(0, prof_.data()));
    ORB_SLAM3::Tracking* t = sys_->WebTracker();
    const long kf = t->mnWebKLTFrames, of = t->mnWebORBFrames;
    prof_ = {(double)kf, (double)of, kf ? t->mdWebKLTms / kf : 0.0, t->mdWebKLTPyrMs / std::max(1L, kf + of)};
#ifdef REGISTER_TIMES
    auto mean = [](const std::vector<double>& v) {
      const size_t n = v.size();
      double s = 0;
      for (size_t i = 0; i < n; ++i) s += v[i];
      return n ? s / n : 0.0;
    };
    ORB_SLAM3::LocalMapping* m = sys_->WebLocalMapper();
    prof_.insert(prof_.end(), {mean(t->vdORBExtract_ms), mean(t->vdPosePred_ms), mean(t->vdLMTrack_ms), mean(t->vdNewKF_ms),
                               mean(t->vdTrackTotal_ms), (double)t->vdTrackTotal_ms.size(),
                               mean(m->vdKFInsert_ms), mean(m->vdMPCulling_ms), mean(m->vdMPCreation_ms), mean(m->vdLBA_ms),
                               mean(m->vdKFCulling_ms), mean(m->vdLMTotal_ms), (double)m->vdLMTotal_ms.size()});
#endif
    return val(typed_memory_view(prof_.size(), prof_.data()));
  }

  void reset() { if (sys_) sys_->ResetActiveMap(); }
  // AR page: while an anchor is placed, a tracking loss keeps relocalising in the active map instead of resetting it
  // or starting a new one (WebConfig::keepMap). Called between frames, on the thread that calls track().
  void setKeepMap(bool on) { ORB_SLAM3::webConfig().keepMap = on ? 1 : 0; }
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
  std::vector<float> pts_, kps_, mps_, kfs_, edges_, ini_;
  std::vector<double> prof_;
  int nMatches_ = 0;
  int stats_[4] = {0, 0, 0, 0};
  double lastMs_ = 0;
};

// Camera calibration from chessboard views (web/calib.html): OpenCV findChessboardCorners + calibrateCamera, the
// pinhole + radial-tangential model that ORB-SLAM3's "PinHole" camera uses (k1 k2 p1 p2; k3 fixed at 0, as in
// ORB-SLAM3's settings files). Corners are in pixels of the processed image (the same stream size as the AR page).
class Calibrator {
 public:
  uintptr_t frameBuffer(int w, int h) {
    if (gray_.cols != w || gray_.rows != h) gray_.create(h, w, CV_8UC1);
    return reinterpret_cast<uintptr_t>(gray_.data);
  }

  // Chessboard with cols x rows inner corners in the frame buffer: (x, y) per corner, empty if not found.
  val detect(int cols, int rows) {
    corners_.clear();
    const bool found = cv::findChessboardCorners(
        gray_, cv::Size(cols, rows), corners_,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
    if (found)
      cv::cornerSubPix(gray_, corners_, cv::Size(5, 5), cv::Size(-1, -1),
                       cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
    else
      corners_.clear();
    out_.clear();
    for (const cv::Point2f& p : corners_) out_.insert(out_.end(), {p.x, p.y});
    return val(typed_memory_view(out_.size(), out_.data()));
  }

  // Keep the corners of the last successful detect() as a calibration view; returns the number of views.
  int addView() {
    if (!corners_.empty()) {
      views_.push_back(corners_);
      size_ = gray_.size();
    }
    return (int)views_.size();
  }
  void clear() { views_.clear(); }

  // [rms, fx, fy, cx, cy, k1, k2, p1, p2, then the RMS reprojection error of each view] (pixels); empty if < 4 views.
  val solve(int cols, int rows) {
    res_.clear();
    if (views_.size() < 4) return val(typed_memory_view(0, res_.data()));
    std::vector<cv::Point3f> board;
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c) board.emplace_back((float)c, (float)r, 0.f);  // square size 1: intrinsics only
    std::vector<std::vector<cv::Point3f>> obj(views_.size(), board);
    cv::Mat K, D;
    std::vector<cv::Mat> rvecs, tvecs;
    const double rms = cv::calibrateCamera(obj, views_, size_, K, D, rvecs, tvecs, cv::CALIB_FIX_K3);
    res_ = {rms, K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2),
            D.at<double>(0), D.at<double>(1), D.at<double>(2), D.at<double>(3)};
    for (size_t i = 0; i < views_.size(); ++i) {
      std::vector<cv::Point2f> proj;
      cv::projectPoints(board, rvecs[i], tvecs[i], K, D, proj);
      double e2 = 0;
      for (size_t j = 0; j < proj.size(); ++j) {
        const cv::Point2f d = proj[j] - views_[i][j];
        e2 += d.dot(d);
      }
      res_.push_back(std::sqrt(e2 / proj.size()));
    }
    return val(typed_memory_view(res_.size(), res_.data()));
  }

  // Drop view i (e.g. an outlier from solve()).
  int removeView(int i) {
    if (i >= 0 && i < (int)views_.size()) views_.erase(views_.begin() + i);
    return (int)views_.size();
  }

 private:
  cv::Mat gray_;
  cv::Size size_;
  std::vector<cv::Point2f> corners_;
  std::vector<std::vector<cv::Point2f>> views_;
  std::vector<float> out_;
  std::vector<double> res_;
};

EMSCRIPTEN_BINDINGS(orbslam3_web) {
  class_<Calibrator>("Calibrator")
      .constructor<>()
      .function("frameBuffer", &Calibrator::frameBuffer)
      .function("detect", &Calibrator::detect)
      .function("addView", &Calibrator::addView)
      .function("removeView", &Calibrator::removeView)
      .function("clear", &Calibrator::clear)
      .function("solve", &Calibrator::solve);
  class_<WebSlam>("WebSlam")
      .constructor<>()
      .function("init", &WebSlam::init)
      .function("frameBuffer", &WebSlam::frameBuffer)
      .function("track", &WebSlam::track)
      .function("poseTwc", &WebSlam::poseTwc)
      .function("lastTrackMs", &WebSlam::lastTrackMs)
      .function("trackedPoints", &WebSlam::trackedPoints)
      .function("frameKeyPoints", &WebSlam::frameKeyPoints)
      .function("mapPoints", &WebSlam::mapPoints)
      .function("keyFrames", &WebSlam::keyFrames)
      .function("graphEdges", &WebSlam::graphEdges)
      .function("initMatches", &WebSlam::initMatches)
      .function("mapStats", &WebSlam::mapStats)
      .function("reset", &WebSlam::reset)
      .function("profile", &WebSlam::profile)
      .function("setKeepMap", &WebSlam::setKeepMap)
      .function("shutdown", &WebSlam::shutdown)
      .function("finishAndSave", &WebSlam::finishAndSave);
}
