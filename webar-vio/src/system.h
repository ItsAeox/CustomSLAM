
#pragma once

#include <vector>
#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/features2d.hpp>

class System {
public:
  System();

  // Initialize image size and camera intrinsics
  void init(int width, int height, double fx, double fy, double cx, double cy);

  // img: RGBA or Gray buffer; ts in seconds
  // stride is not required; buffer is assumed tightly packed per row.
  void feedFrame(const uint8_t* img, double ts, int width, int height, bool isRGBA);

  // Flattened [x0,y0, x1,y1, ...] in FULL-RES pixel coords (origin top-left)
  std::vector<double> getPoints2D() const;

  int getNumKeypoints() const { return static_cast<int>(ptsCur_.size()); }
  int getTrackState()   const { return trackingState_; } // 0=uninit,1=tracking
  double getLastTS()    const { return lastTS_; }
  double getLastTotalMS() const { return t_last_total_ms_; }
  double getLastKltMS()   const { return t_last_klt_ms_; }
  double getLastSeedMS()  const { return t_last_seed_ms_; }
  int maxReturnPts_ = 200;
  double getLastMeanY() const { return lastMeanY_; }
  std::array<int,2> getLastProcWH() const { return { curProc_.cols, curProc_.rows }; }
  enum class TrackerType { KLT = 0, ORB = 1, HYBRID = 2 };

  void setTrackerType(int t) {
    if      (t == 2) trackerType_ = TrackerType::HYBRID;
    else if (t == 1) trackerType_ = TrackerType::ORB;
    else             trackerType_ = TrackerType::KLT;
  }
  int getTrackerType() const {
    switch (trackerType_) {
      case TrackerType::HYBRID: return 2;
      case TrackerType::ORB:    return 1;
      default:                  return 0;
    }
  }  

  double getLastOrbMS() const { return t_last_orb_ms_; }
  void setHybridEveryN(int n) { hybridEveryN_ = std::max(1, n); }
  int  getHybridEveryN() const { return hybridEveryN_; }
  bool getRanOrbThisFrame() const { return ranOrbThisFrame_; }
  int  getOrbKFCount() const { return orbKFCount_; }
  uint64_t getHybFrameIdx() const { return hybFrameIdx_; }
  // --- VO / path telemetry ---
  // Returns a flattened [x0,z0, x1,z1, ...] in arbitrary scale (monocular)
  std::vector<float> getPathXZ() const;
  // --- Pose accessors for UI ---
  // World position (x,y,z)
  std::array<double,3> getTwc() const {
    return { twc_[0], twc_[1], twc_[2] };
  }
  // Heading (yaw on XZ), pitch, roll in radians (right-handed, OpenCV: x right, y down, z forward)
  // yaw: angle of forward (Rwc_.col(2)) projected on XZ
  // pitch: elevation of forward
  // roll: rotation around forward axis using right/up
  std::array<double,3> getYPR() const;
  // --- E/H gate telemetry (for HUD/logging) ---
  // model: 0=NONE, 1=E, 2=H
  int    getEHModel()       const { return ehModel_; }
  int    getEHInliersE()    const { return ehInliersE_; }
  int    getEHInliersH()    const { return ehInliersH_; }
  double getEHParallaxDeg() const { return ehParallaxDeg_; }
  // Mapping stats (public getters)
  int getNumKFs() const { return (int)kfs_.size(); }
  int getNumMPs() const { return (int)mps_.size(); }


private:
  int   procScale_      = 2;        // 2 => process at half-res (major speedup)
  int   kltWin_         = 21;
  int   kltLevels_      = 3;
  float kltErrMax_      = 20.f;     // LK per-point error gate
  float fbMax_          = 2.0f;     // forward-backward gate (pixels)
  int   cellSize_       = 28;       // grid cell size for seeding (processing scale) ***** Scale DOWN 
  int   targetKps_      = 200;      // feature budget at processing scale ***** Scale UP
  int   descEveryN_     = 8;        // ORB compute cadence (frames); 0 disables
  int   maxTracks_    =200;  // hard ceiling after tracking+reseeding
  double t_last_total_ms_ = 0.0;
  double t_last_klt_ms_   = 0.0;
  double t_last_seed_ms_  = 0.0;
  double lastMeanY_ = -1.0;

  cv::TermCriteria termcrit_{cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01};

  // Image geometry / intrinsics
  int imgW_ = 0, imgH_ = 0;
  double fx_ = 0., fy_ = 0., cx_ = 0., cy_ = 0.;

  // Working images (reused every frame)
  cv::Mat prevGray_, curGray_;   // full-res gray
  cv::Mat prevProc_, curProc_;   // downscaled gray (working resolution)

  // Pyramids (processing scale)
  std::vector<cv::Mat> pyrPrev_, pyrCur_;

  // Tracks (processing scale coordinates)
  std::vector<cv::Point2f> ptsPrev_, ptsCur_;
  int frameCount_ = 0;

  // State
  int trackingState_ = 0; // 0=uninitialized, 1=tracking
  double lastTS_ = 0.0;

  TrackerType trackerType_ = TrackerType::KLT;

  // Hybrid config/state
  int hybridEveryN_ = 8;               // default: run ORB every 8th frame
  uint64_t hybFrameIdx_ = 0;            // increments each feedFrame
  bool     ranOrbThisFrame_ = false;
  uint64_t lastOrbKF_ = 0;       // last frame idx that ran ORB
  int      orbKFCount_ = 0;      // number of ORB keyframes run

  // ORB objects + frame-to-frame state
  cv::Ptr<cv::ORB> orb_;
  std::vector<cv::KeyPoint> orbPrevKps_, orbCurKps_;
  cv::Mat orbPrevDesc_, orbCurDesc_;

  // Tunables (reasonable defaults; tweak later)
  int   orbNFeatures_     = 600;
  float orbScaleFactor_   = 1.2f;
  int   orbNLevels_       = 4;
  int   orbEdgeThreshold_ = 31;
  int   orbFirstLevel_    = 0;
  int   orbWtaK_          = 2;
  cv::ORB::ScoreType orbScore_ = cv::ORB::HARRIS_SCORE;
  int   orbPatchSize_     = 31;
  int   orbFastThreshold_ = 20;

  // ===== Mapping state (SFM → VIO) =====
  struct Keyframe {
    int id = -1;
    cv::Matx33d Rwc;   // world-from-camera
    cv::Vec3d   twc;
    std::vector<cv::KeyPoint> kps; // ORB keypoints at processing scale
    cv::Mat desc;                   // ORB descriptors (rows=kps)
  };
  struct MapPoint {
    cv::Vec3d Xw;
    cv::Mat   desc;      // 1x32 (cloned row) OR empty if unknown
    int       hostKF = -1;
    int       seen   = 0;
    int       found  = 0;
    float     invScale = 1.f; // quick gating by distance (optional)
  };

  bool mapInitialized_ = false;
  std::vector<Keyframe>  kfs_;
  std::vector<MapPoint>  mps_;
  int nextKFId_ = 0;

  // Working buffers reused each frame (avoid allocs in hot loop)
  std::vector<int>     pnp_indices_;     // indices into mps_
  std::vector<cv::Point2f> pnp_pixels_;  // matched 2D
  std::vector<cv::Point3f> pnp_points_;  // 3D (float for cv PnP)
  cv::Mat rvec_, tvec_;                  // current cam pose (cw) for PnP refine

  // KF policy
  double lastKFTs_ = 0.0;
  int    lastKFInliers_ = 0;

  // ====== API ======
  // 2-view init from 2D-2D (processing-scale points)
  bool tryTwoViewInit(const std::vector<cv::Point2f>& prevProcPts,
                      const std::vector<cv::Point2f>& curProcPts);

  // Per-frame 3D-2D tracking using MapPoints (fills Rwc_/twc_ on success)
  bool trackWithPnP();

  // KF insertion + triangulation vs last KF
  bool shouldInsertKF(int pnpInliers, double nowTs) const;
  void insertKeyframeAndTriangulate();

  // Helper: compute ORB at arbitrary pixel locations (processing scale)
  void computeORBAtPoints(const cv::Mat& img,
                          const std::vector<cv::Point2f>& pts,
                          cv::Mat& outDesc);

  // Project MapPoints and collect 3D-2D with small reprojection window
  int harvestPnpCorrespondences(float reprojThreshPx = 8.f, int maxTake = 500);

  // Utility: K (intrinsics) and its inverse at **full-res**
  inline cv::Matx33d K()  const { return cv::Matx33d(fx_, 0,  cx_,
                                                     0,  fy_, cy_,
                                                     0,  0,  1); }
  inline cv::Matx33d Ki() const {
    const double ix = 1.0/std::max(1e-9, fx_);
    const double iy = 1.0/std::max(1e-9, fy_);
    return cv::Matx33d(ix,0,-cx_*ix,  0,iy,-cy_*iy,  0,0,1);
  }

  // Timing
  double t_last_orb_ms_   = 0.0;
  // ===== Visual Odometry (VO) state =====
  // World pose of camera: Rwc_ (3x3), twc_ (3x1); start at identity
  cv::Matx33d Rwc_ = cv::Matx33d::eye();
  cv::Vec3d   twc_ = cv::Vec3d(0,0,0);

  // History of world positions for drawing (x,z used for top-down path)
  std::vector<cv::Point3f> path_; // (x,y,z), push one per frame

  // Integrate relative pose from Essential-matrix inlier correspondences
  void integrateVO_E(const std::vector<cv::Point2f>& prevProcPts,
                     const std::vector<cv::Point2f>& curProcPts);

  // Utility already present:
  // void toFullResPixels(const std::vector<cv::Point2f>& procPts,
  //                      std::vector<cv::Point2f>& fullResPx) const;

  // ===== E/H model gate state (unique names: eh*) =====
  // 0 = NONE, 1 = E (Essential), 2 = H (Homography)
  int    ehModel_        = 0;
  int    ehInliersE_     = 0;
  int    ehInliersH_     = 0;
  double ehParallaxDeg_  = 0.0;

  // Compute E vs H on corresponding point pairs (processing-scale coords)
  void   runEvsHGate(const std::vector<cv::Point2f>& prevProcPts,
                     const std::vector<cv::Point2f>& curProcPts);

  // Utility: build full-res pixel pairs from processing-scale points
  void   toFullResPixels(const std::vector<cv::Point2f>& procPts,
                         std::vector<cv::Point2f>& fullResPx) const;

};

